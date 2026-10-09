#!/usr/bin/env python3
"""
NSE -> Telegram reporter - self-contained, fast, fail-fast (no nse_engine.py needed).

Per run (all fetches in parallel, hard 40 s deadline):
  Yahoo 15m x3 (charts + spot)   Yahoo daily x3 (only once per day, cached)   NSE option chain x2

Sends: 1 album of 3 charts (NIFTY, BANKNIFTY, SENSEX) + 1 text message with OI change + link buttons.
Photos always go straight to Telegram; text goes via N8N_WEBHOOK_URL if set, else straight to Telegram.

  python nse_telegram.py             # only 09:15-15:30 IST Mon-Fri
  python nse_telegram.py --force     # ignore market hours
  python nse_telegram.py --dry-run   # print text, save charts to ./charts, send nothing
"""
import argparse
import datetime as dt
import html
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
BASE = "https://www.nseindia.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
DEADLINE = 40          # seconds to wait for all fetches; slow sources are skipped
STATE = Path(os.getenv("STATE_FILE", "state/snapshot.json"))
YF_SYM = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
HAS_CHAIN = ("NIFTY", "BANKNIFTY")   # Sensex options are on BSE, not NSE
TV = {"NIFTY": "NSE%3ANIFTY", "BANKNIFTY": "NSE%3ABANKNIFTY", "SENSEX": "BSE%3ASENSEX"}
STALE_AFTER = 20 * 60
MAX_MSG = 3900
BIAS_ICON = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}

# chart palette
BG, UP, DN = "#F4F7FB", "#1B8A4B", "#D64545"
RED, GREEN, BLUE, PURPLE, INK = "#D64545", "#1B8A4B", "#1F5FBF", "#7A3FB5", "#14213D"


def market_open(now=None):
    n = now or dt.datetime.now(IST)
    return n.weekday() < 5 and dt.time(9, 15) <= n.time() <= dt.time(15, 30)


# ----------------------------------------------------------------- fast data fetchers
def yf(sym, interval, rng):
    r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/" + quote(sym),
                     params=dict(interval=interval, range=rng), headers={"User-Agent": UA}, timeout=(4, 8))
    j = r.json()["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    a = {k: np.array(q[k], dtype=float) for k in ("open", "high", "low", "close")}
    t = np.array(j["timestamp"], dtype=float)
    ok = np.isfinite(a["close"]) & np.isfinite(a["high"]) & np.isfinite(a["low"]) & np.isfinite(a["open"])
    return dict(t=t[ok], o=a["open"][ok], h=a["high"][ok], l=a["low"][ok], c=a["close"][ok])


def analyse_chain(j):
    rec = j["records"]
    spot = float(rec["underlyingValue"])
    rows = rec["data"]

    def rexp(r):
        return (r.get("expiryDate") or r.get("expiryDates")
                or (r.get("CE") or {}).get("expiryDate") or (r.get("PE") or {}).get("expiryDate")
                or (rec.get("expiryDates") or ["NA"])[0])

    def pdate(x):
        for f in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return dt.datetime.strptime(x, f)
            except ValueError:
                pass
        return dt.datetime.max

    ex = min({rexp(r) for r in rows}, key=pdate)
    rows = sorted((r for r in rows if rexp(r) == ex), key=lambda r: r["strikePrice"])
    K = np.array([r["strikePrice"] for r in rows], float)
    g = lambda s, f: np.array([(r.get(s) or {}).get(f) or 0 for r in rows], float)
    co, cc = g("CE", "openInterest"), g("CE", "changeinOpenInterest")
    po, pc = g("PE", "openInterest"), g("PE", "changeinOpenInterest")
    ia = int(abs(K - spot).argmin())
    near = slice(max(0, ia - 5), ia + 6)
    pain = ((co[None, :] * np.maximum(K[:, None] - K[None, :], 0)).sum(1)
            + (po[None, :] * np.maximum(K[None, :] - K[:, None], 0)).sum(1))

    def walls(mask, o, ch):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-o[idx])][:3]
        return [dict(strike=int(K[i]), oi=float(o[i]), chg=float(ch[i])) for i in idx]

    sup, res = walls(K <= spot, po, pc), walls(K >= spot, co, cc)
    bars = [dict(k=int(K[i]), ce=float(co[i]), pe=float(po[i])) for i in range(max(0, ia - 8), min(ia + 9, len(K)))]
    pcr, pcr_atm = po.sum() / max(co.sum(), 1), po[near].sum() / max(co[near].sum(), 1)
    ce_d, pe_d = float(cc.sum()), float(pc.sum())
    sc = 0
    sc += 1 if pcr_atm > 1.1 else -1 if pcr_atm < 0.9 else 0
    sc += 1 if pe_d > ce_d * 1.2 else -1 if ce_d > pe_d * 1.2 else 0
    if sup and res:
        sc += (sup[0]["chg"] > 0 and res[0]["chg"] < 0) - (res[0]["chg"] > 0 and sup[0]["chg"] < 0)
    sc += 1 if spot > K[pain.argmin()] else -1
    return dict(spot=spot, atm=int(K[ia]), expiry=ex, pcr=round(float(pcr), 2), max_pain=int(K[pain.argmin()]),
                bias="BULLISH" if sc >= 2 else "BEARISH" if sc <= -2 else "NEUTRAL", score=int(sc),
                ce_d=ce_d, pe_d=pe_d, ce_tot=float(co.sum()), pe_tot=float(po.sum()), sup=sup, res=res, bars=bars)


def nse_chain(sym):
    """Fail-fast NSE option chain: one cookie warm-up, then legacy endpoint, then v3. Max ~25 s."""
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json,text/plain,*/*",
                      "Accept-Language": "en-US,en;q=0.9", "Referer": BASE + "/option-chain"})
    s.get(BASE + "/option-chain", timeout=(4, 6))

    def get(path, **p):
        r = s.get(BASE + path, params=p, timeout=(4, 7))
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        return r.json()

    last = None
    try:
        j = get("/api/option-chain-indices", symbol=sym)
        if j.get("records", {}).get("data"):
            return analyse_chain(j)
        last = RuntimeError("empty legacy chain")
    except Exception as e:
        last = e
    try:
        ex = get("/api/option-chain-contract-info", symbol=sym)["expiryDates"][0]
        j = get("/api/option-chain-v3", type="Indices", symbol=sym, expiry=ex)
        if j.get("records", {}).get("data"):
            return analyse_chain(j)
        last = RuntimeError("empty v3 chain")
    except Exception as e:
        last = e
    raise RuntimeError(f"{sym}: {last}")


# ----------------------------------------------------------------- indicators
def ema(x, n):
    a, o = 2 / (n + 1), np.empty_like(x)
    o[0] = x[0]
    for i in range(1, len(x)):
        o[i] = a * x[i] + (1 - a) * o[i - 1]
    return o


def swings(h, l, k):
    hi, lo = [], []
    for i in range(k, len(h) - k):
        if h[i] == h[i - k:i + k + 1].max():
            hi.append(i)
        if l[i] == l[i - k:i + k + 1].min():
            lo.append(i)
    return np.array(hi, int), np.array(lo, int)


# ----------------------------------------------------------------- formatting
def n0(x):
    return f"{x:,.0f}"


def lk(x):
    return f"{x / 1e5:.1f}L"


def kd(x):
    return "—" if x is None else f"{x / 1e3:+.1f}k"


def pct(a, b):
    return (a - b) / b * 100 if b else 0.0


def arrow(chg):
    return f"{'🟢▲' if chg >= 0 else '🔴▼'}{abs(chg):.2f}%"


# ----------------------------------------------------------------- daily levels
def daily_levels(d):
    """PDH/PDL/PWH/PWL, Fibonacci, EMA 20/50/200 - same logic as nse_engine.build()."""
    today = dt.datetime.now(IST).date()
    last_today = dt.datetime.fromtimestamp(d["t"][-1], IST).date() == today
    i = -2 if market_open() and last_today else -1
    n = len(d["c"]) + i + 1
    lv = dict(
        PDH=float(d["h"][i]), PDL=float(d["l"][i]),
        PWH=float(d["h"][max(0, n - 5):n].max()), PWL=float(d["l"][max(0, n - 5):n].min()),
        live=float(d["c"][-1]),
        prev_close=float(d["c"][-2] if last_today and len(d["c"]) > 1 else d["c"][-1]),
    )
    h, l = d["h"][-120:], d["l"][-120:]
    ih, il = swings(h, l, 5)
    fib = {}
    if len(ih) and len(il):
        fh, fl = h[ih[-1]], l[il[-1]]
        for r in (0.382, 0.5, 0.618):
            fib[f"{r * 100:g}"] = float(fh - (fh - fl) * r if il[-1] < ih[-1] else fl + (fh - fl) * r)
    lv["fib"] = fib
    lv["ema"] = {str(k): float(ema(d["c"], k)[-1]) for k in (20, 50, 200)}
    return lv


def last_days(b, n=3):
    """Keep only the last n trading sessions of an intraday series."""
    day = np.array([dt.datetime.fromtimestamp(t, IST).date().toordinal() for t in b["t"]])
    keep = sorted(set(day.tolist()))[-n:]
    m = np.isin(day, keep)
    out = {k: v[m] for k, v in b.items()}
    out["day"] = day[m]
    return out


# ----------------------------------------------------------------- snapshot (before vs now)
def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def usable_prev(state, name):
    p = state.get(name)
    if not p:
        return None, 0
    age = time.time() - p.get("ts", 0)
    same_day = p.get("date") == dt.datetime.now(IST).strftime("%Y-%m-%d")
    return (p, max(1, round(age / 60))) if same_day and age <= STALE_AFTER else (None, 0)


def snapshot(oc):
    return dict(
        ts=time.time(), date=dt.datetime.now(IST).strftime("%Y-%m-%d"), spot=oc["spot"],
        pcr=oc["pcr"], ce_tot=oc["ce_tot"], pe_tot=oc["pe_tot"],
        strikes={str(b["k"]): [b["ce"], b["pe"]] for b in oc["bars"]},
        walls={f"{s}{w['strike']}": w["oi"] for s in ("sup", "res") for w in oc[s]},
    )


# ----------------------------------------------------------------- chart
def trendline(xs, ps, side, tol=0.0015):
    """Best line through 2+ swing points: rising lows (sup) or falling highs (res)."""
    pts, best = list(zip(xs[-8:], ps[-8:])), None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (x1, y1), (x2, y2) = pts[i], pts[j]
            if x2 == x1 or (side == "sup" and y2 <= y1) or (side == "res" and y2 >= y1):
                continue
            m = (y2 - y1) / (x2 - x1)
            t = sum(abs(y - (y1 + m * (x - x1))) <= tol * y for x, y in pts)
            if t >= 2 and (best is None or (t, x2) > (best[0], best[1])):
                best = (t, x2, x1, y1, m)
    return best


def spread_labels(items, gap):
    """items: [(y, text, color)] -> nudge y so labels don't overlap."""
    items = sorted(items, key=lambda t: t[0])
    out, last = [], -1e18
    for y, txt, col in items:
        yy = max(y, last + gap)
        out.append((y, yy, txt, col))
        last = yy
    return out


def make_chart(name, b, lv, oc, spot, chg):
    o, h, l, c, day = b["o"], b["h"], b["l"], b["c"], b["day"]
    n = len(c)
    x = np.arange(n)
    lo, hi = float(l.min()), float(h.max())

    zones, lines = [], []   # zones: (price, color, label)   lines: (price, color, label)
    if oc:
        zones += [(w["strike"], RED, f"R {w['strike']:,}  {lk(w['oi'])}") for w in oc["res"]]
        zones += [(w["strike"], GREEN, f"S {w['strike']:,}  {lk(w['oi'])}") for w in oc["sup"]]
    if lv:
        lines += [(lv["PDH"], "#5B6B8C", f"PDH {n0(lv['PDH'])}"), (lv["PDL"], "#5B6B8C", f"PDL {n0(lv['PDL'])}"),
                  (lv["PWH"], "#8A94A6", f"PWH {n0(lv['PWH'])}"), (lv["PWL"], "#8A94A6", f"PWL {n0(lv['PWL'])}")]
    near = lambda p: lo - 0.006 * spot <= p <= hi + 0.006 * spot
    zones = [z for z in zones if near(z[0])]
    lines = [z for z in lines if near(z[0])]
    ys = [lo, hi] + [z[0] for z in zones + lines]
    ylo, yhi = min(ys), max(ys)
    pad = (yhi - ylo) * 0.06
    ylo, yhi = ylo - pad, yhi + pad

    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # resistance / support zones (bands like the reference chart)
    bw = spot * 0.00035
    labels = []
    for p, col, lab in zones:
        ax.axhspan(p - bw, p + bw, color=col, alpha=0.22, lw=0, zorder=1)
        ax.axhline(p, color=col, lw=0.7, alpha=0.7, zorder=1)
        labels.append((p, lab, col))
    for p, col, lab in lines:
        ax.axhline(p, color=col, lw=0.9, ls=(0, (4, 3)), alpha=0.9, zorder=1)
        labels.append((p, lab, col))

    # candles
    colors = [UP if cc >= oo else DN for oo, cc in zip(o, c)]
    ax.vlines(x, l, h, colors=colors, lw=0.8, zorder=3)
    body = np.maximum(np.abs(c - o), spot * 0.00005)
    ax.bar(x, body, bottom=np.minimum(o, c), width=0.68, color=colors, zorder=4)

    # trendlines (15m swings over the 3 sessions)
    xe = n + 1
    ih, il = swings(h, l, 3)
    for side, ix, arr, col, lab in (("sup", il, l, BLUE, "Trend sup"), ("res", ih, h, PURPLE, "Trend res")):
        if len(ix) >= 2:
            t = trendline(ix, arr[ix], side)
            if t:
                touches, x2, x1, y1, m = t
                ax.plot([x1, xe], [y1 + m * (x1 - x1), y1 + m * (xe - x1)], color=col, lw=1.6, zorder=5)
                py = y1 + m * (n - 1 - x1)
                if ylo <= py <= yhi:
                    labels.append((py, f"{lab} {n0(py)} ({touches}x)", col))

    # day separators
    starts = [0] + [i for i in range(1, n) if day[i] != day[i - 1]]
    for s in starts:
        if s:
            ax.axvline(s - 0.5, color="#B8C2D4", lw=0.8, ls=":", zorder=0)
    ax.set_xticks(starts)
    ax.set_xticklabels([dt.date.fromordinal(int(day[s])).strftime("%d %b") for s in starts], fontsize=8, color=INK)

    # spot marker
    ax.axhline(spot, color=INK, lw=0.9, zorder=6)
    ax.annotate(f" {n0(spot)} ", xy=(n + 0.4, spot), xycoords="data", fontsize=8, fontweight="bold",
                color="white", va="center", ha="left", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc=INK, ec="none"))

    # right-hand labels (de-overlapped)
    gap = (yhi - ylo) * 0.05
    for y, yy, txt, col in spread_labels([(p, t, cl) for p, t, cl in labels if abs(p - spot) > gap * 0.4], gap):
        ax.text(n + 0.4, yy, txt, fontsize=7.5, color=col, va="center", ha="left", fontweight="bold", zorder=7)

    ax.set_xlim(-1, n + 15)
    ax.set_ylim(ylo, yhi)
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", labelsize=8, colors=INK, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#DCE3EE", lw=0.5, zorder=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{name}  {n0(spot)}  ({chg:+.2f}%)", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.text(1.0, 1.015, "15m · last 3 sessions", transform=ax.transAxes, ha="right", fontsize=8, color="#5B6B8C")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# ----------------------------------------------------------------- caption (under each chart)
def walls_str(walls):
    return " · ".join(f"{w['strike']:,} ({lk(w['oi'])})" for w in walls) or "n/a"


def caption(r, spot, chg):
    name, oc, lv = r["name"], r["oc"], r["lv"]
    L = [f"<b>{name}</b>  <b>{n0(spot)}</b>  {arrow(chg)}"]
    if oc:
        L.append(f"{BIAS_ICON.get(oc['bias'], '🟡')} {oc['bias']} · PCR {oc['pcr']} · Pain {oc['max_pain']:,} · Exp {oc['expiry']}")
    if lv:
        L.append(f"📅 PDH <b>{n0(lv['PDH'])}</b> · PDL <b>{n0(lv['PDL'])}</b> · PWH {n0(lv['PWH'])} · PWL {n0(lv['PWL'])}")
        if lv["fib"]:
            L.append("🌀 Fib " + " · ".join(f"{k} <b>{n0(v)}</b>" for k, v in lv["fib"].items()))
        e = lv["ema"]
        L.append(f"📉 EMA 20/50/200: {n0(e['20'])} · {n0(e['50'])} · {n0(e['200'])}")
    if oc:
        L.append(f"🔴 Resist (call OI): {walls_str(oc['res'])}")
        L.append(f"🟢 Support (put OI): {walls_str(oc['sup'])}")
    else:
        L.append("ℹ️ Option chain: n/a" + (" (BSE)" if name == "SENSEX" else " this run"))
    return "\n".join(L)[:1000]


# ----------------------------------------------------------------- text message (OI change)
def oi_table(oc, prev):
    bars = oc["bars"]
    ia = next((i for i, b in enumerate(bars) if b["k"] == oc["atm"]), len(bars) // 2)
    ps = (prev or {}).get("strikes", {})
    out = [f"{'Strike':<7}{'CE L':>6}{'Δk':>8}{'PE L':>7}{'Δk':>8}"]
    for b in bars[max(0, ia - 3): ia + 4]:
        p = ps.get(str(b["k"]))
        d1 = f"{(b['ce'] - p[0]) / 1e3:+.1f}" if p else "—"
        d2 = f"{(b['pe'] - p[1]) / 1e3:+.1f}" if p else "—"
        mark = "*" if b["k"] == oc["atm"] else " "
        out.append(f"{str(b['k']) + mark:<7}{b['ce'] / 1e5:>6.1f}{d1:>8}{b['pe'] / 1e5:>7.1f}{d2:>8}")
    return "<pre>" + html.escape("\n".join(out)) + "</pre>"


def wall_changes(walls, prev, tag, mins, icon, label):
    out = []
    for n, w in enumerate(walls, 1):
        before = (prev or {}).get("walls", {}).get(f"{tag}{w['strike']}")
        step = f" · {mins}m <b>{kd(w['oi'] - before)}</b>" if before is not None else ""
        out.append(f"{icon} {label}{n} <b>{w['strike']:,}</b> {lk(w['oi'])} · today {kd(w['chg'])}{step}")
    return "\n".join(out)


def explain(oc, lv, prev, mins):
    s, sup, res = oc["spot"], oc["sup"], oc["res"]
    L = []
    if sup and res:
        L.append(f"Range: put wall {sup[0]['strike']:,} (support) to call wall {res[0]['strike']:,} (resistance). "
                 f"Expect chop inside the band; max pain {oc['max_pain']:,} tends to pull price into expiry.")
    ce_d, pe_d = oc["ce_d"], oc["pe_d"]
    if pe_d > ce_d * 1.2:
        L.append(f"Today: put OI {kd(pe_d)} vs call {kd(ce_d)} - more put writing, supportive/bullish tilt.")
    elif ce_d > pe_d * 1.2:
        L.append(f"Today: call OI {kd(ce_d)} vs put {kd(pe_d)} - more call writing, upside capped/bearish tilt.")
    else:
        L.append(f"Today: call {kd(ce_d)} vs put {kd(pe_d)} - balanced writing, no clear side.")
    if prev:
        ds = s - prev["spot"]
        dce, dpe = oc["ce_tot"] - prev["ce_tot"], oc["pe_tot"] - prev["pe_tot"]
        if abs(ds) < 0.0003 * s:
            ds = 0
        hits = []
        if ds > 0 and dce < 0:
            hits.append("price up + call OI falling = short covering, upside momentum")
        if ds > 0 and dpe > 0:
            hits.append("price up + put OI rising = fresh put writing, support building")
        if ds > 0 and dce > 0 and dpe <= 0:
            hits.append("price up but call writers adding = rally being sold into")
        if ds < 0 and dpe < 0:
            hits.append("price down + put OI falling = long unwinding, support weakening")
        if ds < 0 and dce > 0:
            hits.append("price down + call OI rising = fresh call writing, sellers in control")
        if ds < 0 and dpe > 0 and dce <= 0:
            hits.append("dip being bought by put writers")
        L.append(f"Last {mins}m (spot {ds:+.0f}): " + ("; ".join(hits[:2]) if hits else "no clear OI shift."))
    if res:
        r = res[0]
        t = f"Resistance {r['strike']:,}: call OI {'rising - wall firm' if r['chg'] > 0 else 'falling - wall weakening'}"
        t += f"; sustained break opens {res[1]['strike']:,}" if len(res) > 1 else ""
        L.append(t + ".")
    if sup:
        x = sup[0]
        t = f"Support {x['strike']:,}: put OI {'rising - floor firm' if x['chg'] > 0 else 'falling - floor weakening'}"
        t += f"; breakdown opens {sup[1]['strike']:,}" if len(sup) > 1 else ""
        L.append(t + ".")
    pcr = oc["pcr"]
    L.append(f"PCR {pcr}: " + ("above 1.5, crowded longs - reversal risk." if pcr > 1.5 else
                              "above 1.2, put-heavy (supportive)." if pcr > 1.2 else
                              "below 0.7, call-heavy - upside capped." if pcr < 0.7 else
                              "below 0.9, call-leaning." if pcr < 0.9 else "neutral zone."))
    e20 = lv["ema"]["20"] if lv else None
    if e20:
        ctx = f"Spot {'above' if s > e20 else 'below'} daily EMA20 ({n0(e20)})"
        ctx += ", above PDH - intraday strength" if s > lv["PDH"] else ", below PDL - intraday weakness" if s < lv["PDL"] else ""
        L.append(ctx + ".")
    L.append(f"Net OI bias: {oc['bias']} (score {oc['score']:+d}). Rule-based reading, not advice.")
    return "\n".join("• " + html.escape(x) for x in L)


def oi_block(r, prev, mins):
    oc, lv, name = r["oc"], r["lv"], r["name"]
    when = f"vs {mins}m ago" if prev else "first run today"
    out = [f"{BIAS_ICON.get(oc['bias'], '🟡')} <b>{name}</b> · OI change <i>({when})</i>", oi_table(oc, prev)]
    if prev:
        dce, dpe = oc["ce_tot"] - prev["ce_tot"], oc["pe_tot"] - prev["pe_tot"]
        out.append(f"🔴 Calls {lk(oc['ce_tot'])} <b>{kd(dce)}</b> · 🟢 Puts {lk(oc['pe_tot'])} <b>{kd(dpe)}</b> ({mins}m)")
    out.append(f"📆 Today vs prev close: 🔴 calls <b>{kd(oc['ce_d'])}</b> · 🟢 puts <b>{kd(oc['pe_d'])}</b>")
    out.append(wall_changes(oc["res"], prev, "res", mins, "🔴", "R"))
    out.append(wall_changes(oc["sup"], prev, "sup", mins, "🟢", "S"))
    out.append("<blockquote expandable><b>What it means</b>\n" + explain(oc, lv, prev, mins) + "</blockquote>")
    return "\n".join(x for x in out if x)


def header(results):
    out = [f"🕒 <b>{dt.datetime.now(IST).strftime('%d %b %H:%M')} IST</b>"]
    for r in results:
        s = r["spot"]
        out.append(f"<b>{r['name']}</b> {n0(s)} {arrow(r['chg'])}" if s is not None else f"<b>{r['name']}</b> n/a")
    return "\n".join(out)


def chunk(blocks):
    msgs, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) + 2 > MAX_MSG:
            msgs.append(cur)
            cur = ""
        cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        msgs.append(cur)
    return msgs


def buttons():
    row = [{"text": f"📈 {n}", "url": f"https://www.tradingview.com/chart/?symbol={s}"} for n, s in TV.items()]
    return {"inline_keyboard": [[{"text": "📋 NSE Option Chain", "url": "https://www.nseindia.com/option-chain"}], row]}


# ----------------------------------------------------------------- telegram
def plain(s):
    return re.sub(r"<[^>]+>", "", html.unescape(s))


def send_text(msg, markup, token, chat, hook):
    payload = dict(text=msg, parse_mode="HTML", disable_web_page_preview=True)
    if markup:
        payload["reply_markup"] = markup
    url, extra = (hook, {}) if hook else (f"https://api.telegram.org/bot{token}/sendMessage", dict(chat_id=chat))
    r = requests.post(url, json={**payload, **extra}, timeout=30)
    if r.status_code == 400:   # formatting rejected -> resend as plain text
        payload.update(text=plain(msg))
        payload.pop("parse_mode")
        r = requests.post(url, json={**payload, **extra}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Telegram text {r.status_code}: {r.text[:300]}")


def send_photos(photos, token, chat):
    """photos: [(png_bytes, caption_html)] -> album (or single photo)."""
    api = f"https://api.telegram.org/bot{token}/"
    for as_html in (True, False):
        caps = [c if as_html else plain(c) for _, c in photos]
        if len(photos) == 1:
            data = dict(chat_id=chat, caption=caps[0])
            if as_html:
                data["parse_mode"] = "HTML"
            r = requests.post(api + "sendPhoto", data=data, files={"photo": ("chart.png", photos[0][0])}, timeout=60)
        else:
            media = []
            for i, cap in enumerate(caps):
                m = dict(type="photo", media=f"attach://p{i}", caption=cap)
                if as_html:
                    m["parse_mode"] = "HTML"
                media.append(m)
            r = requests.post(api + "sendMediaGroup", data=dict(chat_id=chat, media=json.dumps(media)),
                              files={f"p{i}": (f"p{i}.png", png) for i, (png, _) in enumerate(photos)}, timeout=60)
        if r.ok:
            return
        if r.status_code != 400:
            break
    raise RuntimeError(f"Telegram photo {r.status_code}: {r.text[:300]}")


# ----------------------------------------------------------------- main
def gather(state):
    """Run every fetch in parallel; anything slower than DEADLINE is dropped."""
    today = dt.datetime.now(IST).strftime("%Y-%m-%d")
    cache = state.setdefault("_lv", {})
    ex = ThreadPoolExecutor(8)
    jobs = {}
    for n, sym in YF_SYM.items():
        jobs[(n, "m15")] = ex.submit(yf, sym, "15m", "5d")
        if cache.get(n, {}).get("date") != today:          # daily levels: once per day
            jobs[(n, "d")] = ex.submit(yf, sym, "1d", "1y")
        if n in HAS_CHAIN:
            jobs[(n, "oc")] = ex.submit(nse_chain, n)
    done, _ = wait(list(jobs.values()), timeout=DEADLINE)
    ex.shutdown(wait=False, cancel_futures=True)

    def val(key):
        f = jobs.get(key)
        if f is None:
            return None, None
        if f not in done:
            return None, "timeout"
        e = f.exception()
        return (None, repr(e)) if e else (f.result(), None)

    out = []
    for n in YF_SYM:
        r = dict(name=n, err=[])
        b, e = val((n, "m15"))
        r["b"] = last_days(b) if b and len(b["c"]) > 5 else None
        if e:
            r["err"].append(f"15m {e}")
        d, e = val((n, "d"))
        if d:
            cache[n] = dict(daily_levels(d), date=today)
        elif e:
            r["err"].append(f"daily {e}")
        r["lv"] = cache.get(n)
        r["oc"], e = val((n, "oc"))
        if e:
            r["err"].append(f"NSE {e}")
        live = r["oc"]["spot"] if r["oc"] else (float(r["b"]["c"][-1]) if r["b"] else None)
        r["spot"] = live
        r["chg"] = pct(live, r["lv"]["prev_close"]) if live and r["lv"] else 0.0
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.force and not market_open():
        print("Market closed - nothing sent.")
        return 0

    token, chat = os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")
    hook = os.getenv("N8N_WEBHOOK_URL", "").strip()
    if not a.dry_run and not hook and not (token and chat):
        print("Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (or N8N_WEBHOOK_URL)")
        return 1

    t0 = time.time()
    state = load_state()
    results = gather(state)
    print(f"fetched in {time.time() - t0:.1f}s | " + " | ".join(
        f"{r['name']}: chart={'y' if r['b'] else 'n'} chain={'y' if r['oc'] else 'n'} {r['err']}" for r in results))
    if not any(r["spot"] is not None for r in results):
        print("No data from any source.")
        return 1

    photos, text_blocks = [], [header(results)]
    for r in results:
        if r["spot"] is None:
            text_blocks.append(f"⚠ <b>{r['name']}</b>: no data ({html.escape('; '.join(r['err']))})")
            continue
        cap = caption(r, r["spot"], r["chg"])
        png = None
        if r["b"] is not None:
            try:
                png = make_chart(r["name"], r["b"], r["lv"], r["oc"], r["spot"], r["chg"])
            except Exception as e:
                print(f"chart {r['name']} failed: {e!r}")
        if png:
            photos.append((png, cap))
            if a.dry_run:
                Path("charts").mkdir(exist_ok=True)
                Path(f"charts/{r['name']}.png").write_bytes(png)
        else:
            text_blocks.append(cap)
        if r["oc"]:
            prev, mins = usable_prev(state, r["name"])
            text_blocks.append(oi_block(r, prev, mins))
            state[r["name"]] = snapshot(r["oc"])
        elif r["name"] in HAS_CHAIN:
            text_blocks.append(f"⚠ <b>{r['name']}</b> option chain unavailable this run (NSE slow/blocked).")

    msgs = chunk(text_blocks)
    if a.dry_run:
        print("\n\n---- next message ----\n\n".join(msgs))
        for _, c in photos:
            print("\n[caption]\n" + c)
        return 0

    if photos and token and chat:
        send_photos(photos, token, chat)
    for i, m in enumerate(msgs):
        send_text(m, buttons() if i == len(msgs) - 1 else None, token, chat, hook)
        time.sleep(0.5)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state))
    print(f"Sent {len(photos)} chart(s) + {len(msgs)} text message(s) in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)      # don't wait for any straggling fetch threads
