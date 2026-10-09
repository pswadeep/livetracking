#!/usr/bin/env python3
"""
NSE -> Telegram reporter (runs headless, built for GitHub Actions every 5 min).

Needs nse_engine.py (your existing file) in the same folder.

Message contains ONLY:
  * NIFTY / BANKNIFTY / SENSEX current value
  * OI + option-chain change (vs previous run and vs previous close) with a plain-English reading
  * PDH PDL PWH PWL, Fibonacci 38.2/50/61.8, EMA 20/50/200 (daily)
  * OI by strike, Resistances (call OI), Supports (put OI)

Delivery:
  N8N_WEBHOOK_URL set  -> POST {"text","parse_mode"} to the n8n webhook (n8n forwards to Telegram)
  otherwise            -> direct Telegram Bot API (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)

Usage:
  python nse_telegram.py              # only runs 09:15-15:30 IST Mon-Fri
  python nse_telegram.py --force      # run outside market hours
  python nse_telegram.py --dry-run    # print instead of sending
"""
import argparse
import datetime as dt
import html
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from nse_engine import IST, NSE, analyse_chain, ema, market_open, swings, yf

STATE = Path(os.getenv("STATE_FILE", "state/snapshot.json"))
YF_SYM = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
HAS_CHAIN = ("NIFTY", "BANKNIFTY")  # Sensex options live on BSE, not NSE
STALE_AFTER = 20 * 60               # ignore previous snapshot older than 20 min
MAX_MSG = 3900


# ----------------------------------------------------------------- formatting
def n0(x):
    return f"{x:,.0f}"


def lk(x):  # open interest in lakh contracts
    return f"{x / 1e5:.2f}L"


def kd(x):  # OI delta in thousands
    return "  —" if x is None else f"{x / 1e3:+.1f}k"


def pct(a, b):
    return (a - b) / b * 100 if b else 0.0


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
            fib[f"{r * 100:g}%"] = float(fh - (fh - fl) * r if il[-1] < ih[-1] else fl + (fh - fl) * r)
    lv["fib"] = fib
    lv["ema"] = {k: float(ema(d["c"], k)[-1]) for k in (20, 50, 200)}
    return lv


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


# ----------------------------------------------------------------- text blocks
def oi_table(oc, prev, mins):
    bars = oc["bars"]
    ia = next((i for i, b in enumerate(bars) if b["k"] == oc["atm"]), len(bars) // 2)
    rows = bars[max(0, ia - 5): ia + 6]
    ps = (prev or {}).get("strikes", {})
    head = f"{'Strike':<7}{'CE OI':>8}{'chg':>8}{'PE OI':>8}{'chg':>8}"
    out = [head]
    for b in rows:
        p = ps.get(str(b["k"]))
        dce = b["ce"] - p[0] if p else None
        dpe = b["pe"] - p[1] if p else None
        mark = "*" if b["k"] == oc["atm"] else " "
        out.append(f"{str(b['k']) + mark:<7}{lk(b['ce']):>8}{kd(dce):>8}{lk(b['pe']):>8}{kd(dpe):>8}")
    note = f"chg = vs {mins}m ago, * = ATM" if prev else "chg = n/a (first run today), * = ATM"
    return "<pre>" + html.escape("\n".join(out)) + "</pre>" + note


def wall_lines(walls, prev, tag, mins):
    out = []
    for n, w in enumerate(walls, 1):
        before = (prev or {}).get("walls", {}).get(f"{tag}{w['strike']}")
        step = f" | {kd(w['oi'] - before)} in {mins}m" if before is not None else ""
        out.append(f"{n}. {w['strike']}  OI {lk(w['oi'])} | today {kd(w['chg'])}{step}")
    return "\n".join(out) or "n/a"


def explain(oc, lv, prev, mins):
    s, sup, res = oc["spot"], oc["sup"], oc["res"]
    L = []
    if sup and res:
        L.append(f"Range: put wall {sup[0]['strike']} (support) to call wall {res[0]['strike']} (resistance). "
                 f"Expect chop inside the band; max pain {oc['max_pain']} tends to pull price into expiry.")
    # flow since previous close (NSE change-in-OI)
    ce_d, pe_d = oc["ce_d"], oc["pe_d"]
    if pe_d > ce_d * 1.2:
        L.append(f"Today: put OI added {kd(pe_d)} vs call {kd(ce_d)} - more put writing, supportive/bullish tilt.")
    elif ce_d > pe_d * 1.2:
        L.append(f"Today: call OI added {kd(ce_d)} vs put {kd(pe_d)} - more call writing, upside capped/bearish tilt.")
    else:
        L.append(f"Today: call {kd(ce_d)} vs put {kd(pe_d)} - balanced writing, no clear side.")
    # flow since last run (before -> now)
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
        L.append(f"Last {mins}m (spot {ds:+.0f}, call OI {kd(dce)}, put OI {kd(dpe)}): "
                 + ("; ".join(hits[:2]) if hits else "no clear OI shift."))
    # wall health
    if res:
        r = res[0]
        t = f"Resistance {r['strike']}: call OI {'rising - wall firm' if r['chg'] > 0 else 'falling - wall weakening'}"
        if len(res) > 1:
            t += f"; sustained break opens {res[1]['strike']}"
        L.append(t + ".")
    if sup:
        x = sup[0]
        t = f"Support {x['strike']}: put OI {'rising - floor firm' if x['chg'] > 0 else 'falling - floor weakening'}"
        if len(sup) > 1:
            t += f"; breakdown opens {sup[1]['strike']}"
        L.append(t + ".")
    pcr = oc["pcr"]
    L.append(f"PCR {pcr}: " + ("above 1.5, crowded longs - reversal risk." if pcr > 1.5 else
                              "above 1.2, put-heavy (supportive)." if pcr > 1.2 else
                              "below 0.7, call-heavy - upside capped." if pcr < 0.7 else
                              "below 0.9, call-leaning." if pcr < 0.9 else "neutral zone."))
    e20 = lv["ema"][20]
    ctx = f"Spot {'above' if s > e20 else 'below'} daily EMA20 ({n0(e20)})"
    if s > lv["PDH"]:
        ctx += ", above PDH - intraday strength"
    elif s < lv["PDL"]:
        ctx += ", below PDL - intraday weakness"
    L.append(ctx + ".")
    L.append(f"Net bias from OI: <b>{oc['bias']}</b> (score {oc['score']:+d}).")
    return "\n".join("• " + html.escape(x).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>") for x in L)


def levels_block(lv):
    f = lv["fib"]
    fib = " | ".join(f"{k} {n0(v)}" for k, v in f.items()) or "n/a"
    e = lv["ema"]
    return (f"<b>Previous day / week</b>\n"
            f"PDH {n0(lv['PDH'])} | PDL {n0(lv['PDL'])} | PWH {n0(lv['PWH'])} | PWL {n0(lv['PWL'])}\n"
            f"<b>Fibonacci (latest daily swing)</b>\n{fib}\n"
            f"<b>EMA 20 / 50 / 200 (daily)</b>\n{n0(e[20])} / {n0(e[50])} / {n0(e[200])}")


def index_block(r, prev, mins):
    name, oc, lv = r["name"], r["oc"], r["lv"]
    spot = oc["spot"] if oc else (lv["live"] if lv else None)
    if spot is None:
        return f"<b>{name}</b>: no data ({html.escape('; '.join(r['err']))})"
    chg = pct(spot, lv["prev_close"]) if lv else 0.0
    out = [f"📊 <b>{name}</b>  {n0(spot)}  ({chg:+.2f}%)"]
    if oc:
        out.append(f"Expiry {oc['expiry']} | PCR {oc['pcr']} | Max pain {oc['max_pain']}")
    if lv:
        out.append("\n" + levels_block(lv))
    if oc:
        out.append("\n<b>OI by strike (CE vs PE)</b>\n" + oi_table(oc, prev, mins))
        out.append("\n<b>Resistances (call OI)</b>\n" + wall_lines(oc["res"], prev, "res", mins))
        out.append("<b>Supports (put OI)</b>\n" + wall_lines(oc["sup"], prev, "sup", mins))
        if lv:
            out.append("\n<b>What it means</b>\n" + explain(oc, lv, prev, mins))
    elif name in HAS_CHAIN:
        out.append("\n⚠ Option chain unavailable this run (NSE blocked/failed).")
    else:
        out.append("\nOption chain: n/a (Sensex options are on BSE, not NSE).")
    if r["err"] and oc:
        out.append("⚠ " + html.escape("; ".join(r["err"])))
    return "\n".join(out)


def summary_block(results):
    now = dt.datetime.now(IST).strftime("%d %b %H:%M")
    out = [f"🕒 <b>Market snapshot</b> {now} IST"]
    for r in results:
        spot = r["oc"]["spot"] if r["oc"] else (r["lv"]["live"] if r["lv"] else None)
        if spot is None:
            out.append(f"{r['name']}: n/a")
        else:
            c = pct(spot, r["lv"]["prev_close"]) if r["lv"] else 0.0
            out.append(f"{r['name']}: <b>{n0(spot)}</b> ({c:+.2f}%)")
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


# ----------------------------------------------------------------- fetch + send
def fetch(name):
    r = dict(name=name, oc=None, lv=None, err=[])
    try:
        r["lv"] = daily_levels(yf(YF_SYM[name], "1d", "2y"))
    except Exception as e:
        r["err"].append(f"price: {e!r}")
    if name in HAS_CHAIN:
        try:
            r["oc"] = analyse_chain(NSE().chain(name))
        except Exception as e:
            r["err"].append(f"NSE: {e!r}")
    return r


def send(msgs):
    hook = os.getenv("N8N_WEBHOOK_URL", "").strip()
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")
    if not hook and not (token and chat):
        sys.exit("Set N8N_WEBHOOK_URL or TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")
    for m in msgs:
        if hook:
            r = requests.post(hook, json={"text": m, "parse_mode": "HTML"}, timeout=20)
        else:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=20,
                              json=dict(chat_id=chat, text=m, parse_mode="HTML", disable_web_page_preview=True))
        r.raise_for_status()
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.force and not market_open():
        print("Market closed - nothing sent.")
        return 0

    state = load_state()
    with ThreadPoolExecutor(3) as ex:
        results = list(ex.map(fetch, YF_SYM))
    if not any(r["oc"] or r["lv"] for r in results):
        print("No data from any source:", [r["err"] for r in results])
        return 1

    blocks = [summary_block(results)]
    for r in results:
        prev, mins = usable_prev(state, r["name"])
        blocks.append(index_block(r, prev, mins))
        if r["oc"]:
            state[r["name"]] = snapshot(r["oc"])
    msgs = chunk(blocks)

    if a.dry_run:
        print("\n\n----\n\n".join(msgs))
    else:
        send(msgs)
        print(f"Sent {len(msgs)} message(s).")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
