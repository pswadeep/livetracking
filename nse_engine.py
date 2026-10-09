#!/usr/bin/env python3
"""
NSE Support/Resistance Confluence Engine  (personal use)

Data   : NSE option chain (cookie-warmed session, v3 endpoint -> legacy fallback)
         Yahoo Finance candles for ^NSEI / ^NSEBANK (daily, 15m, 5m)
Methods: swings + trendlines, anchored VWAP, OI / change-in-OI / PCR / max pain,
         standard pivots, Camarilla, Fibonacci, EMA 20/50/200, volume profile
         (TPO fallback when index volume is 0), Gann Square of 9
Output : confluence zones scored out of 14 + live signals, served as JSON to index.html

Run    : pip install requests numpy
         python nse_engine.py            -> http://127.0.0.1:8000
         python nse_engine.py --once     -> one text report in the terminal
"""
import datetime as dt, json, math, sys, threading, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
BASE = "https://www.nseindia.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
SYMS = {  # tol = zone width as fraction of spot, bin = volume-profile bucket (points)
    "NIFTY":     dict(yf="^NSEI",   tol=0.0012, bin=10),
    "BANKNIFTY": dict(yf="^NSEBANK", tol=0.0012, bin=25),
}
# confluence weights (distinct method per zone, each counted once) -> max 14
W = dict(swing=1, trend3=2, trend2=1, avwap=2, oi=2, oichg=1, pcr=1, pivot=1, fib=1, ema=1, vp=2)
MAX_SCORE = 14
GRADES = [(14, "Extreme"), (11, "Very strong"), (8, "Strong"), (5, "Moderate"), (0, "Weak")]
OPEN_REFRESH, CLOSED_REFRESH = 180, 900


def market_open(now=None):
    n = now or dt.datetime.now(IST)
    return n.weekday() < 5 and dt.time(9, 15) <= n.time() <= dt.time(15, 30)


def grade(s):
    return next(g for t, g in GRADES if s >= t)


def clean(o):  # JSON-safe: numpy -> python, NaN/inf -> None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# --------------------------------------------------------------------- NSE scraper
class NSE:
    H = {"User-Agent": UA, "Accept": "application/json,text/plain,*/*",
         "Accept-Language": "en-US,en;q=0.9", "Accept-Encoding": "gzip, deflate",
         "Referer": BASE + "/option-chain"}

    def __init__(self):
        self.s, self.t = None, 0.0

    def _sess(self, force=False):
        if force or self.s is None or time.time() - self.t > 240:
            self.s = requests.Session()
            self.s.headers.update(self.H)
            self.s.get(BASE + "/option-chain", timeout=10)  # cookie warm-up
            self.t = time.time()
        return self.s

    def get(self, path, **params):
        err = None
        for a in range(3):
            try:
                r = self._sess(force=a > 0).get(BASE + path, params=params, timeout=12)
                if r.status_code == 200 and r.text.lstrip()[:1] in "{[":
                    return r.json()
                err = f"HTTP {r.status_code}"
            except Exception as e:  # network / cookie problems -> rebuild session
                err = str(e)
            time.sleep(1.5 * (a + 1))
        raise RuntimeError(f"NSE {path}: {err}")

    def chain(self, sym):
        try:  # newer endpoint needs an explicit expiry
            ex = self.get("/api/option-chain-contract-info", symbol=sym)["expiryDates"][0]
            j = self.get("/api/option-chain-v3", type="Indices", symbol=sym, expiry=ex)
            if j.get("records", {}).get("data"):
                return j
        except Exception:
            pass
        return self.get("/api/option-chain-indices", symbol=sym)


# --------------------------------------------------------------------- price data
def yf(sym, interval, rng):
    r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/" + quote(sym),
                     params=dict(interval=interval, range=rng), headers={"User-Agent": UA}, timeout=12)
    j = r.json()["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    a = {k: np.array(q[k], dtype=float) for k in ("open", "high", "low", "close", "volume")}
    t = np.array(j["timestamp"], dtype=float)
    ok = np.isfinite(a["close"]) & np.isfinite(a["high"]) & np.isfinite(a["low"])
    return dict(t=t[ok], o=a["open"][ok], h=a["high"][ok], l=a["low"][ok], c=a["close"][ok],
                v=np.nan_to_num(a["volume"][ok]))


def load_prices(cfg, old=None):
    """Daily refreshed hourly, intraday every 2 min."""
    old, now = old or {}, time.time()
    px = dict(old)
    if now - old.get("ts_d", 0) > 3600:
        px["d"], px["ts_d"] = yf(cfg["yf"], "1d", "2y"), now
    if now - old.get("ts_i", 0) > 120:
        px["m5"], px["m15"], px["ts_i"] = yf(cfg["yf"], "5m", "5d"), yf(cfg["yf"], "15m", "1mo"), now
    return px


# --------------------------------------------------------------------- option chain
def analyse_chain(j, hist_prev=None):
    rec = j["records"]
    spot = float(rec["underlyingValue"])
    rows = rec["data"]

    def rexp(r):  # v3 uses "expiryDates" on the row, legacy "expiryDate"; CE/PE legs carry it too
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
    co, cc, cl = g("CE", "openInterest"), g("CE", "changeinOpenInterest"), g("CE", "lastPrice")
    po, pc, pl = g("PE", "openInterest"), g("PE", "changeinOpenInterest"), g("PE", "lastPrice")
    ia = int(abs(K - spot).argmin())
    atm = int(K[ia])
    near = slice(max(0, ia - 5), ia + 6)
    pain = ((co[None, :] * np.maximum(K[:, None] - K[None, :], 0)).sum(1)
            + (po[None, :] * np.maximum(K[None, :] - K[:, None], 0)).sum(1))

    def walls(mask, o, ch, opp):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-o[idx])][:3]
        return [dict(strike=int(K[i]), oi=float(o[i]), chg=float(ch[i]), ratio=float(o[i] / max(opp[i], 1)))
                for i in idx]

    sup, res = walls(K <= spot, po, pc, co), walls(K >= spot, co, cc, po)
    lo, hi = max(0, ia - 8), ia + 9
    bars = [dict(k=int(K[i]), ce=float(co[i]), pe=float(po[i])) for i in range(lo, min(hi, len(K)))]
    pcr = po.sum() / max(co.sum(), 1)
    pcr_atm = po[near].sum() / max(co[near].sum(), 1)
    ce_d, pe_d = float(cc.sum()), float(pc.sum())
    sc = 0
    sc += 1 if pcr_atm > 1.1 else -1 if pcr_atm < 0.9 else 0
    sc += 1 if pe_d > ce_d * 1.2 else -1 if ce_d > pe_d * 1.2 else 0
    if sup and res:
        sc += (sup[0]["chg"] > 0 and res[0]["chg"] < 0) - (res[0]["chg"] > 0 and sup[0]["chg"] < 0)
    sc += 1 if spot > K[pain.argmin()] else -1
    bias = "BULLISH" if sc >= 2 else "BEARISH" if sc <= -2 else "NEUTRAL"
    return dict(spot=spot, atm=atm, expiry=ex, time=str(rec.get("timestamp", "")), pcr=round(pcr, 2),
                pcr_atm=round(pcr_atm, 2), max_pain=int(K[pain.argmin()]),
                straddle=round(float(cl[ia] + pl[ia]), 1), bias=bias, score=int(sc),
                ce_d=ce_d, pe_d=pe_d, ce_tot=float(co.sum()), pe_tot=float(po.sum()),
                sup=sup, res=res, bars=bars, iv_atm=None)


# --------------------------------------------------------------------- indicators
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


def best_trendline(xs, ps, side, nxt, tol):
    """Line through 2+ swing points (ascending lows / descending highs), most touches then most recent."""
    pts, best = list(zip(xs[-8:], ps[-8:])), None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (x1, y1), (x2, y2) = pts[i], pts[j]
            if x2 == x1 or (side == "sup" and y2 <= y1) or (side == "res" and y2 >= y1):
                continue
            m = (y2 - y1) / (x2 - x1)
            t = sum(abs(y - (y1 + m * (x - x1))) <= tol * y for x, y in pts)
            if t >= 2 and (best is None or (t, x2) > (best[0], best[1])):
                best = (t, x2, y1 + m * (nxt - x1))
    return best


def anchored_vwap(b, i):
    v = b["v"][i:] if b["v"].sum() > 0 else np.ones(len(b["c"]) - i)
    tp = ((b["h"] + b["l"] + b["c"]) / 3)[i:]
    return float((tp * v).sum() / v.sum())


def volume_profile(b, step):
    lo, hi = b["l"].min(), b["h"].max()
    edges = np.arange(math.floor(lo / step) * step, hi + step, step)
    prof = np.zeros(len(edges))
    tpo = b["v"].sum() == 0
    for h, l, v in zip(b["h"], b["l"], b["v"]):
        m = (edges >= l - step / 2) & (edges <= h + step / 2)
        if m.any():
            prof[m] += (1 if tpo else v) / m.sum()
    sm = np.convolve(prof, [1, 1, 1], "same") / 3
    mean = sm[sm > 0].mean()
    hvn = [i for i in range(1, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > 1.2 * mean]
    lvn = [i for i in range(1, len(sm) - 1) if sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1] and 0 < sm[i] < 0.6 * mean]
    hvn = sorted(hvn, key=lambda i: -sm[i])[:4]
    return dict(poc=float(edges[sm.argmax()]), hvn=[float(edges[i]) for i in hvn],
                lvn=[float(edges[i]) for i in lvn][:4], mode="TPO (index has no volume)" if tpo else "Volume")


def structure(h, l):
    hi, lo = swings(h, l, 3)
    if len(hi) < 2 or len(lo) < 2:
        return "UNDEFINED"
    hh, hl = h[hi[-1]] > h[hi[-2]], l[lo[-1]] > l[lo[-2]]
    return ("UPTREND (HH-HL)" if hh and hl else "DOWNTREND (LH-LL)" if not hh and not hl
            else "EXPANSION (HH-LL)" if hh else "CONTRACTION (LH-HL)")


# --------------------------------------------------------------------- confluence
def build(oc, px, cfg, is_open):
    spot, tol = oc["spot"], cfg["tol"]
    C, LV = [], {}
    near = lambda p: p and abs(p - spot) <= 0.04 * spot

    def add(p, kind, label, w):
        if near(p):
            C.append((float(p), kind, label, w))

    d, m5, m15 = px["d"], px["m5"], px["m15"]
    today = dt.datetime.now(IST).date()
    i = -2 if is_open and dt.datetime.fromtimestamp(d["t"][-1], IST).date() == today else -1
    H, Lw, Cl = d["h"][i], d["l"][i], d["c"][i]
    n = len(d["c"]) + i + 1
    P = (H + Lw + Cl) / 3
    rg = H - Lw
    piv = dict(R3=H + 2 * (P - Lw), R2=P + rg, R1=2 * P - Lw, P=P, S1=2 * P - H, S2=P - rg, S3=Lw - 2 * (H - P))
    cam = dict(R4=Cl + rg * 1.1 / 2, R3=Cl + rg * 1.1 / 4, C=Cl, S3=Cl - rg * 1.1 / 4, S4=Cl - rg * 1.1 / 2)
    for k, v in piv.items():
        add(v, "pivot", f"Pivot {k}", W["pivot"])
    for k, v in cam.items():
        if k != "C":
            add(v, "pivot", f"Camarilla {k}", W["pivot"])
    prev = dict(PDH=H, PDL=Lw, PWH=d["h"][max(0, n - 5):n].max(), PWL=d["l"][max(0, n - 5):n].min())
    for k, v in prev.items():
        add(v, "swing", k, W["swing"])
    LV.update(pivot=piv, cam=cam, prev=prev)

    # swings (daily + 15m)
    dh, dl = swings(d["h"][-150:], d["l"][-150:], 4)
    for ix in dh[-5:]:
        add(d["h"][-150:][ix], "swing", "Daily swing high", W["swing"])
    for ix in dl[-5:]:
        add(d["l"][-150:][ix], "swing", "Daily swing low", W["swing"])

    # Fibonacci off latest daily swing
    ih, il = swings(d["h"][-120:], d["l"][-120:], 5)
    fib = {}
    if len(ih) and len(il):
        fh, fl = d["h"][-120:][ih[-1]], d["l"][-120:][il[-1]]
        for r in (0.382, 0.5, 0.618):
            fib[f"{r*100:g}%"] = fh - (fh - fl) * r if il[-1] < ih[-1] else fl + (fh - fl) * r
            add(fib[f"{r*100:g}%"], "fib", f"Fib {r*100:g}%", W["fib"])
    LV["fib"] = fib

    # EMAs (daily)
    em = {str(n_): float(ema(d["c"], n_)[-1]) for n_ in (20, 50, 200)}
    for k, v in em.items():
        add(v, "ema", f"EMA {k}", W["ema"])
    LV["ema"] = em

    # anchored VWAP (swing high / swing low / start of 5-day window)
    av = dict(swing_high=anchored_vwap(m5, int(m5["h"].argmax())), swing_low=anchored_vwap(m5, int(m5["l"].argmin())),
              week_start=anchored_vwap(m5, 0))
    for k, v in av.items():
        add(v, "avwap", f"AVWAP {k}", W["avwap"])
    LV["avwap"] = av

    # volume profile
    vp = volume_profile(m5, cfg["bin"])
    add(vp["poc"], "vp", "POC", W["vp"])
    for v in vp["hvn"]:
        add(v, "vp", "HVN", W["vp"])
    LV["vp"] = vp

    # trendlines (daily 90 bars, 15m)
    trend = []
    for tf, b, k, lim in (("1D", d, 3, 90), ("15m", m15, 4, 300)):
        h, l = b["h"][-lim:], b["l"][-lim:]
        hi, lo = swings(h, l, k)
        nxt = len(h)
        for side, ix, arr in (("sup", lo, l), ("res", hi, h)):
            if len(ix) >= 2:
                t = best_trendline(ix, arr[ix], side, nxt, 0.0015)
                if t and near(t[2]):
                    trend.append(dict(name=f"{tf} {'rising' if side == 'sup' else 'falling'}", price=float(t[2]),
                                      touches=int(t[0])))
                    add(t[2], "trend", f"{tf} trendline ({t[0]} touches)", W["trend3"] if t[0] >= 3 else W["trend2"])
    LV["trend"] = trend

    # option-chain levels
    for side in ("sup", "res"):
        for w in oc[side]:
            add(w["strike"], "oi", f"{'Put' if side == 'sup' else 'Call'} OI wall {w['strike']}", W["oi"])
            if w["chg"] > 0:
                add(w["strike"], "oichg", f"OI rising {w['strike']}", W["oichg"])
    add(oc["max_pain"], "gann", "Max pain", 0)

    # Gann square of 9 (45 degree steps) - informational, no score
    sq = math.sqrt(spot)
    gann = [round((sq + k * 0.25) ** 2) for k in range(-6, 7) if k]
    for g in gann:
        add(g, "gann", "Gann 45°", 0)
    LV["gann"] = gann

    # cluster -> zones
    C.sort()
    clusters, cur = [], [C[0]]
    for c in C[1:]:
        if c[0] - cur[0][0] <= 2 * tol * spot:
            cur.append(c)
        else:
            clusters.append(cur)
            cur = [c]
    clusters.append(cur)
    sup_z, res_z = [], []
    for cl in clusters:
        kinds = {}
        for p, k, lbl, w in cl:
            k = "trend" if k == "trend" else k
            if k not in kinds or w > kinds[k][0]:
                kinds[k] = (w, lbl)
        sc = sum(w for k, (w, _) in kinds.items() if k != "gann")
        mid = float(np.mean([p for p, k, _, _ in cl if k != "gann"] or [cl[0][0]]))
        side = "sup" if mid < spot else "res"
        pcr_ok = oc["pcr"] >= 1.0 if side == "sup" else oc["pcr"] <= 0.8
        if sc >= 2 and pcr_ok:
            sc += W["pcr"]
            kinds["pcr"] = (1, f"PCR {oc['pcr']}")
        sc = min(sc, MAX_SCORE)
        if sc < 3:
            continue
        z = dict(lo=min(p for p, *_ in cl), hi=max(p for p, *_ in cl), mid=mid, score=sc, grade=grade(sc),
                 dist=(mid - spot) / spot * 100, tags=[l for _, (w, l) in kinds.items()])
        (sup_z if side == "sup" else res_z).append(z)
    sup_z = sorted(sup_z, key=lambda z: (-z["score"], -z["mid"]))[:5]
    res_z = sorted(res_z, key=lambda z: (-z["score"], z["mid"]))[:5]

    # signals
    sig = []
    for z, side in [(z, "sup") for z in sup_z[:2]] + [(z, "res") for z in res_z[:2]]:
        if z["score"] >= 8 and abs(z["dist"]) <= 0.2:
            sig.append(f"{'BUY' if side == 'sup' else 'SELL'} watch: at {side} zone {z['lo']:.0f}-{z['hi']:.0f} "
                       f"({z['score']}/{MAX_SCORE} {z['grade']}) - wait for rejection candle, not a blind entry")
    if spot > cam["R4"]:
        sig.append("BUY breakout: spot above Camarilla R4")
    elif spot < cam["S4"]:
        sig.append("SELL breakdown: spot below Camarilla S4")
    elif cam["S3"] < spot < cam["R3"]:
        sig.append("Range day bias: spot between Camarilla S3 and R3")
    if oc["pcr"] > 1.5:
        sig.append("Caution: PCR > 1.5, crowded bullish positioning")
    if oc["pcr"] < 0.7:
        sig.append("Caution: PCR < 0.7, call-side dominance")
    if abs(spot - oc["max_pain"]) / spot < 0.002:
        sig.append(f"Pinning risk: spot near max pain {oc['max_pain']}")
    if not sig:
        sig.append("No high-confluence setup at spot - wait for price to reach a zone")

    mid = [z["mid"] for z in sup_z + res_z]
    return dict(zones_sup=sup_z, zones_res=res_z, levels=LV, structure=structure(d["h"][-120:], d["l"][-120:]),
                signals=sig, price_time=dt.datetime.fromtimestamp(m5["t"][-1], IST).strftime("%d %b %H:%M"))


# --------------------------------------------------------------------- engine + server
class Engine:
    def __init__(self):
        self.data = {n: {} for n in SYMS}
        self.hist = {n: deque(maxlen=14) for n in SYMS}
        self.prev = {}
        self.px = {n: {} for n in SYMS}
        self.nse = {n: NSE() for n in SYMS}
        self.next = time.time()
        self.wake = threading.Event()

    def one(self, n):
        cfg = SYMS[n]
        try:
            oc = analyse_chain(self.nse[n].chain(n))
        except Exception as e:
            self.data[n] = {**self.data[n], "error": f"NSE: {e!r}"}
            return
        warn = ""
        try:
            self.px[n] = load_prices(cfg, self.px[n])
        except Exception as e:
            warn = f"price feed: {e!r}"
        try:
            adv = build(oc, self.px[n], cfg, market_open()) if "d" in self.px[n] else {}
        except Exception as e:
            adv, warn = {}, warn or f"analysis: {e!r}"
        p = self.prev.get(n)
        self.prev[n] = (oc["ce_tot"], oc["pe_tot"])
        if p:
            self.hist[n].appendleft(dict(t=dt.datetime.now(IST).strftime("%H:%M"), spot=oc["spot"], pcr=oc["pcr"],
                                         ce=oc["ce_tot"] - p[0], pe=oc["pe_tot"] - p[1], bias=oc["bias"]))
        sig = adv.pop("signals", []) if adv else []
        self.data[n] = clean({**oc, **adv, "signals": sig, "history": list(self.hist[n]), "error": warn})

    def cycle(self):
        with ThreadPoolExecutor(len(SYMS)) as ex:
            list(ex.map(self.one, SYMS))

    def loop(self):
        while True:
            self.cycle()
            iv = OPEN_REFRESH if market_open() else CLOSED_REFRESH
            self.next = time.time() + iv
            self.wake.wait(iv)
            self.wake.clear()

    def report(self):
        out = []
        for n, r in self.data.items():
            if not r.get("spot"):
                out.append(f"{n}: no data ({r.get('error')})")
                continue
            out += [f"== {n} {r['spot']} | exp {r['expiry']} | PCR {r['pcr']} | max pain {r['max_pain']} | "
                    f"bias {r['bias']} | structure {r.get('structure', '-')}"]
            for z in reversed(r.get("zones_res", [])):
                out.append(f"  R {z['lo']:.0f}-{z['hi']:.0f}  {z['score']}/{MAX_SCORE} {z['grade']}: {', '.join(z['tags'])}")
            out.append(f"  ---- spot {r['spot']}")
            for z in r.get("zones_sup", []):
                out.append(f"  S {z['lo']:.0f}-{z['hi']:.0f}  {z['score']}/{MAX_SCORE} {z['grade']}: {', '.join(z['tags'])}")
            out += ["  " + s for s in r["signals"]]
        return "\n".join(out)


ENG = Engine()
HTML = Path(__file__).with_name("index.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/data":
            self.send(json.dumps(ENG.data))
        elif p == "/api/status":
            self.send(json.dumps(dict(next=ENG.next, market_open=market_open())))
        elif p == "/api/report":
            self.send(ENG.report(), "text/plain")
        elif p in ("/", "/index.html") and HTML.exists():
            self.send(HTML.read_bytes(), "text/html")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/fetch":
            ENG.wake.set()
            self.send("{}")
        else:
            self.send_error(404)


if __name__ == "__main__":
    if "--once" in sys.argv:
        ENG.cycle()
        print(ENG.report())
    else:
        port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
        threading.Thread(target=ENG.loop, daemon=True).start()
        print(f"Dashboard: http://127.0.0.1:{port}  (Ctrl+C to stop)")
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()