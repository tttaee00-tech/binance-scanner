"""
Gate.io Futures Scalping Scanner
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="⚡ Scalping Scanner", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp { background: #080c14; color: #e2e8f0; }
.signal-card { background: #0f1923; border-radius: 12px; padding: 14px 16px; margin-bottom: 8px; border-left: 3px solid #334155; }
.long-card  { border-left-color: #10b981 !important; background: #0a1f17 !important; }
.short-card { border-left-color: #ef4444 !important; background: #1f0a0a !important; }
.metric-box { background: #0f1923; border-radius: 8px; padding: 10px 14px; text-align: center; }
</style>
""", unsafe_allow_html=True)

GATE = "https://api.gateio.ws/api/v4"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

@st.cache_data(ttl=30, show_spinner=False)
def get_tickers():
    try:
        r = requests.get(f"{GATE}/futures/usdt/tickers", headers=HEADERS, timeout=8)
        if r.status_code == 200:
            return {x["contract"]: x for x in r.json()}
    except: pass
    return {}

@st.cache_data(ttl=60, show_spinner=False)
def get_klines(symbol: str, interval: str, limit: int = 100):
    try:
        r = requests.get(f"{GATE}/futures/usdt/candlesticks",
            params={"contract": symbol, "interval": interval, "limit": limit},
            headers=HEADERS, timeout=8)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data)
            # Gate.io 컬럼: t(시간) o c h l v
            df = df.rename(columns={"o":"o","h":"h","l":"l","c":"c","v":"v","t":"ts"})
            for col in ["o","h","l","c","v"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def get_funding():
    try:
        r = requests.get(f"{GATE}/futures/usdt/contracts", headers=HEADERS, timeout=8)
        if r.status_code == 200:
            return {x["name"]: float(x.get("funding_rate", 0)) for x in r.json()}
    except: pass
    return {}

def calc_indicators(df):
    if df.empty or len(df) < 50: return {}
    c = df["c"]; h = df["h"]; l = df["l"]

    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    bb_pct = (c - bb_dn) / (bb_up - bb_dn + 1e-10)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    up  = h.diff(); dn = -l.diff()
    pdm = up.where((up>dn)&(up>0), 0)
    ndm = dn.where((dn>up)&(dn>0), 0)
    pdi = 100*pdm.rolling(14).mean() / atr.replace(0,1e-10)
    ndi = 100*ndm.rolling(14).mean() / atr.replace(0,1e-10)
    dx  = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0,1e-10)
    adx = dx.rolling(14).mean()

    return {
        "price":     float(c.iloc[-1]),
        "ema_long":  float(ema9.iloc[-1]) > float(ema21.iloc[-1]) > float(ema50.iloc[-1]),
        "ema_short": float(ema9.iloc[-1]) < float(ema21.iloc[-1]) < float(ema50.iloc[-1]),
        "bb_pct":    float(bb_pct.iloc[-1]),
        "rsi":       float(rsi.iloc[-1]),
        "atr":       float(atr.iloc[-1]),
        "atr_pct":   float(atr.iloc[-1] / c.iloc[-1] * 100),
        "adx":       float(adx.iloc[-1]),
    }

def score_signal(ind15, ind5, funding, btc_long, btc_short, lev):
    if not ind15 or not ind5:
        return {"signal":"NONE","score":0,"reasons":[]}

    long_ok = []; short_ok = []

    if ind15.get("ema_long"):  long_ok.append("✅ 15분 정배열")
    if ind15.get("ema_short"): short_ok.append("✅ 15분 역배열")
    if ind5.get("atr_pct",0) > 0.05:
        long_ok.append("✅ 변동성 OK"); short_ok.append("✅ 변동성 OK")
    if funding < 0.001:  long_ok.append(f"✅ 펀딩비 {funding*100:.3f}%")
    if funding > -0.001: short_ok.append(f"✅ 펀딩비 {funding*100:.3f}%")

    bb  = ind5.get("bb_pct", 0.5)
    rsi = ind5.get("rsi", 50)
    if bb < 0.15 and rsi < 40: long_ok.append("✅ 볼린저 하단 반전")
    if bb > 0.85 and rsi > 60: short_ok.append("✅ 볼린저 상단 반전")
    if btc_long:  long_ok.append("✅ BTC 방향 일치")
    if btc_short: short_ok.append("✅ BTC 방향 일치")
    if ind15.get("adx",0) > 20:
        long_ok.append(f"✅ ADX {ind15['adx']:.0f}")
        short_ok.append(f"✅ ADX {ind15['adx']:.0f}")

    atr   = ind5.get("atr", 0)
    price = ind5.get("price", 0)
    sl_p  = atr * 1.2 / price * 100 if price else 0

    if len(long_ok) >= 4 and ind15.get("ema_long") and bb < 0.2:
        return {"signal":"LONG","score":len(long_ok),"reasons":long_ok,
                "entry":price,"sl":round(price-atr*1.2,4),
                "tp1":round(price+atr*1.5,4),"tp2":round(price+atr*2.5,4),
                "rsi":rsi,"adx":ind15.get("adx",0),"funding":funding,
                "atr_pct":ind5.get("atr_pct",0),"sl_pct":sl_p}
    elif len(short_ok) >= 4 and ind15.get("ema_short") and bb > 0.8:
        return {"signal":"SHORT","score":len(short_ok),"reasons":short_ok,
                "entry":price,"sl":round(price+atr*1.2,4),
                "tp1":round(price-atr*1.5,4),"tp2":round(price-atr*2.5,4),
                "rsi":rsi,"adx":ind15.get("adx",0),"funding":funding,
                "atr_pct":ind5.get("atr_pct",0),"sl_pct":sl_p}
    else:
        return {"signal":"WAIT","score":max(len(long_ok),len(short_ok)),"reasons":[],"rsi":rsi}

# Gate.io 선물 심볼은 BTC_USDT 형식
TARGETS = ["BTC_USDT","ETH_USDT","BNB_USDT","SOL_USDT","XRP_USDT",
           "ADA_USDT","DOGE_USDT","AVAX_USDT","DOT_USDT","LINK_USDT",
           "MATIC_USDT","LTC_USDT","UNI_USDT","ATOM_USDT","NEAR_USDT",
           "APT_USDT","ARB_USDT","OP_USDT","INJ_USDT","SUI_USDT"]

# ── 헤더 ──
st.markdown('<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:12px;padding:16px 20px;margin-bottom:16px;"><span style="font-size:22px;font-weight:700;">⚡ Gate.io Scalping Scanner</span><span style="font-size:11px;color:#64748b;margin-left:10px;">EMA+BB+ATR+펀딩비 | A등급 전략</span></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([2,1,1])
with c1: scan_btn = st.button("⚡ 스캔", type="primary", use_container_width=True)
with c2: lev = st.selectbox("레버리지", [10,15,20], index=0)
with c3: auto = st.toggle("30초 자동갱신")

if auto:
    import time; time.sleep(0.5); st.rerun()

if scan_btn or auto or "scan_done" not in st.session_state:
    prog = st.progress(0, "스캔 중...")
    funding_map = get_funding()
    tickers     = get_tickers()

    btc15 = calc_indicators(get_klines("BTC_USDT", "15m"))
    btc_long  = btc15.get("ema_long",  False)
    btc_short = btc15.get("ema_short", False)

    results = []
    for i, sym in enumerate(TARGETS):
        prog.progress((i+1)/len(TARGETS), f"스캔 중... {sym}")
        try:
            df15 = get_klines(sym, "15m")
            df5  = get_klines(sym, "5m", 60)
            ind15 = calc_indicators(df15)
            ind5  = calc_indicators(df5)
            fund  = funding_map.get(sym, 0)
            tk    = tickers.get(sym, {})
            chg24 = float(tk.get("change_percentage", 0)) if tk else 0
            sig   = score_signal(ind15, ind5, fund, btc_long, btc_short, lev)
            sig["symbol"] = sym.replace("_USDT","")
            sig["chg24"]  = chg24
            results.append(sig)
        except: pass

    prog.empty()
    st.session_state.update({"scan_done":True,"results":results,
        "scan_time":datetime.now().strftime("%H:%M:%S"),
        "btc_long":btc_long,"btc_short":btc_short})

results   = st.session_state.get("results", [])
scan_t    = st.session_state.get("scan_time", "-")
btc_long  = st.session_state.get("btc_long", False)
btc_short = st.session_state.get("btc_short", False)

longs  = sorted([r for r in results if r["signal"]=="LONG"],  key=lambda x:x["score"], reverse=True)
shorts = sorted([r for r in results if r["signal"]=="SHORT"], key=lambda x:x["score"], reverse=True)

btc_dir = "🟢 롱" if btc_long else "🔴 숏" if btc_short else "⬜ 중립"
btc_col = "#10b981" if btc_long else "#ef4444" if btc_short else "#94a3b8"

st.markdown(f"""
<div style="display:flex;gap:10px;margin-bottom:14px;">
<div style="flex:1;background:#0f1923;border-radius:8px;padding:10px;text-align:center;">
<div style="font-size:18px;font-weight:700;color:{btc_col};">{btc_dir}</div>
<div style="font-size:11px;color:#64748b;">BTC 15분 추세</div>
</div>
<div style="flex:1;background:#0f1923;border-radius:8px;padding:10px;text-align:center;">
<div style="font-size:18px;font-weight:700;color:#60a5fa;">{scan_t}</div>
<div style="font-size:11px;color:#64748b;">스캔 시각</div>
</div>
<div style="flex:1;background:#0f1923;border-radius:8px;padding:10px;text-align:center;">
<div style="font-size:18px;font-weight:700;color:#10b981;">{len(longs)}</div>
<div style="font-size:11px;color:#64748b;">롱 신호</div>
</div>
<div style="flex:1;background:#0f1923;border-radius:8px;padding:10px;text-align:center;">
<div style="font-size:18px;font-weight:700;color:#ef4444;">{len(shorts)}</div>
<div style="font-size:11px;color:#64748b;">숏 신호</div>
</div>
</div>
""", unsafe_allow_html=True)

def render_card(r, kind):
    sym   = r["symbol"]
    score = r["score"]
    entry = r.get("entry", 0)
    sl    = r.get("sl", 0)
    tp1   = r.get("tp1", 0)
    tp2   = r.get("tp2", 0)
    rsi   = r.get("rsi", 0)
    adx   = r.get("adx", 0)
    fund  = r.get("funding", 0)
    atr_p = r.get("atr_pct", 0)
    chg   = r.get("chg24", 0)
    sl_p  = r.get("sl_pct", 0)
    reasons = r.get("reasons", [])
    css   = "long-card" if kind=="LONG" else "short-card"
    em    = "🚀" if kind=="LONG" else "💀"
    col   = "#10b981" if kind=="LONG" else "#ef4444"
    tp1_p = abs(tp1-entry)/entry*100 if entry else 0
    reasons_html = "".join([f"<div style='font-size:11px;color:#94a3b8;margin-top:2px;'>{rv}</div>" for rv in reasons[:5]])

    st.markdown(f"""
<div class="signal-card {css}">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
  <span style="font-size:16px;font-weight:700;">{em} {sym}</span>
  <span style="background:{'#065f46' if kind=='LONG' else '#7f1d1d'};color:{col};
    padding:2px 10px;border-radius:4px;font-size:12px;font-weight:700;">{'LONG' if kind=='LONG' else 'SHORT'} {score}/6</span>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px;background:#0a0f1a;border-radius:8px;padding:8px;">
  <div style="text-align:center;">
    <div style="font-size:10px;color:#64748b;">진입</div>
    <div style="font-size:13px;font-weight:600;font-family:monospace;">{entry:,.3f}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:10px;color:#ef4444;">손절 -{sl_p:.2f}%</div>
    <div style="font-size:13px;font-family:monospace;color:#ef4444;">{sl:,.3f}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:10px;color:#10b981;">익절 +{tp1_p:.2f}%</div>
    <div style="font-size:13px;font-family:monospace;color:#10b981;">{tp1:,.3f}</div>
  </div>
</div>
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px;">
  <span style="font-size:11px;color:#94a3b8;">RSI {rsi:.0f}</span>
  <span style="font-size:11px;color:#94a3b8;">ADX {adx:.0f}</span>
  <span style="font-size:11px;color:#94a3b8;">ATR {atr_p:.2f}%</span>
  <span style="font-size:11px;color:#94a3b8;">펀딩 {fund*100:.3f}%</span>
  <span style="font-size:11px;color:{'#10b981' if chg>0 else '#ef4444'};">24H {chg:+.1f}%</span>
</div>
{reasons_html}
<div style="font-size:10px;color:#334155;margin-top:6px;">TP2: {tp2:,.3f} | 레버 {lev}배 → 손절 {sl_p*lev:.1f}% / 익절 {tp1_p*lev:.1f}%</div>
</div>
""", unsafe_allow_html=True)

col_l, col_s = st.columns(2)
with col_l:
    st.markdown("### 🟢 롱 신호")
    if longs:
        for r in longs: render_card(r, "LONG")
    else:
        st.markdown("<div class='signal-card'><span style='color:#475569;'>신호 없음</span></div>", unsafe_allow_html=True)

with col_s:
    st.markdown("### 🔴 숏 신호")
    if shorts:
        for r in shorts: render_card(r, "SHORT")
    else:
        st.markdown("<div class='signal-card'><span style='color:#475569;'>신호 없음</span></div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📊 전체 종목")
rows = []
for r in sorted(results, key=lambda x: x.get("chg24",0), reverse=True):
    rows.append({
        "종목": r["symbol"],
        "신호": "🟢 LONG" if r["signal"]=="LONG" else "🔴 SHORT" if r["signal"]=="SHORT" else "⬜",
        "점수": r.get("score",0),
        "RSI": f"{r.get('rsi',0):.0f}",
        "24H%": f"{r.get('chg24',0):+.1f}%",
        "펀딩비": f"{r.get('funding',0)*100:.4f}%",
    })
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown('<div style="text-align:center;color:#334155;font-size:11px;margin-top:20px;">⚡ Gate.io Futures Scanner | 스캘핑 전용 | 투자 판단은 본인 책임</div>', unsafe_allow_html=True)
