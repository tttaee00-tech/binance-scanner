"""
Binance Futures Scalping Scanner
스캘핑 전략: EMA + 볼린저 + ATR + 펀딩비 + BTC 방향 필터
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time

st.set_page_config(
    page_title="Binance Scalping Scanner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── 스타일 ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }

.stApp { background: #080c14; color: #e2e8f0; }

.signal-card {
    background: #0f1923;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
    border-left: 3px solid #334155;
    transition: all 0.2s;
}
.long-card  { border-left-color: #10b981; background: #0a1f17; }
.short-card { border-left-color: #ef4444; background: #1f0a0a; }
.wait-card  { border-left-color: #f59e0b; background: #1a1505; }

.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 4px;
}
.tag-long  { background: #065f46; color: #34d399; }
.tag-short { background: #7f1d1d; color: #fca5a5; }
.tag-wait  { background: #78350f; color: #fcd34d; }

.metric-box {
    background: #0f1923;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.metric-val { font-size: 20px; font-weight: 700; font-family: 'JetBrains Mono'; }
.metric-lbl { font-size: 11px; color: #64748b; margin-top: 2px; }

.header-bar {
    background: linear-gradient(135deg, #0f1923 0%, #131d2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── 바이낸스 API ──
BASE = "https://fapi.binance.com"

@st.cache_data(ttl=30, show_spinner=False)
def get_funding_rates() -> dict:
    try:
        r = requests.get(f"{BASE}/fapi/v1/premiumIndex", timeout=5).json()
        return {x["symbol"]: float(x["lastFundingRate"]) for x in r if "USDT" in x["symbol"]}
    except: return {}

@st.cache_data(ttl=30, show_spinner=False)
def get_tickers() -> dict:
    try:
        r = requests.get(f"{BASE}/fapi/v1/ticker/24hr", timeout=5).json()
        return {x["symbol"]: x for x in r if x["symbol"].endswith("USDT")}
    except: return {}

@st.cache_data(ttl=60, show_spinner=False)
def get_klines(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    try:
        r = requests.get(f"{BASE}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=5).json()
        df = pd.DataFrame(r, columns=["ts","o","h","l","c","v","ct","qv","n","tbv","tqv","ig"])
        for col in ["o","h","l","c","v"]:
            df[col] = df[col].astype(float)
        return df
    except: return pd.DataFrame()

def calc_indicators(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 50: return {}
    c = df["c"]
    h = df["h"]
    l = df["l"]

    # EMA
    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    # 볼린저
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    bb_pct = (c - bb_dn) / (bb_up - bb_dn)  # 0=하단, 1=상단

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # ADX
    up   = h.diff(); dn = -l.diff()
    pdm  = up.where((up > dn) & (up > 0), 0)
    ndm  = dn.where((dn > up) & (dn > 0), 0)
    pdi  = 100 * pdm.rolling(14).mean() / atr.replace(0, 1e-10)
    ndi  = 100 * ndm.rolling(14).mean() / atr.replace(0, 1e-10)
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, 1e-10)
    adx  = dx.rolling(14).mean()

    last = -1
    return {
        "price":   float(c.iloc[last]),
        "ema9":    float(ema9.iloc[last]),
        "ema21":   float(ema21.iloc[last]),
        "ema50":   float(ema50.iloc[last]),
        "bb_pct":  float(bb_pct.iloc[last]),
        "bb_up":   float(bb_up.iloc[last]),
        "bb_dn":   float(bb_dn.iloc[last]),
        "rsi":     float(rsi.iloc[last]),
        "atr":     float(atr.iloc[last]),
        "atr_pct": float(atr.iloc[last] / c.iloc[last] * 100),
        "adx":     float(adx.iloc[last]),
        "vol_ratio": float(df["v"].iloc[-5:].mean() / df["v"].iloc[-20:].mean()) if len(df) >= 20 else 1.0,
        "ema_long":  float(ema9.iloc[last]) > float(ema21.iloc[last]) > float(ema50.iloc[last]),
        "ema_short": float(ema9.iloc[last]) < float(ema21.iloc[last]) < float(ema50.iloc[last]),
    }

def score_signal(ind15: dict, ind5: dict, funding: float, btc_long: bool, btc_short: bool) -> dict:
    """A등급 진입 체크리스트 기반 신호 판단"""
    if not ind15 or not ind5:
        return {"signal": "NONE", "score": 0, "reasons": []}

    long_checks  = []
    short_checks = []

    # ✅ 1. 15분봉 EMA 추세
    if ind15.get("ema_long"):  long_checks.append("✅ 15분 정배열")
    else:                       short_checks.append("✅ 15분 역배열") if ind15.get("ema_short") else None

    # ✅ 2. ATR 충분 (변동성)
    atr_ok = ind5.get("atr_pct", 0) > 0.05
    if atr_ok:
        long_checks.append("✅ 변동성 충분")
        short_checks.append("✅ 변동성 충분")

    # ✅ 3. 펀딩비 필터
    funding_long_ok  = funding < 0.001   # 0.1% 미만
    funding_short_ok = funding > -0.001
    if funding_long_ok:  long_checks.append(f"✅ 펀딩비 OK ({funding*100:.3f}%)")
    if funding_short_ok: short_checks.append(f"✅ 펀딩비 OK ({funding*100:.3f}%)")

    # ✅ 4. 5분봉 볼린저 반전
    bb = ind5.get("bb_pct", 0.5)
    rsi = ind5.get("rsi", 50)
    if bb < 0.15 and rsi < 40: long_checks.append("✅ 볼린저 하단 반전")
    if bb > 0.85 and rsi > 60: short_checks.append("✅ 볼린저 상단 반전")

    # ✅ 5. BTC 방향 일치
    if btc_long:  long_checks.append("✅ BTC 방향 일치")
    if btc_short: short_checks.append("✅ BTC 방향 일치")

    # ADX 추세 강도
    adx = ind15.get("adx", 0)
    adx_ok = adx > 20
    if adx_ok:
        long_checks.append(f"✅ ADX {adx:.0f} (추세 강함)")
        short_checks.append(f"✅ ADX {adx:.0f} (추세 강함)")

    long_score  = len(long_checks)
    short_score = len(short_checks)

    atr = ind5.get("atr", 0)
    price = ind5.get("price", 0)

    if long_score >= 4 and ind15.get("ema_long") and bb < 0.2:
        return {
            "signal": "LONG",
            "score":  long_score,
            "reasons": long_checks,
            "entry":  price,
            "sl":     round(price - atr * 1.2, 4),
            "tp1":    round(price + atr * 1.5, 4),
            "tp2":    round(price + atr * 2.5, 4),
            "rsi":    rsi,
            "atr_pct": ind5.get("atr_pct", 0),
            "adx":    adx,
            "funding": funding,
        }
    elif short_score >= 4 and ind15.get("ema_short") and bb > 0.8:
        return {
            "signal": "SHORT",
            "score":  short_score,
            "reasons": short_checks,
            "entry":  price,
            "sl":     round(price + atr * 1.2, 4),
            "tp1":    round(price - atr * 1.5, 4),
            "tp2":    round(price - atr * 2.5, 4),
            "rsi":    rsi,
            "atr_pct": ind5.get("atr_pct", 0),
            "adx":    adx,
            "funding": funding,
        }
    else:
        return {"signal": "WAIT", "score": max(long_score, short_score), "reasons": [], "rsi": rsi}

# ── 대상 종목 ──
TARGETS = [
    "BTCUSDT","ETHUSDT",
    "BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
    "AVAXUSDT","DOTUSDT","LINKUSDT","MATICUSDT","LTCUSDT",
    "UNIUSDT","ATOMUSDT","NEARUSDT","APTUSDT","ARBUSDT",
    "OPUSDT","INJUSDT","SUIUSDT","TIAUSDT","SEIUSDT",
]

# ── 헤더 ──
st.markdown("""
<div class="header-bar">
<span style="font-size:22px;font-weight:700;">⚡ Binance Futures Scanner</span>
<span style="font-size:12px;color:#64748b;margin-left:12px;">스캘핑 | EMA+볼린저+ATR+펀딩비 | A등급 전략</span>
</div>
""", unsafe_allow_html=True)

# ── 자동 갱신 ──
col_h1, col_h2, col_h3 = st.columns([2,1,1])
with col_h1:
    auto = st.toggle("🔄 30초 자동갱신", value=False)
with col_h2:
    scan_btn = st.button("⚡ 스캔", type="primary", use_container_width=True)
with col_h3:
    lev = st.selectbox("레버리지", [10, 15, 20], index=0)

if auto:
    time.sleep(0.1)
    st.rerun()

# ── 스캔 ──
if scan_btn or auto or "scan_result" not in st.session_state:
    with st.spinner("스캔 중..."):
        funding_map = get_funding_rates()
        tickers     = get_tickers()

        # BTC 방향 판단
        btc15 = calc_indicators(get_klines("BTCUSDT", "15m"))
        btc_long  = btc15.get("ema_long",  False)
        btc_short = btc15.get("ema_short", False)

        results = []
        for sym in TARGETS:
            try:
                ind15 = calc_indicators(get_klines(sym, "15m"))
                ind5  = calc_indicators(get_klines(sym, "5m", 60))
                fund  = funding_map.get(sym, 0)
                sig   = score_signal(ind15, ind5, fund, btc_long, btc_short)
                sig["symbol"] = sym.replace("USDT","")
                tk = tickers.get(sym, {})
                sig["chg24"] = float(tk.get("priceChangePercent", 0))
                sig["vol24"] = float(tk.get("quoteVolume", 0))
                results.append(sig)
            except: pass

        st.session_state["scan_result"] = results
        st.session_state["scan_time"]   = datetime.now().strftime("%H:%M:%S")
        st.session_state["btc_long"]    = btc_long
        st.session_state["btc_short"]   = btc_short

results  = st.session_state.get("scan_result", [])
scan_t   = st.session_state.get("scan_time", "-")
btc_long  = st.session_state.get("btc_long", False)
btc_short = st.session_state.get("btc_short", False)

# ── BTC 방향 + 시간 ──
btc_dir = "🟢 롱 우위" if btc_long else "🔴 숏 우위" if btc_short else "⬜ 중립"
st.markdown(f"""
<div style="display:flex;gap:12px;margin-bottom:14px;align-items:center;">
<div class="metric-box" style="flex:1;">
<div class="metric-val" style="color:{'#10b981' if btc_long else '#ef4444' if btc_short else '#94a3b8'};">{btc_dir}</div>
<div class="metric-lbl">BTC 15분봉 추세</div>
</div>
<div class="metric-box" style="flex:1;">
<div class="metric-val" style="color:#60a5fa;">{scan_t}</div>
<div class="metric-lbl">마지막 스캔</div>
</div>
<div class="metric-box" style="flex:1;">
<div class="metric-val">{len([r for r in results if r['signal']=='LONG'])}</div>
<div class="metric-lbl">롱 신호</div>
</div>
<div class="metric-box" style="flex:1;">
<div class="metric-val">{len([r for r in results if r['signal']=='SHORT'])}</div>
<div class="metric-lbl">숏 신호</div>
</div>
</div>
""", unsafe_allow_html=True)

# ── 신호 분류 ──
longs  = [r for r in results if r["signal"] == "LONG"]
shorts = [r for r in results if r["signal"] == "SHORT"]
waits  = [r for r in results if r["signal"] == "WAIT"]

longs.sort(key=lambda x: x["score"], reverse=True)
shorts.sort(key=lambda x: x["score"], reverse=True)

col_l, col_s = st.columns(2)

def render_card(r, kind):
    sym   = r["symbol"]
    score = r["score"]
    entry = r.get("entry", 0)
    sl    = r.get("sl",    0)
    tp1   = r.get("tp1",   0)
    tp2   = r.get("tp2",   0)
    rsi   = r.get("rsi",   0)
    adx   = r.get("adx",   0)
    fund  = r.get("funding", 0)
    atr_p = r.get("atr_pct", 0)
    chg   = r.get("chg24",  0)
    reasons = r.get("reasons", [])

    css   = "long-card" if kind == "LONG" else "short-card"
    tag   = "tag-long"  if kind == "LONG" else "tag-short"
    label = "🟢 LONG"   if kind == "LONG" else "🔴 SHORT"
    emoji = "🚀" if kind == "LONG" else "💀"

    # 레버리지 기준 손절%
    sl_pct = abs(entry - sl) / entry * 100 if entry else 0
    tp1_pct = abs(tp1 - entry) / entry * 100 if entry else 0

    reasons_html = "".join([f"<div style='font-size:11px;color:#94a3b8;'>{r}</div>" for r in reasons[:4]])

    st.markdown(f"""
<div class="signal-card {css}">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
  <span style="font-size:15px;font-weight:700;">{emoji} {sym}</span>
  <span class="tag {tag}">{label} {score}/6</span>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px;">
  <div style="text-align:center;">
    <div style="font-size:10px;color:#64748b;">진입가</div>
    <div style="font-size:12px;font-weight:600;font-family:monospace;">{entry:,.4f}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:10px;color:#ef4444;">손절 -{sl_pct:.2f}%</div>
    <div style="font-size:12px;font-weight:600;font-family:monospace;color:#ef4444;">{sl:,.4f}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:10px;color:#10b981;">익절 +{tp1_pct:.2f}%</div>
    <div style="font-size:12px;font-weight:600;font-family:monospace;color:#10b981;">{tp1:,.4f}</div>
  </div>
</div>
<div style="display:flex;gap:8px;margin-bottom:6px;">
  <span style="font-size:11px;color:#94a3b8;">RSI {rsi:.0f}</span>
  <span style="font-size:11px;color:#94a3b8;">ADX {adx:.0f}</span>
  <span style="font-size:11px;color:#94a3b8;">ATR {atr_p:.2f}%</span>
  <span style="font-size:11px;color:#94a3b8;">펀딩 {fund*100:.3f}%</span>
  <span style="font-size:11px;color:{'#10b981' if chg>0 else '#ef4444'};">24H {chg:+.1f}%</span>
</div>
{reasons_html}
<div style="font-size:10px;color:#475569;margin-top:6px;">TP2: {tp2:,.4f} | 레버 {lev}배 기준 손절 {sl_pct*lev:.1f}%</div>
</div>
""", unsafe_allow_html=True)

with col_l:
    st.markdown("### 🟢 롱 신호")
    if longs:
        for r in longs:
            render_card(r, "LONG")
    else:
        st.markdown("<div class='signal-card'><span style='color:#475569;'>롱 신호 없음</span></div>", unsafe_allow_html=True)

with col_s:
    st.markdown("### 🔴 숏 신호")
    if shorts:
        for r in shorts:
            render_card(r, "SHORT")
    else:
        st.markdown("<div class='signal-card'><span style='color:#475569;'>숏 신호 없음</span></div>", unsafe_allow_html=True)

# ── 전체 종목 상태 ──
st.markdown("---")
st.markdown("### 📊 전체 종목")
wait_data = []
for r in sorted(results, key=lambda x: x.get("chg24",0), reverse=True):
    rsi  = r.get("rsi", 0)
    sig  = r.get("signal","")
    chg  = r.get("chg24", 0)
    fund = r.get("funding", 0)
    wait_data.append({
        "종목": r["symbol"],
        "신호": "🟢 LONG" if sig=="LONG" else "🔴 SHORT" if sig=="SHORT" else "⬜",
        "점수": r.get("score",0),
        "RSI": f"{rsi:.0f}",
        "24H%": f"{chg:+.1f}%",
        "펀딩비": f"{fund*100:.3f}%",
    })

if wait_data:
    st.dataframe(pd.DataFrame(wait_data), use_container_width=True, hide_index=True)

st.markdown(f"""
<div style="text-align:center;color:#334155;font-size:11px;margin-top:20px;">
⚡ Binance Futures Scanner | 스캘핑 전용 | 투자 판단은 본인 책임
</div>
""", unsafe_allow_html=True)
