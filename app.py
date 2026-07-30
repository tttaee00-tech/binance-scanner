"""
Gate.io Futures Scalping Scanner v3
- 캔들 파싱 수정 (o,v,t,c,l,h,sum 구조)
- 거래량 TOP50 자동 스캔
- 멀티스레드
- 1분봉 재확인
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="⚡ Scalping Scanner", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp { background: #080c14; color: #e2e8f0; }
.signal-card { background: #0f1923; border-radius: 12px; padding: 14px 16px; margin-bottom: 8px; border-left: 3px solid #334155; }
.long-card  { border-left-color: #10b981 !important; background: #0a1f17 !important; }
.short-card { border-left-color: #ef4444 !important; background: #1f0a0a !important; }
</style>
""", unsafe_allow_html=True)

GATE = "https://api.gateio.ws/api/v4"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

def gate_get(path, params=None):
    try:
        r = requests.get(f"{GATE}{path}", params=params, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            return r.json()
    except: pass
    return None

@st.cache_data(ttl=60, show_spinner=False)
def get_top50():
    data = gate_get("/futures/usdt/tickers")
    if not data: return ["BTC_USDT","ETH_USDT","SOL_USDT","BNB_USDT","XRP_USDT"]
    df = pd.DataFrame(data)
    df["vol"] = pd.to_numeric(df["volume_24h_quote"], errors="coerce").fillna(0)
    df = df[df["contract"].str.endswith("_USDT") & (df["vol"] > 1_000_000)]
    return df.nlargest(50, "vol")["contract"].tolist()

@st.cache_data(ttl=30, show_spinner=False)
def get_tickers_map():
    data = gate_get("/futures/usdt/tickers")
    if not data: return {}
    return {x["contract"]: x for x in data}

@st.cache_data(ttl=300, show_spinner=False)
def get_funding_map():
    data = gate_get("/futures/usdt/contracts")
    if not data: return {}
    return {x["name"]: float(x.get("funding_rate", 0) or 0) for x in data}

@st.cache_data(ttl=30, show_spinner=False)
def get_klines(symbol: str, interval: str, limit: int = 100):
    data = gate_get("/futures/usdt/candlesticks",
        {"contract": symbol, "interval": interval, "limit": limit})
    if not data or not isinstance(data, list): return pd.DataFrame()
    df = pd.DataFrame(data)
    # Gate.io 컬럼 구조: o(시가) v(거래량) t(시간) c(종가) l(저가) h(고가) sum
    for col in ["o","h","l","c","v"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["o","h","l","c"]).reset_index(drop=True)
    return df

def calc_ind(df):
    if df.empty or len(df) < 30: return {}
    c = df["c"]; h = df["h"]; l = df["l"]

    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=min(50,len(df)-1), adjust=False).mean()

    bm = c.rolling(20).mean(); bs = c.rolling(20).std()
    bu = bm + 2*bs; bd = bm - 2*bs
    bp = (c - bd) / (bu - bd + 1e-10)

    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean()
    ls = (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + g / ls.replace(0,1e-10)))

    tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    up=h.diff(); dn=-l.diff()
    pdm=up.where((up>dn)&(up>0),0); ndm=dn.where((dn>up)&(dn>0),0)
    pdi=100*pdm.rolling(14).mean()/atr.replace(0,1e-10)
    ndi=100*ndm.rolling(14).mean()/atr.replace(0,1e-10)
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,1e-10)
    adx=dx.rolling(14).mean()

    return {
        "price":     float(c.iloc[-1]),
        "ema_long":  float(ema9.iloc[-1])>float(ema21.iloc[-1])>float(ema50.iloc[-1]),
        "ema_short": float(ema9.iloc[-1])<float(ema21.iloc[-1])<float(ema50.iloc[-1]),
        "bb_pct":    float(bp.iloc[-1]) if not np.isnan(bp.iloc[-1]) else 0.5,
        "rsi":       float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50,
        "atr":       float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0,
        "atr_pct":   float(atr.iloc[-1]/c.iloc[-1]*100) if c.iloc[-1] else 0,
        "adx":       float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0,
    }

def score_sig(i15, i5, fund, btc_long, btc_short):
    if not i15 or not i5: return {"signal":"NONE","score":0,"reasons":[]}

    lo=[]; so=[]
    if i15.get("ema_long"):   lo.append("✅ 15분 정배열")
    if i15.get("ema_short"):  so.append("✅ 15분 역배열")
    if i5.get("atr_pct",0)>0.03:
        lo.append("✅ 변동성 OK"); so.append("✅ 변동성 OK")
    if fund < 0.001:  lo.append(f"✅ 펀딩비 {fund*100:.3f}%")
    if fund > -0.001: so.append(f"✅ 펀딩비 {fund*100:.3f}%")

    bb=i5.get("bb_pct",0.5); rsi=i5.get("rsi",50)
    if bb < 0.2 and rsi < 45: lo.append(f"✅ 볼린저 하단 RSI{rsi:.0f}")
    if bb > 0.8 and rsi > 55: so.append(f"✅ 볼린저 상단 RSI{rsi:.0f}")
    if btc_long:  lo.append("✅ BTC 롱 방향")
    if btc_short: so.append("✅ BTC 숏 방향")
    if i15.get("adx",0) > 20:
        lo.append(f"✅ ADX {i15['adx']:.0f}")
        so.append(f"✅ ADX {i15['adx']:.0f}")

    atr=i5.get("atr",0); price=i5.get("price",0)
    sl_p = atr*1.2/price*100 if price else 0

    long_ok  = i15.get("ema_long") or (bb < 0.25 and rsi < 50)
    short_ok = i15.get("ema_short") or (bb > 0.75 and rsi > 50)

    if len(lo) >= 3 and long_ok:
        return {"signal":"LONG","score":len(lo),"reasons":lo,
                "entry":price,"sl":round(price-atr*1.2,4),
                "tp1":round(price+atr*1.5,4),"tp2":round(price+atr*2.5,4),
                "rsi":rsi,"adx":i15.get("adx",0),"funding":fund,
                "atr_pct":i5.get("atr_pct",0),"sl_pct":sl_p}
    elif len(so) >= 3 and short_ok:
        return {"signal":"SHORT","score":len(so),"reasons":so,
                "entry":price,"sl":round(price+atr*1.2,4),
                "tp1":round(price-atr*1.5,4),"tp2":round(price-atr*2.5,4),
                "rsi":rsi,"adx":i15.get("adx",0),"funding":fund,
                "atr_pct":i5.get("atr_pct",0),"sl_pct":sl_p}
    return {"signal":"WAIT","score":max(len(lo),len(so)),"reasons":[],"rsi":rsi,"funding":fund}

def confirm_1m(sym, kind):
    try:
        i1 = calc_ind(get_klines(sym, "1m", 30))
        if not i1: return False, "-"
        bb=i1.get("bb_pct",0.5); rsi=i1.get("rsi",50)
        if kind=="LONG":  return bb<0.5 and rsi<65, f"BB:{bb:.2f} RSI:{rsi:.0f}"
        else:             return bb>0.5 and rsi>35, f"BB:{bb:.2f} RSI:{rsi:.0f}"
    except: return False, "-"

def scan_one(sym, fm, tm, btc_long, btc_short):
    try:
        i15 = calc_ind(get_klines(sym, "15m", 100))
        i5  = calc_ind(get_klines(sym, "5m",  60))
        fund = fm.get(sym, 0)
        tk   = tm.get(sym, {})
        chg  = float(tk.get("change_percentage",0) or 0)
        vol  = float(tk.get("volume_24h_quote",0) or 0)
        sig  = score_sig(i15, i5, fund, btc_long, btc_short)
        sig["symbol"] = sym.replace("_USDT","")
        sig["chg24"]  = chg
        sig["vol24"]  = vol
        sig["confirmed"] = False
        sig["note"] = ""
        if sig["signal"] in ("LONG","SHORT"):
            ok, note = confirm_1m(sym, sig["signal"])
            sig["confirmed"] = ok
            sig["note"] = note
            if not ok: sig["signal"] = "WAIT_CONFIRM"
        return sig
    except:
        return {"signal":"NONE","score":0,"symbol":sym.replace("_USDT",""),"reasons":[],"rsi":0,"chg24":0,"vol24":0,"funding":0}

# ── UI ──
st.markdown('<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:12px;padding:16px 20px;margin-bottom:16px;"><span style="font-size:22px;font-weight:700;">⚡ Gate.io Scalping Scanner</span><span style="font-size:11px;color:#64748b;margin-left:10px;">거래량 TOP50 | 멀티스레드 | 1분봉 재확인</span></div>', unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns([2,1,1,1])
with c1: scan_btn = st.button("⚡ 스캔", type="primary", use_container_width=True)
with c2: lev = st.selectbox("레버리지", [10,15,20], index=0)
with c3: auto = st.toggle("자동갱신")
with c4: show_all = st.toggle("미확인 포함")

if auto:
    import time; time.sleep(1); st.rerun()

if scan_btn or auto or "done" not in st.session_state:
    prog = st.progress(0, "거래량 TOP50 선정...")
    targets = get_top50()
    fm = get_funding_map()
    tm = get_tickers_map()

    prog.progress(0.1, "BTC 추세 확인...")
    btc15 = calc_ind(get_klines("BTC_USDT","15m"))
    bl = btc15.get("ema_long",False)
    bs = btc15.get("ema_short",False)

    results = []
    done_n = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(scan_one, s, fm, tm, bl, bs): s for s in targets}
        for fut in as_completed(futs):
            done_n += 1
            prog.progress(0.1+done_n/len(targets)*0.9, f"스캔 중... {done_n}/{len(targets)}")
            try: results.append(fut.result())
            except: pass

    prog.empty()
    st.session_state.update({"done":True,"results":results,
        "t":datetime.now().strftime("%H:%M:%S"),"bl":bl,"bs":bs,"tc":len(targets)})

results = st.session_state.get("results",[])
scan_t  = st.session_state.get("t","-")
bl = st.session_state.get("bl",False)
bs = st.session_state.get("bs",False)
tc = st.session_state.get("tc",0)

if show_all:
    longs  = sorted([r for r in results if r["signal"] in ("LONG","WAIT_CONFIRM") and "LONG" in str(r.get("reasons",""))], key=lambda x:x["score"], reverse=True)
    shorts = sorted([r for r in results if r["signal"] in ("SHORT","WAIT_CONFIRM") and "SHORT" in str(r.get("reasons",""))], key=lambda x:x["score"], reverse=True)
else:
    longs  = sorted([r for r in results if r["signal"]=="LONG"],  key=lambda x:x["score"], reverse=True)
    shorts = sorted([r for r in results if r["signal"]=="SHORT"], key=lambda x:x["score"], reverse=True)

bc = "#10b981" if bl else "#ef4444" if bs else "#94a3b8"
bd = "🟢 롱" if bl else "🔴 숏" if bs else "⬜ 중립"

st.markdown(f"""
<div style="display:flex;gap:10px;margin-bottom:14px;">
<div style="flex:1;background:#0f1923;border-radius:8px;padding:10px;text-align:center;">
<div style="font-size:16px;font-weight:700;color:{bc};">{bd}</div>
<div style="font-size:11px;color:#64748b;">BTC 15분 추세</div>
</div>
<div style="flex:1;background:#0f1923;border-radius:8px;padding:10px;text-align:center;">
<div style="font-size:16px;font-weight:700;color:#60a5fa;">{scan_t}</div>
<div style="font-size:11px;color:#64748b;">스캔 ({tc}개)</div>
</div>
<div style="flex:1;background:#0f1923;border-radius:8px;padding:10px;text-align:center;">
<div style="font-size:16px;font-weight:700;color:#10b981;">{len(longs)}</div>
<div style="font-size:11px;color:#64748b;">🟢 롱</div>
</div>
<div style="flex:1;background:#0f1923;border-radius:8px;padding:10px;text-align:center;">
<div style="font-size:16px;font-weight:700;color:#ef4444;">{len(shorts)}</div>
<div style="font-size:11px;color:#64748b;">🔴 숏</div>
</div>
</div>
""", unsafe_allow_html=True)

def card(r, kind, lev):
    sym=r["symbol"]; score=r["score"]
    entry=r.get("entry",0); sl=r.get("sl",0); tp1=r.get("tp1",0); tp2=r.get("tp2",0)
    rsi=r.get("rsi",0); adx=r.get("adx",0); fund=r.get("funding",0)
    atr_p=r.get("atr_pct",0); chg=r.get("chg24",0); sl_p=r.get("sl_pct",0)
    conf=r.get("confirmed",False); note=r.get("note",""); reasons=r.get("reasons",[])
    sig=r.get("signal","")
    css="long-card" if "LONG" in sig else "short-card"
    em="🚀" if "LONG" in sig else "💀"
    col="#10b981" if "LONG" in sig else "#ef4444"
    tp1_p=abs(tp1-entry)/entry*100 if entry else 0
    cb=f'<span style="background:#1e3a5f;color:#60a5fa;padding:2px 6px;border-radius:4px;font-size:10px;">✅ 1분봉 확인</span>' if conf else f'<span style="background:#2d1b00;color:#f59e0b;padding:2px 6px;border-radius:4px;font-size:10px;">⚠️ 미확인</span>'
    rh="".join([f"<div style='font-size:11px;color:#94a3b8;margin-top:2px;'>{rv}</div>" for rv in reasons[:5]])
    st.markdown(f"""
<div class="signal-card {css}">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
  <span style="font-size:16px;font-weight:700;">{em} {sym} {cb}</span>
  <span style="background:{'#065f46' if 'LONG' in sig else '#7f1d1d'};color:{col};padding:2px 10px;border-radius:4px;font-size:12px;font-weight:700;">{'LONG' if 'LONG' in sig else 'SHORT'} {score}/6</span>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px;background:#0a0f1a;border-radius:8px;padding:8px;">
  <div style="text-align:center;"><div style="font-size:10px;color:#64748b;">진입</div><div style="font-size:13px;font-weight:600;font-family:monospace;">{entry:,.3f}</div></div>
  <div style="text-align:center;"><div style="font-size:10px;color:#ef4444;">손절 -{sl_p:.2f}%</div><div style="font-size:13px;font-family:monospace;color:#ef4444;">{sl:,.3f}</div></div>
  <div style="text-align:center;"><div style="font-size:10px;color:#10b981;">익절 +{tp1_p:.2f}%</div><div style="font-size:13px;font-family:monospace;color:#10b981;">{tp1:,.3f}</div></div>
</div>
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
  <span style="font-size:11px;color:#94a3b8;">RSI {rsi:.0f}</span>
  <span style="font-size:11px;color:#94a3b8;">ADX {adx:.0f}</span>
  <span style="font-size:11px;color:#94a3b8;">ATR {atr_p:.2f}%</span>
  <span style="font-size:11px;color:#94a3b8;">펀딩 {fund*100:.3f}%</span>
  <span style="font-size:11px;color:{'#10b981' if chg>0 else '#ef4444'};">24H {chg:+.1f}%</span>
</div>
{rh}
<div style="font-size:10px;color:#475569;margin-top:4px;">{note}</div>
<div style="font-size:10px;color:#334155;margin-top:4px;">TP2: {tp2:,.3f} | 레버 {lev}배 → 손절 {sl_p*lev:.1f}% / 익절 {tp1_p*lev:.1f}%</div>
</div>""", unsafe_allow_html=True)

cl, cs = st.columns(2)
with cl:
    st.markdown("### 🟢 롱 신호")
    if longs:
        for r in longs: card(r,"LONG",lev)
    else:
        st.markdown("<div class='signal-card'><span style='color:#475569;'>신호 없음 — 미확인 포함 토글 켜보세요</span></div>", unsafe_allow_html=True)

with cs:
    st.markdown("### 🔴 숏 신호")
    if shorts:
        for r in shorts: card(r,"SHORT",lev)
    else:
        st.markdown("<div class='signal-card'><span style='color:#475569;'>신호 없음 — 미확인 포함 토글 켜보세요</span></div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📊 전체 종목 (거래량순)")
rows=[]
for r in sorted(results, key=lambda x:x.get("vol24",0), reverse=True):
    rows.append({
        "종목":r["symbol"],
        "신호":"🟢 LONG" if r["signal"]=="LONG" else "🔴 SHORT" if r["signal"]=="SHORT" else "⚠️" if r["signal"]=="WAIT_CONFIRM" else "⬜",
        "점수":r.get("score",0),
        "RSI":f"{r.get('rsi',0):.0f}",
        "24H%":f"{r.get('chg24',0):+.1f}%",
        "펀딩비":f"{r.get('funding',0)*100:.4f}%",
        "1분봉":"✅" if r.get("confirmed") else "-",
    })
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown('<div style="text-align:center;color:#334155;font-size:11px;margin-top:20px;">⚡ Gate.io Scalping Scanner | 투자 판단은 본인 책임</div>', unsafe_allow_html=True)
