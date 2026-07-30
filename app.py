"""
Gate.io API 연결 디버그 + 스캐너
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
        return r.status_code, r.json() if r.status_code == 200 else r.text
    except Exception as e:
        return 0, str(e)

# ── 헤더 ──
st.markdown('<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:12px;padding:16px 20px;margin-bottom:16px;"><span style="font-size:22px;font-weight:700;">⚡ Gate.io Scalping Scanner</span></div>', unsafe_allow_html=True)

# ── API 연결 테스트 ──
st.markdown("### 🔌 API 연결 상태")
col1, col2, col3 = st.columns(3)

with col1:
    status, data = gate_get("/futures/usdt/tickers", {"limit": 1})
    if status == 200:
        st.success(f"✅ Tickers: 연결됨")
    else:
        st.error(f"❌ Tickers: {status} / {str(data)[:80]}")

with col2:
    status2, data2 = gate_get("/futures/usdt/candlesticks",
        {"contract": "BTC_USDT", "interval": "5m", "limit": 5})
    if status2 == 200 and isinstance(data2, list):
        st.success(f"✅ Klines: {len(data2)}개 수신")
        st.caption(f"첫 봉: {data2[0] if data2 else 'empty'}")
    else:
        st.error(f"❌ Klines: {status2} / {str(data2)[:80]}")

with col3:
    status3, data3 = gate_get("/futures/usdt/contracts", {"limit": 1})
    if status3 == 200:
        st.success(f"✅ Contracts: 연결됨")
    else:
        st.error(f"❌ Contracts: {status3} / {str(data3)[:80]}")

# ── BTC 실시간 가격 확인 ──
st.markdown("### 💰 BTC 실시간 데이터")
status4, data4 = gate_get("/futures/usdt/tickers", {"contract": "BTC_USDT"})
if status4 == 200 and isinstance(data4, list) and data4:
    tk = data4[0]
    st.json({
        "price": tk.get("last"),
        "change_24h": tk.get("change_percentage"),
        "volume_24h": tk.get("volume_24h_quote"),
        "funding_rate": tk.get("funding_rate"),
    })
else:
    st.error(f"BTC 데이터 없음: {status4}")

# ── 캔들 데이터 구조 확인 ──
st.markdown("### 📊 캔들 데이터 구조")
if status2 == 200 and isinstance(data2, list) and data2:
    st.write("컬럼 키:", list(data2[0].keys()) if isinstance(data2[0], dict) else f"리스트 형태: {data2[0]}")
    st.dataframe(pd.DataFrame(data2[:3]))
else:
    st.error("캔들 데이터 없음")
