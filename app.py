import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Auto TAFOR + TREND — WARR", layout="wide")

st.title("🛫 Auto TAFOR + TREND — WARR (Juanda)")
st.markdown("Fusion: METAR (OGIMET/NOAA) + Open-Meteo (+BMKG optional). Output: TAF-like + TREND otomatis + grafik.")

# =======================
# 🔹 INPUT DATA
# =======================
icao = "WARR"
validity = st.number_input("Periode validitas TAF (jam)", 6, 30, 9)
issue_dt = st.datetime_input("Issue Time (UTC)", value=datetime.utcnow())

# Ambil METAR
st.subheader("METAR Terbaru")
metar_url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{icao}.TXT"
metar_raw = requests.get(metar_url).text.strip()
st.code(metar_raw)

lines = metar_raw.splitlines()
metar_time_str, metar_report = lines[-2], lines[-1]
obs_hour = int(metar_report[2:4])
obs_min = int(metar_report[4:6])
metar_dt = datetime.utcnow().replace(hour=obs_hour, minute=obs_min, second=0, microsecond=0)

parts = metar_report.split()
wind = next((p for p in parts if p.endswith("KT")), "VRB02KT")
vis = next((p for p in parts if p.endswith("SM") or p.isdigit()), "8000")
weather = next((p for p in parts if p in ["RA", "-RA", "TS", "SHRA", "DZ", "BR", "FG", "NSW"]), "NSW")
cloud = next((p for p in parts if p.startswith(("FEW", "SCT", "BKN", "OVC"))), "FEW020")
temp = next((p for p in parts if "/" in p), "27/24")

# Ambil data Open-Meteo (untuk TREND)
lat, lon = -7.3798, 112.7870  # Juanda
url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,cloudcover,visibility,windspeed_10m&forecast_days=1"
r = requests.get(url).json()
df = pd.DataFrame(r['hourly'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')

# Ambil kondisi 2 jam ke depan dari waktu observasi
end_trend = metar_dt + timedelta(hours=2)
trend_df = df[(df.index >= metar_dt) & (df.index <= end_trend)]

# Analisis perubahan
trend_text = "NOSIG"
if not trend_df.empty:
    vis_now = float(vis.replace("SM", "")) * 1609 if "SM" in vis else float(vis)
    vis_pred = trend_df['visibility'].iloc[-1]
    precip = trend_df['precipitation'].max()
    cloud_pred = trend_df['cloudcover'].iloc[-1]
    wind_pred = trend_df['windspeed_10m'].iloc[-1]

    if abs(vis_pred - vis_now) > 3000 or precip > 0.5:
        trend_text = f"TEMPO {end_trend.strftime('%H%M')} RA"
    elif cloud_pred > 70 or wind_pred > 15:
        trend_text = f"BECMG {end_trend.strftime('%H%M')} BKN020"

# =======================
# 🔹 OUTPUT TAF + TREND
# =======================
valid_from = issue_dt.strftime("%d%H")
valid_to = (issue_dt + timedelta(hours=validity)).strftime("%d%H")

taf = f"TAF {icao} {issue_dt.strftime('%d%H%M')}Z {valid_from}/{valid_to} {wind} {vis} {weather} {cloud}"
st.subheader("🔸 Generated TAF")
st.code(taf)

st.subheader("🔹 Auto TREND (2 jam ke depan)")
st.code(f"{icao} {trend_text}")

# =======================
# 🔹 Grafik Visualisasi
# =======================
st.subheader("📈 Visualisasi Perubahan (2 jam ke depan)")
st.line_chart(trend_df[['temperature_2m', 'visibility', 'precipitation', 'windspeed_10m']])
