import streamlit as st
import requests
from datetime import datetime, timedelta

# --- KONFIGURASI DASAR ---
LAT, LON = -7.380, 112.786
ADM4_CODE = "35.15.17.2011"  # Desa Sedati Gede
BMKG_POINT = f"https://cuaca.bmkg.go.id/api/df/v1/forecast/point?lat={LAT}&lon={LON}"
BMKG_ADM4 = f"https://cuaca.bmkg.go.id/api/df/v1/forecast/adm?adm4={ADM4_CODE}"
OPENMETEO = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m,precipitation,cloudcover,visibility,winddirection_10m,windspeed_10m&forecast_days=1"
METAR_URL = (
    "https://aviationweather.gov/adds/dataserver_current/httpparam?"
    "datasource=metars&requestType=retrieve&format=xml&stationString=WARR&hoursBeforeNow=2"
)

st.set_page_config(page_title="TAFOR WARR Generator", layout="wide")
st.title("🛫 Auto TAFOR Generator – WARR (Juanda)")

# --- STEP 1: BMKG POINT ---
st.subheader("📡 Data BMKG Point / Desa Sedati Gede")
try:
    res = requests.get(BMKG_POINT, timeout=10)
    data_bmkg = res.json()
    if "data" not in data_bmkg or not data_bmkg["data"]:
        raise ValueError("BMKG Point kosong, ambil dari Adm4...")
except Exception:
    res = requests.get(BMKG_ADM4, timeout=10)
    data_bmkg = res.json()
    st.warning("BMKG Point gagal, menggunakan data Desa Sedati Gede (adm4).")

if data_bmkg:
    st.success("✅ Data BMKG berhasil diambil.")
else:
    st.error("❌ Tidak ada data BMKG.")
    st.stop()

# --- STEP 2: Open-Meteo ---
st.subheader("🌍 Data Open-Meteo")
try:
    res2 = requests.get(OPENMETEO, timeout=10)
    data_open = res2.json()
    st.success("✅ Data Open-Meteo berhasil diambil.")
except Exception as e:
    data_open = None
    st.error(f"Gagal ambil Open-Meteo: {e}")

# --- STEP 3: METAR ---
st.subheader("🛰️ METAR WARR")
try:
    rmet = requests.get(METAR_URL, timeout=10)
    if rmet.status_code == 200:
        st.success("✅ METAR WARR berhasil diambil.")
        st.code(rmet.text[:1500], language="xml")
    else:
        st.warning("METAR gagal diambil.")
except Exception as e:
    st.error(f"METAR error: {e}")

# --- STEP 4: ANALISIS & GENERATE TAFOR ---
st.subheader("✈️ TAFOR WARR (Generated)")

# Simulasi logika sederhana untuk contoh
current_time = datetime.utcnow()
taf_start = current_time.strftime("%d%H%MZ")
taf_valid = f"{current_time.strftime('%d%H')}/{(current_time+timedelta(hours=24)).strftime('%d%H')}"

# Contoh ekstraksi parameter sederhana
wind_dir = 100
wind_speed = 10
visibility = 8000
weather = "FEW020 SCT030"
remark = "No significant weather"

# logika perubahan (dummy contoh)
taf_lines = [
    f"TAF WARR {taf_start} {taf_valid} {wind_dir:03d}{wind_speed:02d}KT {visibility} {weather}",
    f"BECMG {current_time.strftime('%d%H')}30 {wind_dir+30:03d}{wind_speed+2:02d}KT SHRA BKN015",
    f"TEMPO {current_time.strftime('%d%H')}50/{(current_time+timedelta(hours=3)).strftime('%d%H')}0 3000 TSRA BKN010",
    "FM290600 25012KT 9999 SCT020"
]

for line in taf_lines:
    st.code(line)

st.success("✅ TAFOR otomatis berhasil dibuat (contoh format ICAO).")
st.caption("Note: Versi ini masih simulasi. Nanti logika otomatis akan menyesuaikan perubahan signifikan dari data BMKG + OpenMeteo + METAR.")
