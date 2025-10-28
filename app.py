import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="TAFOR WARR Generator", layout="wide")

st.title("✈️ Automatic TAFOR Generator – WARR (Juanda)")
st.markdown("Menggabungkan data **BMKG API (v1)** dan **METAR WARR** untuk membuat TAFOR otomatis.")

# ===================
# PARAMETER DASAR
# ===================
lat_ref = -7.380
lon_ref = 112.786
radius_km = 10

tanggal = st.date_input("Tanggal Prakiraan", datetime.now())
jam_mulai = st.time_input("Jam Mulai (UTC)", datetime.utcnow().replace(minute=0, second=0, microsecond=0))
durasi = st.selectbox("Durasi TAF (jam)", [6, 9, 12, 18, 24], index=2)

# ===================
# FUNGSI JARAK
# ===================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ===================
# AMBIL DATA BMKG
# ===================
st.subheader("📡 Data BMKG (API v1)")
bmkg_url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm?adm1=35"

if st.button("🔄 Ambil Data BMKG"):
    try:
        r = requests.get(bmkg_url, timeout=20)
        data = r.json()

        if "data" not in data or len(data["data"]) == 0:
            st.error("❌ Tidak ada data ditemukan dari API BMKG.")
        else:
            df_list = []
            for item in data["data"]:
                lat = item.get("lat")
                lon = item.get("lon")
                dist = haversine(lat_ref, lon_ref, lat, lon)
                if dist <= radius_km:
                    df_list.append({
                        "lokasi": item.get("lokasi"),
                        "lat": lat,
                        "lon": lon,
                        "jarak_km": round(dist, 2),
                        "cuaca": item.get("cuaca"),
                        "suhu": item.get("t"),
                        "kelembapan": item.get("hu"),
                        "arah_angin": item.get("wd"),
                        "kecepatan_angin": item.get("ws")
                    })

            if len(df_list) == 0:
                st.warning(f"Tidak ada lokasi dalam radius {radius_km} km dari Juanda.")
            else:
                df = pd.DataFrame(df_list)
                st.success(f"Ditemukan {len(df)} lokasi dalam radius {radius_km} km dari Juanda.")
                st.dataframe(df)

    except Exception as e:
        st.error(f"Gagal mengambil data BMKG: {e}")

# ===================
# AMBIL METAR WARR
# ===================
st.subheader("🛰️ METAR WARR (Aviation Weather)")
metar_url = "https://aviationweather.gov/api/data/metar?ids=WARR&format=JSON"

if st.button("📡 Ambil METAR WARR"):
    try:
        metar_data = requests.get(metar_url, timeout=15).json()
        if "features" in metar_data:
            metar_text = metar_data["features"][0]["properties"]["rawOb"]
            st.code(metar_text, language="text")
        else:
            st.warning("METAR tidak tersedia.")
    except Exception as e:
        st.error(f"Gagal mengambil METAR: {e}")

# ===================
# GENERATE TAFOR
# ===================
st.subheader("✈️ Generate TAFOR Otomatis")

def generate_tafor(tgl, jam, durasi):
    start = datetime.combine(tgl, jam)
    end = start + timedelta(hours=durasi)
    valid_period = f"{start.strftime('%d%H')}/{end.strftime('%d%H')}"

    tafor_text = f"""TAF WARR {start.strftime('%d%H%M')}Z {valid_period}
    09010KT 8000 FEW018 SCT025
    TEMPO 1000 SHRA
    BECMG { (start + timedelta(hours=3)).strftime('%d%H%M') } 09012KT 6000 -RA SCT020"""
    return tafor_text

if st.button("🧭 Buat TAFOR WARR"):
    tafor_output = generate_tafor(tanggal, jam_mulai, durasi)
    st.code(tafor_output, language="yaml")
    st.download_button(
        label="💾 Unduh TAFOR (TXT)",
        data=tafor_output,
        file_name=f"TAFOR_WARR_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

st.markdown("---")
st.caption("Versi fix API BMKG v1 + radius filtering ±10 km Juanda")
