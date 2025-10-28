import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="TAFOR WARR Generator", layout="wide")

# ====== HEADER ======
st.title("✈️ Automatic TAFOR Generator – WARR (Juanda)")
st.markdown("Aplikasi ini menghasilkan **TAFOR WARR (Juanda)** otomatis berdasarkan data BMKG & METAR real-time.")

# ====== PILIHAN INPUT ======
col1, col2, col3 = st.columns(3)
with col1:
    tanggal = st.date_input("Tanggal Prakiraan", datetime.now())
with col2:
    jam_mulai = st.time_input("Jam Mulai (UTC)", datetime.utcnow().replace(minute=0, second=0, microsecond=0))
with col3:
    durasi = st.selectbox("Durasi TAF (jam)", [6, 9, 12, 18, 24], index=2)

# ====== PENGAMBILAN DATA ======
st.divider()
st.subheader("📡 Ambil Data Otomatis")

# API BMKG (contoh endpoint open-data)
bmkg_url = "https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=3578040"  # Kode wilayah Juanda (Sidoarjo)
metar_url = "https://aviationweather.gov/api/data/metar?ids=WARR&format=JSON"

colA, colB = st.columns(2)

with colA:
    if st.button("🔄 Ambil Data BMKG"):
        try:
            bmkg_data = requests.get(bmkg_url).json()
            st.success("Data BMKG berhasil diambil ✅")
            st.json(bmkg_data)
        except Exception as e:
            st.error(f"Gagal mengambil data BMKG: {e}")

with colB:
    if st.button("🛰️ Ambil METAR WARR"):
        try:
            metar_data = requests.get(metar_url).json()
            st.success("Data METAR berhasil diambil ✅")
            st.json(metar_data)
        except Exception as e:
            st.error(f"Gagal mengambil METAR: {e}")

st.divider()

# ====== PEMBUATAN TAFOR OTOMATIS ======
st.subheader("✈️ Hasil TAFOR Otomatis")

def generate_tafor(tgl, jam, durasi):
    start = datetime.combine(tgl, jam)
    end = start + timedelta(hours=durasi)
    valid_period = f"{start.strftime('%d%H')}/{end.strftime('%d%H')}"

    # (Untuk contoh: cuaca statis, nanti bisa diubah sesuai data BMKG)
    tafor_text = f"""TAF WARR {start.strftime('%d%H%M')}Z {valid_period}
    09010KT 8000 FEW018 SCT025
    TEMPO 1000 SHRA
    BECMG { (start + timedelta(hours=3)).strftime('%d%H%M') } 09012KT 6000 -RA SCT020"""
    return tafor_text

if st.button("🧭 Generate TAFOR WARR"):
    tafor_output = generate_tafor(tanggal, jam_mulai, durasi)
    st.code(tafor_output, language="yaml")

    st.download_button(
        label="💾 Unduh TAFOR (TXT)",
        data=tafor_output,
        file_name=f"TAFOR_WARR_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

st.divider()
st.markdown("🪶 **Catatan:** Versi awal ini dapat dikembangkan dengan pengambilan data otomatis dari BMKG (cuaca, suhu, visibilitas, arah angin) dan integrasi AI untuk prediksi TAF lebih akurat.")
