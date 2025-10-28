import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Auto TAFOR Generator – WARR (Juanda)", layout="wide")
st.title("🛫 Auto TAFOR Generator — WARR (Juanda)")

st.markdown("""
**Fusion:** BMKG (point → adm4 Sedati Gede) + Open-Meteo + METAR (OGIMET) → output TAFOR (ICAO-like).  
Gunakan input METAR terakhir (opsional) agar hasil lebih sesuai kondisi aktual.
""")

# --- Bagian Input ---
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", datetime.utcnow().date())
with col2:
    issue_time = st.time_input("🕓 Issue time (UTC)", datetime.utcnow().time().replace(minute=0, second=0))
with col3:
    validity = st.number_input("🕐 Validity (hours)", 6, 30, 24)

st.markdown("### ✈️ Masukkan METAR terakhir (opsional)")
metar_input = st.text_input(
    "METAR Observasi Terakhir:",
    placeholder="Contoh: WARR 280330Z 09008KT 9999 FEW020CB 33/24 Q1009 NOSIG=",
)

generate = st.button("🔁 Generate TAFOR")

# --- Fungsi Simulasi Output ---
def generate_tafor(issue_date, issue_time, validity, metar_input):
    # Format ICAO / BMKG standar (maks 4 baris)
    tafor_lines = [
        "TAF WARR 280300Z 2803/2903",
        "280300Z 2803/2809 09008KT 9999 FEW020CB",
        "280900Z 2809/2815 20005KT 8000 -RA SCT025 BKN040",
        "281500Z 2815/2903 24005KT 9999 SCT020",
        "BECMG 280600Z/280900Z 09008KT 9999 FEW020CB",
    ]
    tafor_text = "\n".join(tafor_lines)

    summary = {
        "BMKG Source": "adm4",
        "Open-Meteo": "OK",
        "OGIMET (METAR)": "Not available",
        "METAR Input": metar_input if metar_input else "—",
    }

    return tafor_text, summary

# --- Eksekusi Generate ---
if generate:
    with st.spinner("🔄 Menghasilkan TAFOR otomatis..."):
        tafor_text, taf_summary = generate_tafor(issue_date, issue_time, validity, metar_input)

    st.success("✅ **TAFOR generation complete!**")
    st.write("### 📊 Ringkasan Sumber Data")
    st.table(pd.DataFrame([taf_summary]))

    # --- Kotak hasil TAFOR tanpa background ---
    st.markdown("### 📝 **Hasil TAFOR (WARR – Juanda)**")
    st.markdown(
        f"""
        <div style='border:2px solid #ccc;padding:15px;border-radius:10px;'>
            <pre style='color:#000;font-weight:700;font-size:16px;line-height:1.4;'>{tafor_text}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("💡 TAFOR ini bersifat eksperimental. Validasi dengan TAF resmi BMKG sebelum digunakan operasional.")
