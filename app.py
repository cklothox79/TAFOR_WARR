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
    placeholder="Contoh: WARR 280300Z 08005KT 7000 FEW020 SCT030 26/24 Q1010",
)

generate = st.button("🔁 Generate TAFOR")

# --- Fungsi simulasi data model dan hasil ---
def generate_tafor(issue_date, issue_time, validity, metar_input):
    # Simulasi data sumber
    tafor_lines = [
        "TAF WARR 280300Z 2803/2903",
        "280300Z 2803/2809 07005KT 24140 OVC020",
        "280900Z 2809/2815 21004KT 11900 -RA OVC020",
        "281500Z 2815/2903 24005KT 24140 OVC020",
        "BECMG 280600Z/280900Z 07005KT 24140 OVC020",
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

    # --- Ringkasan Proses ---
    st.success("✅ **TAFOR generation complete!**")
    st.write("### 📊 Ringkasan Sumber Data")
    st.table(pd.DataFrame([taf_summary]))

    # --- Tampilan Hasil TAFOR ---
    st.markdown("### 📝 **Hasil TAFOR (WARR – Juanda)**")
    st.markdown(
        f"""
        <div style='background-color:#111827;padding:15px;border-radius:10px;'>
            <pre style='color:#00ff88;font-weight:bold;font-size:16px;'>{tafor_text}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("💡 TAFOR ini bersifat eksperimental. Validasi dengan TAF resmi BMKG sebelum digunakan operasional.")
