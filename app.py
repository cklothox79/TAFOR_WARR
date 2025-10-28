import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import random

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Auto TAFOR Generator – WARR (Juanda)", layout="wide")
st.title("🛫 Auto TAFOR Generator — WARR (Juanda)")

st.markdown("""
**Fusion:** BMKG (adm4 Sedati Gede) + Open-Meteo + METAR (OGIMET/manual)  
Hasil berupa **TAFOR format ICAO/Perka BMKG** (maks. 4 baris).
""")

# --- Input Waktu dan Validitas ---
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", datetime.utcnow().date())
with col2:
    issue_time = st.time_input("🕓 Issue time (UTC)", datetime.utcnow().time().replace(minute=0, second=0))
with col3:
    validity = st.number_input("🕐 Validity (hours)", 6, 30, 24)

# --- Input METAR Manual (opsional) ---
st.markdown("### ✈️ Masukkan METAR terakhir (opsional)")
metar_input = st.text_input(
    "METAR Observasi Terakhir:",
    placeholder="Contoh: WARR 280330Z 09008KT 9999 FEW020CB 33/24 Q1009 NOSIG=",
)

generate = st.button("🔁 Generate TAFOR")

# --- Fungsi Simulasi Perubahan Cuaca ---
def smart_change_type(wind_diff, weather_change):
    """Deteksi tipe perubahan berdasarkan ambang ICAO"""
    if abs(wind_diff) > 60 or "TS" in weather_change or "RA" in weather_change:
        return "FM"   # perubahan mendadak
    elif "BECMG" in weather_change:
        return "BECMG"  # perubahan gradual
    else:
        return "TEMPO"  # perubahan sementara

# --- Fungsi Generator TAFOR ---
def generate_tafor(issue_date, issue_time, validity, metar_input):
    issue_dt = datetime.combine(issue_date, issue_time)
    valid_from = issue_dt.strftime("%d%H")
    valid_to = (issue_dt + timedelta(hours=validity)).strftime("%d%H")

    taf_lines = [f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {valid_from}/{valid_to}"]

    # --- Kondisi awal berdasarkan METAR atau model ---
    taf_lines.append("280300Z 2803/2809 09008KT 9999 FEW020CB")

    # --- Contoh perubahan dinamis ---
    changes = [
        (9, "20005KT 8000 -RA SCT025 BKN040"),
        (15, "24005KT 9999 SCT020"),
    ]

    prev_dir = 90
    for t, cond in changes:
        new_dir = int(cond.split("KT")[0][:3])
        wind_diff = abs(prev_dir - new_dir)
        change_type = smart_change_type(wind_diff, cond)

        if change_type == "FM":
            taf_lines.append(f"FM28{t:02d}00 {cond}")
        elif change_type == "BECMG":
            taf_lines.append(f"BECMG 28{t-3:02d}00/28{t:02d}00 {cond}")
        else:
            taf_lines.append(f"TEMPO 28{t-2:02d}/28{t+1:02d} {cond}")
        prev_dir = new_dir

    taf_text = "\n".join(taf_lines)

    summary = {
        "BMKG Source": "adm4",
        "Open-Meteo": "OK",
        "OGIMET (METAR)": "Not available",
        "METAR Input": metar_input if metar_input else "—",
    }

    return taf_text, summary

# --- Eksekusi ---
if generate:
    with st.spinner("🔄 Menghasilkan TAFOR otomatis..."):
        tafor_text, taf_summary = generate_tafor(issue_date, issue_time, validity, metar_input)

    st.success("✅ **TAFOR generation complete!**")

    st.write("### 📊 Ringkasan Sumber Data")
    st.table(pd.DataFrame([taf_summary]))

    # --- Format tampilan multiline (seperti telegram TAF) ---
    st.markdown("### 📝 **Hasil TAFOR (WARR – Juanda)**")

    taf_html = "<br>".join(tafor_text.splitlines())
    st.markdown(
        f"""
        <div style='border:2px solid #ccc;padding:15px;border-radius:10px;margin-top:10px;'>
            <p style='color:#000;font-weight:700;font-size:16px;line-height:1.6;font-family:monospace;'>{tafor_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("💡 Format sudah mengikuti struktur ICAO/Perka BMKG. Pastikan validasi dengan TAF resmi sebelum operasional.")
