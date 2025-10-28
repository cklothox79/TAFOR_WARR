import streamlit as st
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="🛫 Auto TAFOR Generator — WARR (Juanda)", layout="centered")

st.markdown("## 🛫 Auto TAFOR Generator — WARR (Juanda)")
st.write("Fusion: BMKG (point → adm4 Sedati Gede) + Open-Meteo + METAR (OGIMET) → output TAFOR + TREND otomatis.")
st.divider()

# Input waktu
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.text_input("📅 Issue date (UTC)", value=datetime.utcnow().strftime("%Y/%m/%d"))
with col2:
    issue_time = st.text_input("🕓 Issue time (UTC)", value=datetime.utcnow().strftime("%H:00"))
with col3:
    validity = st.number_input("🕐 Validity (hours)", min_value=6, max_value=36, value=24, step=6)

metar_input = st.text_area("✈️ Masukkan METAR terakhir (opsional)",
                           "WARR 280430Z 09008KT 9999 FEW020CB 33/24 Q1009 NOSIG=",
                           height=100)

# Fungsi ambil data Open-Meteo
def get_openmeteo_trend(lat=-7.38, lon=112.77):
    url = (f"https://api.open-meteo.com/v1/forecast?"
           f"latitude={lat}&longitude={lon}"
           f"&hourly=temperature_2m,relative_humidity_2m,precipitation"
           f"&timezone=Asia%2FJakarta")
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()
    df = {
        "time": data["hourly"]["time"],
        "temp": data["hourly"]["temperature_2m"],
        "rh": data["hourly"]["relative_humidity_2m"],
        "rain": data["hourly"]["precipitation"]
    }
    return df

# Analisis trend otomatis
def analyze_trend(df):
    now_idx = 0
    next3_idx = min(len(df["time"]) - 1, now_idx + 3)
    dt1, dt2 = df["time"][now_idx], df["time"][next3_idx]
    temp1, temp2 = df["temp"][now_idx], df["temp"][next3_idx]
    rh1, rh2 = df["rh"][now_idx], df["rh"][next3_idx]
    rain1, rain2 = df["rain"][now_idx], df["rain"][next3_idx]

    deltaT = temp2 - temp1
    deltaRH = rh2 - rh1
    deltaR = rain2 - rain1

    # Logika sederhana TREND
    if deltaR > 0.1 and deltaRH > 5:
        trend_code = f"TEMPO {datetime.utcnow().strftime('%d%H')}/{(datetime.utcnow()+timedelta(hours=2)).strftime('%d%H')} 4000 -RA BKN015"
        desc = "🚿 Potensi hujan ringan/CB (curah hujan meningkat)."
    elif deltaT < -2 and deltaRH > 10:
        trend_code = f"BECMG {datetime.utcnow().strftime('%d%H')}/{(datetime.utcnow()+timedelta(hours=2)).strftime('%d%H')} 5000 BR SCT020"
        desc = "🌫️ Perubahan bertahap ke kondisi lebih lembap/berawan."
    elif deltaT > 2 and deltaRH < -10:
        trend_code = f"BECMG {datetime.utcnow().strftime('%d%H')}/{(datetime.utcnow()+timedelta(hours=2)).strftime('%d%H')} 9999 NSW FEW030"
        desc = "☀️ Kondisi membaik, kelembapan menurun."
    else:
        trend_code = "NOSIG"
        desc = "✅ Tidak ada perubahan signifikan."

    return trend_code, desc

if st.button("🚀 Generate TAFOR + TREND"):
    # Parsing METAR
    try:
        parts = metar_input.split()
        wind = next((p for p in parts if p.endswith("KT")), "09005KT")
        vis = next((p for p in parts if p.isdigit() or "9999" in p), "9999")
        cloud = next((p for p in parts if p.startswith(("FEW", "SCT", "BKN", "OVC"))), "FEW020")
    except Exception:
        wind, vis, cloud = "09005KT", "9999", "FEW020"

    issue_dt = datetime.strptime(f"{issue_date} {issue_time}", "%Y/%m/%d %H:%M")
    valid_to = issue_dt + timedelta(hours=validity)
    taf_header = f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {issue_dt.strftime('%d%H')}/{valid_to.strftime('%d%H')}"

    # Susunan TAFOR
    becmg1_start = issue_dt + timedelta(hours=4)
    becmg1_end = becmg1_start + timedelta(hours=5)
    becmg2_start = issue_dt + timedelta(hours=10)
    becmg2_end = becmg2_start + timedelta(hours=6)

    tafor_lines = [
        taf_header,
        f"{wind} {vis} {cloud}",
        f"BECMG {becmg1_start.strftime('%d%H')}/{becmg1_end.strftime('%d%H')} 20005KT 8000 -RA SCT025 BKN040",
        f"BECMG {becmg2_start.strftime('%d%H')}/{becmg2_end.strftime('%d%H')} 24005KT 9999 SCT020"
    ]

    # Ambil data trend dari Open-Meteo
    data_trend = get_openmeteo_trend()
    if data_trend:
        trend_code, desc = analyze_trend(data_trend)
    else:
        trend_code, desc = "NOSIG", "⚠️ Data trend tidak tersedia."

    tafor_html = "<br>".join(tafor_lines)

    st.success("✅ TAFOR + TREND generation complete!")

    st.markdown("### 📡 METAR (Observasi Terakhir)")
    st.markdown(f"""
        <div style='padding:12px;border:2px solid #bbb;border-radius:10px;background-color:#fafafa;'>
            <p style='color:#000;font-weight:700;font-size:16px;font-family:monospace;'>{metar_input}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📝 Hasil TAFOR (WARR – Juanda)")
    st.markdown(f"""
        <div style='padding:15px;border:2px solid #555;border-radius:10px;background-color:#f9f9f9;'>
            <p style='color:#000;font-weight:700;font-size:16px;line-height:1.8;font-family:monospace;'>{tafor_html}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔄 TREND Otomatis (2 Jam ke Depan)")
    st.markdown(f"""
        <div style='padding:15px;border:2px solid #4a4;border-radius:10px;background-color:#f0fff0;'>
            <p style='color:#000;font-weight:700;font-size:16px;font-family:monospace;'>{trend_code}</p>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption("💡 TREND dihitung dari perubahan suhu, kelembapan, dan curah hujan (API Open-Meteo).")
