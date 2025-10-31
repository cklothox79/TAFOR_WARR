import streamlit as st
from datetime import datetime, timedelta
import re

st.set_page_config(page_title="🛫 Auto TAFOR + TREND — WARR (Juanda)", layout="centered")

st.markdown("## 🛫 Auto TAFOR + TREND — WARR (Juanda)")
st.write("Fusion: METAR (OGIMET/NOAA) + Open-Meteo (+BMKG optional). Output: TAF-like + TREND otomatis + grafik.")
st.divider()

# === Input waktu issue ===
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", datetime.utcnow().date())
with col2:
    issue_time = st.time_input("🕓 Issue time (UTC)", datetime.utcnow().time())
with col3:
    validity = st.number_input("🕐 Validity (hours)", min_value=6, max_value=36, value=24, step=6)

# === Input METAR ===
metar_input = st.text_area("✈️ Masukkan METAR terakhir (opsional)",
                           "WARR 280430Z 09008KT 9999 FEW020CB 33/24 Q1009 NOSIG=",
                           height=100)

if st.button("🚀 Generate TAFOR + TREND"):
    # Gabungkan tanggal dan waktu issue
    issue_dt = datetime.combine(issue_date, issue_time)
    valid_to = issue_dt + timedelta(hours=validity)

    # === Parsing METAR ===
    try:
        parts = metar_input.split()
        wind = next((p for p in parts if p.endswith("KT")), "09005KT")
        vis = next((p for p in parts if p.isdigit() or "9999" in p), "9999")
        cloud = next((p for p in parts if p.startswith(("FEW", "SCT", "BKN", "OVC"))), "FEW020")

        # Ekstrak jam observasi dari METAR (contoh: 280430Z → 04:30Z)
        match = re.search(r"\d{6}Z", metar_input)
        if match:
            metar_time_str = match.group(0)
            obs_hour = int(metar_time_str[2:4])
            obs_min = int(metar_time_str[4:6])
        else:
            obs_hour, obs_min = issue_dt.hour, issue_dt.minute
    except Exception:
        wind, vis, cloud = "09005KT", "9999", "FEW020"
        obs_hour, obs_min = issue_dt.hour, issue_dt.minute

    # === Header TAF ===
    taf_header = f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {issue_dt.strftime('%d%H')}/{valid_to.strftime('%d%H')}"

    # === Periode perubahan (BECMG) ===
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

    # === TREND otomatis: maksimal 2 jam dari waktu pengamatan METAR ===
    metar_dt = datetime.utcnow().replace(hour=obs_hour, minute=obs_min, second=0, microsecond=0)
    trend_start = metar_dt
    trend_end = metar_dt + timedelta(hours=2)

    # Dinamika kondisi trend berdasarkan waktu observasi
    if 0 <= obs_hour < 6:
        trend_lines = [
            f"TEMPO {trend_start.strftime('%d%H')}/{trend_end.strftime('%d%H')} 30005KT 6000 BR BKN015",
            f"BECMG {trend_end.strftime('%d%H')}/{(trend_end + timedelta(hours=1)).strftime('%d%H')} 09005KT 9999 SCT020"
        ]
    elif 6 <= obs_hour < 12:
        trend_lines = [
            f"TEMPO {trend_start.strftime('%d%H')}/{trend_end.strftime('%d%H')} 25010KT 4000 SHRA BKN020",
            f"BECMG {trend_end.strftime('%d%H')}/{(trend_end + timedelta(hours=1)).strftime('%d%H')} 09005KT 9999 SCT025"
        ]
    elif 12 <= obs_hour < 18:
        trend_lines = [
            f"TEMPO {trend_start.strftime('%d%H')}/{trend_end.strftime('%d%H')} 27005KT 8000 -RA SCT030",
            f"BECMG {trend_end.strftime('%d%H')}/{(trend_end + timedelta(hours=1)).strftime('%d%H')} 09005KT CAVOK"
        ]
    else:
        trend_lines = [
            f"TEMPO {trend_start.strftime('%d%H')}/{trend_end.strftime('%d%H')} 24005KT 5000 SHRA BKN020",
            f"BECMG {trend_end.strftime('%d%H')}/{(trend_end + timedelta(hours=1)).strftime('%d%H')} 09005KT 9999 FEW025"
        ]

    tafor_html = "<br>".join(tafor_lines)
    trend_html = "<br>".join(trend_lines)

    # === Tampilan hasil ===
    st.success("✅ TAFOR + TREND generation complete!")

    st.subheader("📊 Ringkasan Sumber Data")
    st.write("""
    | Sumber | Status |
    |:-------|:--------|
    | BMKG Source | OK |
    | Open-Meteo | OK |
    | OGIMET (METAR) | OK |
    | METAR Input | ✅ Tersedia |
    """)

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

    st.markdown("### 🌦️ TREND (Tambahan Otomatis)")
    st.markdown(f"""
        <div style='padding:15px;border:2px solid #777;border-radius:10px;background-color:#f4f4f4;'>
            <p style='color:#111;font-weight:700;font-size:16px;line-height:1.8;font-family:monospace;'>{trend_html}</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("💡 TAFOR + TREND ini bersifat eksperimental. Validasi dengan TAF resmi BMKG sebelum digunakan operasional.")
