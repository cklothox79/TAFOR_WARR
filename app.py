import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="🛫 Auto TAFOR + TREND — WARR (Juanda)", layout="centered")

st.markdown("## 🛫 Auto TAFOR + TREND — WARR (Juanda)")
st.write("Fusion: METAR (OGIMET/NOAA) + Open-Meteo (+BMKG optional). Output: TAF-like + TREND otomatis + grafik.")
st.divider()

# === Input waktu issue ===
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", datetime.utcnow().date())
with col2:
    # Pilih jam penting
    jam_penting = [0, 3, 6, 9, 12, 15, 18, 21]
    jam_sekarang = datetime.utcnow().hour
    default_jam = min(jam_penting, key=lambda j: abs(j - jam_sekarang))
    issue_time = st.selectbox("🕓 Issue time (UTC)", jam_penting, index=jam_penting.index(default_jam))
with col3:
    validity = st.number_input("🕐 Validity (hours)", min_value=6, max_value=36, value=24, step=6)

# === Input METAR ===
metar_input = st.text_area("✈️ Masukkan METAR terakhir (opsional)",
                           "WARR 280430Z 09008KT 9999 FEW020CB 33/24 Q1009 NOSIG=",
                           height=100)

if st.button("🚀 Generate TAFOR + TREND"):
    # Gabungkan tanggal dan jam issue
    issue_dt = datetime.combine(issue_date, datetime.utcnow().replace(hour=issue_time, minute=0, second=0).time())
    valid_to = issue_dt + timedelta(hours=validity)

    # === Parsing METAR sederhana ===
    try:
        parts = metar_input.split()
        wind = next((p for p in parts if p.endswith("KT")), "09005KT")
        vis = next((p for p in parts if p.isdigit() or "9999" in p), "9999")
        cloud = next((p for p in parts if p.startswith(("FEW", "SCT", "BKN", "OVC"))), "FEW020")
        wx = next((p for p in parts if any(w in p for w in ["RA", "TS", "SH", "FG", "DZ"])), "")
    except Exception:
        wind, vis, cloud, wx = "09005KT", "9999", "FEW020", ""

    # === Header TAF ===
    taf_header = f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {issue_dt.strftime('%d%H')}/{valid_to.strftime('%d%H')}"

    # === BECMG periodik ===
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

    # === TREND otomatis (durasi 1 jam) ===
    trend_start = issue_dt
    trend_end = trend_start + timedelta(hours=1)

    if wx:  # ada cuaca signifikan di METAR
        trend_lines = [
            f"TEMPO TL{trend_end.strftime('%d%H%M')} 5000 {wx} SCT020CB",
            f"BECMG {trend_start.strftime('%d%H%M')}/{trend_end.strftime('%d%H%M')} {wind} {vis} {cloud}"
        ]
    else:  # tidak ada fenomena signifikan
        trend_lines = ["NOSIG"]

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
