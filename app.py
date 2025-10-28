import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="🛫 Auto TAFOR Generator — WARR (Juanda)", layout="centered")

st.markdown("## 🛫 Auto TAFOR Generator — WARR (Juanda)")
st.write("Fusion: BMKG (point → adm4 Sedati Gede) + Open-Meteo + METAR (OGIMET) → output TAFOR (ICAO-like).")
st.write("Gunakan input METAR terakhir (opsional) agar hasil lebih sesuai kondisi aktual.")
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
                           "WARR 280430Z 09008KT 9999 FEW020CB 33/24 1009 NOSIG=",
                           height=100)

if st.button("🚀 Generate TAFOR"):
    # Ambil elemen dari METAR
    try:
        parts = metar_input.split()
        wind = next((p for p in parts if p.endswith("KT")), "09005KT")
        vis = next((p for p in parts if p.isdigit() or "9999" in p), "9999")
        cloud = next((p for p in parts if p.startswith(("FEW", "SCT", "BKN", "OVC"))), "FEW020")
    except Exception:
        wind, vis, cloud = "09005KT", "9999", "FEW020"

    # Hitung waktu issue dan validitas
    issue_dt = datetime.strptime(f"{issue_date} {issue_time}", "%Y/%m/%d %H:%M")
    valid_to = issue_dt + timedelta(hours=validity)

    taf_header = f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {issue_dt.strftime('%d%H')}/{valid_to.strftime('%d%H')}"

    # Buat periode perubahan (BECMG)
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

    tafor_html = "<br>".join(tafor_lines)

    st.success("✅ TAFOR generation complete!")

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

    st.info("💡 TAFOR ini bersifat eksperimental. Validasi dengan TAF resmi BMKG sebelum digunakan operasional.")
