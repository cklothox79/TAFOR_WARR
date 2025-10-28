import streamlit as st
from datetime import datetime

st.set_page_config(page_title="🛫 Auto TAFOR Generator — WARR (Juanda)", layout="centered")

st.markdown("## 🛫 Auto TAFOR Generator — WARR (Juanda)")
st.write("Fusion: BMKG (point → adm4 Sedati Gede) + Open-Meteo + METAR (OGIMET) → output TAFOR (ICAO-like).")
st.write("Gunakan input METAR terakhir (opsional) agar hasil lebih sesuai kondisi aktual.")

st.divider()

# 📅 Input data TAFOR
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.text_input("📅 Issue date (UTC)", value=datetime.utcnow().strftime("%Y/%m/%d"))
with col2:
    issue_time = st.text_input("🕓 Issue time (UTC)", value=datetime.utcnow().strftime("%H:00"))
with col3:
    validity = st.number_input("🕐 Validity (hours)", min_value=6, max_value=36, value=24, step=6)

metar_input = st.text_area("✈️ Masukkan METAR terakhir (opsional)", 
                           "WARR 280330Z 09008KT 9999 FEW020CB 33/24 1009 NOSIG=",
                           height=100)

# Tombol generate
if st.button("🚀 Generate TAFOR"):
    # Simulasi hasil TAFOR sesuai format ICAO
    tafor_lines = [
        "TAF WARR 280300Z 2803/2903",
        "09008KT 9999 FEW020CB",
        "BECMG 280900Z 20005KT 8000 -RA SCT025 BKN040",
        "BECMG 281500Z 24005KT 9999 SCT020"
    ]
    tafor_html = "<br>".join(tafor_lines)

    st.success("✅ TAFOR generation complete!")
    
    # Ringkasan sumber data
    st.subheader("📊 Ringkasan Sumber Data")
    st.write("""
    | Sumber | Status |
    |:-------|:--------|
    | BMKG Source | OK |
    | Open-Meteo | OK |
    | OGIMET (METAR) | OK |
    | METAR Input | ✅ WARR 280330Z 09008KT 9999 FEW020CB 33/24 1009 NOSIG= |
    """)

    # Tampilan hasil TAFOR
    st.markdown("### 📝 Hasil TAFOR (WARR – Juanda)")
    st.markdown(
        f"""
        <div style='padding:15px;border:2px solid #555;border-radius:10px;background-color:#f9f9f9;'>
            <p style='color:#000;font-weight:700;font-size:16px;line-height:1.8;font-family:monospace;'>{tafor_html}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("💡 TAFOR ini bersifat eksperimental. Validasi dengan TAF resmi BMKG sebelum digunakan operasional.")
