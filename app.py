import streamlit as st
import pandas as pd
from core.data_fetch import fetch_all
from core.fusion_core import weighted_fusion
from core.tafor_gen import tafor_generate

st.set_page_config(page_title="TAFOR Fusion Pro — WARR", layout="centered")
st.title("🛫 TAFOR Fusion Pro — Operational v2.5 (WARR / Sedati Gede)")

lat, lon = -7.378, 112.787
bmkg_url = "https://raw.githubusercontent.com/cklothoz79/TAFOR_WARR/main/JSON_BMKG.txt"

if st.button("🔄 Ambil Data & Proses"):
    with st.spinner("Mengambil data dari BMKG + ECMWF + ICON + GFS..."):
        df_all = fetch_all(lat, lon, bmkg_url)
        df_fused = weighted_fusion(df_all)
        tafor_text = tafor_generate(df_fused)

        st.success("✅ Data berhasil digabung!")
        st.text_area("TAFOR Otomatis", tafor_text, height=200)
        st.dataframe(df_fused.head())
