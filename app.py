import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="TAFOR Juanda (WARR)", layout="wide")

st.title("✈️ TAFOR WARR Otomatis – Berdasarkan Desa Gisik Cemandi (Sedati, Sidoarjo)")
st.caption("Data dari API BMKG (adm1=35, Jawa Timur) – lokasi terdekat dari Bandara Juanda")

# -------------------------------
# PARAMETER DASAR
# -------------------------------
tanggal = st.date_input("Tanggal Prakiraan", datetime.now())
jam_mulai = st.time_input("Jam Mulai (UTC)", datetime.utcnow().replace(minute=0, second=0, microsecond=0))
durasi = st.selectbox("Durasi TAF (jam)", [6, 9, 12, 18, 24], index=2)

bmkg_url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm?adm1=35"

# -------------------------------
# AMBIL DATA BMKG
# -------------------------------
st.subheader("📡 Ambil Data BMKG (Jawa Timur)")

if st.button("🔄 Ambil Data BMKG Terdekat Juanda"):
    try:
        r = requests.get(bmkg_url, timeout=20)
        data = r.json()
        df_list = []

        for item in data.get("data", []):
            nama = str(item.get("lokasi", "")).upper()
            adm3 = str(item.get("adm3", "")).upper()
            if "SEDATI" in adm3 or "GISIK" in nama or "CEMANDI" in nama:
                df_list.append({
                    "lokasi": item.get("lokasi"),
                    "cuaca": item.get("cuaca"),
                    "suhu": item.get("t"),
                    "kelembapan": item.get("hu"),
                    "arah_angin": item.get("wd"),
                    "kecepatan_angin": item.get("ws"),
                    "jam": item.get("time")
                })

        if not df_list:
            st.warning("Tidak ditemukan data untuk Gisik Cemandi / Sedati.")
        else:
            df = pd.DataFrame(df_list)
            st.success(f"Ditemukan {len(df)} data prakiraan dari wilayah Sedati.")
            st.dataframe(df)
            st.session_state["df_bmkg"] = df

    except Exception as e:
        st.error(f"Gagal mengambil data BMKG: {e}")

# -------------------------------
# GENERATE TAFOR OTOMATIS
# -------------------------------
st.subheader("✈️ Hasil TAFOR Otomatis (Simulasi WARR)")

def generate_tafor_from_bmkg(df, tgl, jam, durasi):
    start = datetime.combine(tgl, jam)
    end = start + timedelta(hours=durasi)
    valid_period = f"{start.strftime('%d%H')}/{end.strftime('%d%H')}"

    taf_lines = [f"TAF WARR {start.strftime('%d%H%M')}Z {valid_period}"]

    # Ambil data awal dan perubahan
    df = df.head(4)
    prev_cuaca = None
    for i, row in df.iterrows():
        cuaca = str(row['cuaca']).lower()
        ws = row['kecepatan_angin']
        wd = row['arah_angin']
        suhu = row['suhu']

        # Baris awal
        if i == 0:
            taf_lines.append(f"{wd}0{int(ws or 5):02d}KT 8000 FEW018 SCT025 {cuaca.upper()}")
        else:
            if prev_cuaca and cuaca != prev_cuaca:
                taf_lines.append(f"TEMPO {wd}0{int(ws or 5):02d}KT 6000 {cuaca.upper()}")
            elif ws and abs(ws - df.iloc[i-1]['kecepatan_angin']) > 5:
                taf_lines.append(f"BECMG {wd}0{int(ws):02d}KT")
        prev_cuaca = cuaca

    tafor_text = "\n".join(taf_lines)
    return tafor_text

if st.button("🧭 Generate TAFOR Juanda"):
    if "df_bmkg" in st.session_state:
        tafor_text = generate_tafor_from_bmkg(st.session_state["df_bmkg"], tanggal, jam_mulai, durasi)
        st.code(tafor_text, language="yaml")
        st.download_button(
            label="💾 Unduh TAFOR (TXT)",
            data=tafor_text,
            file_name=f"TAFOR_WARR_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )
    else:
        st.warning("Ambil data BMKG terlebih dahulu.")

st.markdown("---")
st.caption("Sumber: API BMKG open data (adm1=35) – Filtering Gisik Cemandi/Sedati (Sidoarjo)")
