# app.py — Auto TAFOR Pro (ICAO + BMKG + OpenMeteo)
import streamlit as st
from datetime import datetime, timedelta
import requests, json, os, pandas as pd, matplotlib.pyplot as plt

st.set_page_config(page_title="🛫 Auto TAFOR Pro — WARR (Juanda)", layout="centered")

st.markdown("## 🛫 Auto TAFOR Pro — WARR (Juanda)")
st.write("Fusi data real BMKG + Open-Meteo + METAR, sesuai format ICAO & Perka BMKG.")
st.divider()

REFRESH_INTERVAL = 900  # 15 menit cache

# === INPUTS ===
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", datetime.utcnow().date())
with col2:
    jam_penting = [0, 3, 6, 9, 12, 15, 18, 21]
    jam_sekarang = datetime.utcnow().hour
    default_jam = min(jam_penting, key=lambda j: abs(j - jam_sekarang))
    issue_time = st.selectbox("🕓 Issue time (UTC)", jam_penting, index=jam_penting.index(default_jam))
with col3:
    validity = st.number_input("🕐 Validity (hours)", min_value=6, max_value=36, value=24, step=6)

metar_input = st.text_area("✈️ Masukkan METAR terakhir (opsional)", "", height=100)

# === FETCH DATA ===
@st.cache_data(ttl=REFRESH_INTERVAL)
def get_metar():
    try:
        url = "https://tgftp.nws.noaa.gov/data/observations/metar/stations/WARR.TXT"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.text.strip().split("\n")[-1]
    except:
        return None

@st.cache_data(ttl=REFRESH_INTERVAL)
def get_bmkg_data():
    url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm"
    params = {"adm1":"35","adm2":"35.15","adm3":"35.15.17","adm4":"35.15.17.2001"}
    try:
        r = requests.get(url, params=params, timeout=15, verify=False)
        data = r.json()
    except:
        if os.path.exists("JSON_BMKG.txt"):
            with open("JSON_BMKG.txt","r",encoding="utf-8") as f:
                data = json.load(f)
        else:
            return {"status":"Unavailable"}
    try:
        cuaca = data["data"][0]["cuaca"][0][0]
        now_utc = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        nearest = min(cuaca, key=lambda c: abs(datetime.fromisoformat(c["datetime"].replace("Z","+00:00"))-now_utc))
        return {
            "status":"OK","data":cuaca,
            "wx_desc": nearest.get("weather_desc"), "t": nearest.get("t"),
            "hu": nearest.get("hu"), "tcc": nearest.get("tcc"),
            "wd": nearest.get("wd_deg"), "ws": nearest.get("ws"),
            "vs": nearest.get("vs_text")
        }
    except Exception:
        return {"status":"Unavailable"}

@st.cache_data(ttl=REFRESH_INTERVAL)
def get_openmeteo_data():
    try:
        url = ("https://api.open-meteo.com/v1/forecast?latitude=-7.379&longitude=112.787"
               "&hourly=temperature_2m,relative_humidity_2m,cloud_cover,"
               "windspeed_10m,winddirection_10m&forecast_days=1&timezone=UTC")
        r = requests.get(url, timeout=10)
        data = r.json()
        hour = datetime.utcnow().hour
        idx = min(range(len(data["hourly"]["time"])),
                  key=lambda i: abs(datetime.fromisoformat(data["hourly"]["time"][i]).hour - hour))
        return {
            "status":"OK","data":data,
            "t":data["hourly"]["temperature_2m"][idx],
            "hu":data["hourly"]["relative_humidity_2m"][idx],
            "tcc":data["hourly"]["cloud_cover"][idx],
            "wd":data["hourly"]["winddirection_10m"][idx],
            "ws":data["hourly"]["windspeed_10m"][idx]
        }
    except:
        return {"status":"Unavailable"}

# === Helper Functions ===
def weather_to_icao(desc):
    if not desc: return ""
    desc = desc.lower()
    mapping = {
        "hujan ringan":"-RA","hujan":"RA","hujan lebat":"+RA",
        "berawan":"BKN","awan banyak":"OVC","cerah":"FEW","cerah berawan":"SCT",
        "kabut":"FG","badai petir":"TS","gerimis":"DZ"
    }
    for k,v in mapping.items():
        if k in desc: return v
    return ""

def tcc_to_cloud(tcc):
    if tcc is None: return "FEW020"
    tcc = float(tcc)
    if tcc < 25: return "FEW020"
    elif tcc < 50: return "SCT025"
    elif tcc < 85: return "BKN030"
    else: return "OVC030"

# === MAIN ===
if st.button("🚀 Generate TAFOR + TREND"):
    issue_dt = datetime.combine(issue_date, datetime.utcnow().replace(hour=issue_time, minute=0, second=0).time())
    valid_to = issue_dt + timedelta(hours=validity)

    metar = metar_input.strip() or get_metar()
    bmkg = get_bmkg_data()
    openm = get_openmeteo_data()

    bmkg_status, open_status = bmkg.get("status"), openm.get("status")

    # Pilih sumber utama
    if bmkg_status == "OK":
        src = bmkg
    elif open_status == "OK":
        src = openm
    else:
        src = {"wd":90,"ws":5,"tcc":20,"hu":60,"wx_desc":"","vs":"9999"}
        bmkg_status = open_status = "Unavailable"

    wind = f"{int(src['wd']):03d}{int(round(float(src['ws']))):02d}KT"
    vis = src.get("vs","9999").replace("> ","")
    cloud = tcc_to_cloud(src.get("tcc"))
    wx = weather_to_icao(src.get("wx_desc",""))

    # === Generate TAF ===
    taf_header = f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {issue_dt.strftime('%d%H')}/{valid_to.strftime('%d%H')}"
    tafor_lines = [taf_header, f"{wind} {vis} {cloud} {wx}".strip()]
    if "RA" in wx or "TS" in wx:
        tafor_lines.append(f"TEMPO {issue_dt.strftime('%d%H')}/{(issue_dt+timedelta(hours=2)).strftime('%d%H')} 4000 {wx} SCT020CB")
    elif "FG" in wx:
        tafor_lines.append(f"BECMG {issue_dt.strftime('%d%H')}/{(issue_dt+timedelta(hours=1)).strftime('%d%H')} 2000 FG")
    taf_html = "<br>".join(tafor_lines)

    trend = "NOSIG" if wx == "" else f"TEMPO TL{(issue_dt+timedelta(hours=1)).strftime('%d%H%M')} 5000 {wx} SCT020CB"

    st.success("✅ TAFOR + TREND profesional selesai dibuat!")

    st.subheader("📊 Ringkasan Sumber Data")
    st.write(f"""
    | Sumber | Status |
    |:-------|:--------|
    | BMKG Source | {bmkg_status} |
    | Open Meteo | {open_status} |
    | METAR Input | {'✅ Manual' if metar_input else 'Auto/OGIMET'} |
    """)

    st.markdown("### 📝 Hasil TAFOR (WARR – Juanda)")
    st.markdown(f"<div style='padding:15px;border:2px solid #555;border-radius:10px;background-color:#f9f9f9;'><p style='font-family:monospace;font-size:16px;font-weight:700;'>{taf_html}</p></div>", unsafe_allow_html=True)
    st.markdown("### 🌦️ TREND (Tambahan Otomatis)")
    st.markdown(f"<div style='padding:15px;border:2px solid #777;border-radius:10px;background-color:#f4f4f4;'><p style='font-family:monospace;font-size:16px;font-weight:700;'>{trend}</p></div>", unsafe_allow_html=True)

    # === ANALISIS MODEL ===
    st.markdown("### 🧠 Analisis Model (Interpretasi Otomatis)")
    tcc, rh = src.get("tcc"), src.get("hu")
    if tcc is not None and rh is not None:
        sky = "Cerah" if tcc < 25 else "Berawan" if tcc < 70 else "Tertutup"
        hum = "Kering" if rh < 60 else "Lembap" if rh < 80 else "Basah"
        signif = "Tidak ada cuaca signifikan terdeteksi." if wx == "" else f"Fenomena signifikan terdeteksi: {wx}"
        st.markdown(f"""
        <div style='padding:12px;border:2px solid #888;border-radius:10px;background-color:#f6f6f6;'>
            <b>RH:</b> {rh:.0f}% ({hum})<br>
            <b>Tutupan awan:</b> {tcc:.0f}% ({sky})<br>
            <b>Angin:</b> {wind} ({'lemah' if float(src['ws'])<10 else 'sedang' if float(src['ws'])<20 else 'kuat'})<br>
            <b>Interpretasi:</b> {signif}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Analisis model tidak tersedia (data BMKG dan Open-Meteo tidak lengkap).")

    # === GRAFIK FUSI MODEL ===
    st.markdown("### 📈 Grafik Prakiraan 24 Jam (BMKG vs Open-Meteo)")
    try:
        df_bmkg, df_open = [], []
        if bmkg_status == "OK":
            for c in bmkg["data"]:
                df_bmkg.append({
                    "time": datetime.fromisoformat(c["datetime"].replace("Z","+00:00")),
                    "RH": c.get("hu"), "Cloud": c.get("tcc"),
                    "WindSpeed": c.get("ws")
                })
        if open_status == "OK":
            om = openm["data"]["hourly"]
            for t, rh, cc, ws in zip(
                om["time"], om["relative_humidity_2m"],
                om["cloud_cover"], om["windspeed_10m"]
            ):
                df_open.append({
                    "time": datetime.fromisoformat(t),
                    "RH": rh, "Cloud": cc, "WindSpeed": ws
                })
        if df_bmkg or df_open:
            fig, ax = plt.subplots()
            if df_bmkg:
                df1 = pd.DataFrame(df_bmkg)
                ax.plot(df1["time"], df1["RH"], 'b-', label="BMKG RH")
                ax.plot(df1["time"], df1["Cloud"], 'g-', label="BMKG Cloud")
                ax.plot(df1["time"], df1["WindSpeed"], 'r-', label="BMKG Wind")
            if df_open:
                df2 = pd.DataFrame(df_open)
                ax.plot(df2["time"], df2["RH"], 'b--', label="Open RH")
                ax.plot(df2["time"], df2["Cloud"], 'g--', label="Open Cloud")
                ax.plot(df2["time"], df2["WindSpeed"], 'r--', label="Open Wind")
            ax.set_ylabel("Nilai (%) / Kecepatan (kt)")
            ax.set_xlabel("Waktu (UTC)")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)
        else:
            st.info("Grafik tidak tersedia.")
    except Exception as e:
        st.warning(f"Gagal menampilkan grafik: {e}")

    st.info("💡 TAFOR mengikuti ICAO Annex 3 & Perka BMKG No.9/2021. Validasi tetap oleh forecaster sebelum publikasi.")
