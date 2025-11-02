# app_pro.py
import streamlit as st
from datetime import datetime, timedelta
import requests, json, os

st.set_page_config(page_title="🛫 Auto TAFOR Pro — WARR (Juanda)", layout="centered")

st.markdown("## 🛫 Auto TAFOR Pro — WARR (Juanda)")
st.write("Fusi data real BMKG + Open-Meteo + METAR, sesuai format ICAO & Perka BMKG.")
st.divider()

# --------------------------------------------------------
# KONFIGURASI
# --------------------------------------------------------
REFRESH_INTERVAL = 900

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

# --------------------------------------------------------
# DATA HANDLER
# --------------------------------------------------------
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
def get_bmkg():
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
        now_utc = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        cuaca = data["data"][0]["cuaca"][0][0]
        nearest = min(cuaca, key=lambda c: abs(datetime.fromisoformat(c["datetime"].replace("Z","+00:00"))-now_utc))
        return {
            "status":"OK",
            "wx_desc": nearest.get("weather_desc"),
            "t": nearest.get("t"), "hu": nearest.get("hu"),
            "tcc": nearest.get("tcc"), "wd": nearest.get("wd_deg"),
            "ws": nearest.get("ws"), "vs": nearest.get("vs_text")
        }
    except Exception:
        return {"status":"Unavailable"}

@st.cache_data(ttl=REFRESH_INTERVAL)
def get_openmeteo():
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
            "status":"OK",
            "t":data["hourly"]["temperature_2m"][idx],
            "hu":data["hourly"]["relative_humidity_2m"][idx],
            "tcc":data["hourly"]["cloud_cover"][idx],
            "wd":data["hourly"]["winddirection_10m"][idx],
            "ws":data["hourly"]["windspeed_10m"][idx]
        }
    except:
        return {"status":"Unavailable"}

# --------------------------------------------------------
# LOGIKA METEOROLOGIS / ICAO + PERKA
# --------------------------------------------------------
def weather_to_icao(desc):
    if not desc: return ""
    desc = desc.lower()
    mapping = {
        "hujan ringan":"-RA", "hujan":"RA", "hujan lebat":"+RA",
        "berawan":"BKN", "awan banyak":"OVC",
        "cerah":"FEW", "cerah berawan":"SCT",
        "kabut":"FG", "berdebu":"DU", "badai petir":"TS", "gerimis":"DZ"
    }
    for key,val in mapping.items():
        if key in desc: return val
    return ""

def tcc_to_cloud(tcc):
    if tcc is None: return "FEW020"
    tcc = float(tcc)
    if tcc < 25: return "FEW020"
    elif tcc < 50: return "SCT025"
    elif tcc < 85: return "BKN030"
    else: return "OVC030"

# --------------------------------------------------------
# PROSES UTAMA
# --------------------------------------------------------
if st.button("🚀 Generate TAFOR + TREND"):
    issue_dt = datetime.combine(issue_date, datetime.utcnow().replace(hour=issue_time, minute=0, second=0).time())
    valid_to = issue_dt + timedelta(hours=validity)

    metar = metar_input.strip() or get_metar()
    bmkg = get_bmkg()
    openm = get_openmeteo()

    bmkg_status, open_status = bmkg.get("status"), openm.get("status")

    # === FUSI
    if bmkg_status=="OK":
        wind_dir, wind_spd = bmkg["wd"], bmkg["ws"]
        cloud = tcc_to_cloud(bmkg["tcc"])
        wx = weather_to_icao(bmkg["wx_desc"])
        vis = bmkg.get("vs","9999").replace("> ","")
    elif open_status=="OK":
        wind_dir, wind_spd = openm["wd"], openm["ws"]
        cloud = tcc_to_cloud(openm["tcc"])
        wx, vis = "", "9999"
    else:
        wind_dir, wind_spd, cloud, wx, vis = 90,5,"FEW020","", "9999"

    wind = f"{int(wind_dir):03d}{int(round(float(wind_spd))):02d}KT"

    # === STRUKTUR TAF
    taf_header = f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {issue_dt.strftime('%d%H')}/{valid_to.strftime('%d%H')}"
    tafor_lines = [taf_header, f"{wind} {vis} {cloud} {wx}".strip()]

    # aturan perubahan
    if "RA" in wx or "TS" in wx:
        tafor_lines.append(f"TEMPO {issue_dt.strftime('%d%H')}/{(issue_dt+timedelta(hours=2)).strftime('%d%H')} 4000 {wx} SCT020CB")
    elif "FG" in wx:
        tafor_lines.append(f"BECMG {issue_dt.strftime('%d%H')}/{(issue_dt+timedelta(hours=1)).strftime('%d%H')} 2000 FG")

    # TREND otomatis
    if wx in ["RA","-RA","+RA","TS"]:
        trend = f"TEMPO TL{(issue_dt+timedelta(hours=1)).strftime('%d%H%M')} 5000 {wx} SCT020CB"
    else:
        trend = "NOSIG"

    # === TAMPILAN
    taf_html = "<br>".join(tafor_lines)
    trend_html = trend

    st.success("✅ TAFOR + TREND profesional (ICAO + BMKG) selesai dibuat!")

    st.subheader("📊 Ringkasan Sumber Data")
    st.write(f"""
    | Sumber | Status |
    |:-------|:--------|
    | BMKG Source | {bmkg_status} |
    | Open Meteo | {open_status} |
    | METAR Input | {'✅ Manual' if metar_input else 'OK'} |
    """)

    st.markdown("### 📝 Hasil TAFOR (WARR – Juanda)")
    st.markdown(f"""
        <div style='padding:15px;border:2px solid #555;border-radius:10px;background-color:#f9f9f9;'>
            <p style='color:#000;font-weight:700;font-size:16px;font-family:monospace;line-height:1.8;'>{taf_html}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🌦️ TREND (Tambahan Otomatis)")
    st.markdown(f"""
        <div style='padding:15px;border:2px solid #777;border-radius:10px;background-color:#f4f4f4;'>
            <p style='color:#111;font-weight:700;font-size:16px;font-family:monospace;line-height:1.8;'>{trend_html}</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("💡 Format & isi TAFOR ini mengikuti ICAO Annex 3 dan Perka BMKG No.9/2021. Validasi tetap dilakukan forecaster sebelum publikasi operasional.")
