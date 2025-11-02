# app.py
import streamlit as st
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="🛫 Auto TAFOR + TREND — WARR (Juanda)", layout="centered")

st.markdown("## 🛫 Auto TAFOR + TREND — WARR (Juanda)")
st.write("Fusion: METAR (OGIMET/NOAA) + BMKG (Sedati Gede) + Open-Meteo. Output: TAF-like + TREND otomatis + grafik.")
st.divider()

# === Input waktu issue ===
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

# === Input METAR ===
metar_input = st.text_area("✈️ Masukkan METAR terakhir (opsional)", "", height=100)

# === FUNGSI PENGAMBIL DATA ===

def get_metar_ogimet():
    """Ambil METAR realtime dari OGIMET (NOAA fallback)."""
    try:
        url = "https://tgftp.nws.noaa.gov/data/observations/metar/stations/WARR.TXT"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")
            if len(lines) >= 2:
                return lines[-1]
    except Exception:
        pass
    return None


def get_bmkg_forecast():
    """Ambil data cuaca BMKG (Sedati Gede ADM4)."""
    url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm"
    params = {
        "adm1": "35",
        "adm2": "35.15",
        "adm3": "35.15.17",
        "adm4": "35.15.17.2001"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            forecasts = data.get("data", {}).get("forecasts", [])
            if forecasts:
                f = forecasts[0]
                return {
                    "status": "OK",
                    "time": f.get("datetime"),
                    "temp": f.get("t", {}).get("value"),
                    "rh": f.get("hu", {}).get("value"),
                    "wind_dir": f.get("wd", {}).get("value"),
                    "wind_spd": f.get("ws", {}).get("value"),
                    "clouds": f.get("tcc", {}).get("value"),
                    "wx": f.get("weather", {}).get("desc")
                }
    except Exception:
        pass
    return {"status": "Unavailable"}


def get_openmeteo_forecast():
    """Ambil data Open-Meteo untuk koordinat Juanda."""
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=-7.379&longitude=112.787"
            "&hourly=temperature_2m,relative_humidity_2m,cloud_cover,"
            "windspeed_10m,winddirection_10m&forecast_days=1&timezone=UTC"
        )
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            now_hour = datetime.utcnow().hour
            idx = min(range(len(data["hourly"]["time"])),
                      key=lambda i: abs(datetime.fromisoformat(data["hourly"]["time"][i]).hour - now_hour))
            return {
                "status": "OK",
                "temp": data["hourly"]["temperature_2m"][idx],
                "rh": data["hourly"]["relative_humidity_2m"][idx],
                "wind_dir": data["hourly"]["winddirection_10m"][idx],
                "wind_spd": data["hourly"]["windspeed_10m"][idx],
                "clouds": data["hourly"]["cloud_cover"][idx]
            }
    except Exception:
        pass
    return {"status": "Unavailable"}


# === MAIN PROCESS ===
if st.button("🚀 Generate TAFOR + TREND"):

    issue_dt = datetime.combine(issue_date, datetime.utcnow().replace(hour=issue_time, minute=0, second=0).time())
    valid_to = issue_dt + timedelta(hours=validity)

    # --- Data Sources ---
    metar = metar_input.strip() or get_metar_ogimet()
    bmkg_data = get_bmkg_forecast()
    openmeteo_data = get_openmeteo_forecast()

    bmkg_status = bmkg_data.get("status", "Unavailable")
    openmeteo_status = openmeteo_data.get("status", "Unavailable")
    metar_status = "✅ Manual" if metar_input.strip() else ("OK" if metar else "Unavailable")

    # --- Parsing METAR ---
    try:
        parts = metar.split()
        wind = next((p for p in parts if p.endswith("KT")), "09005KT")
        vis = next((p for p in parts if p.isdigit() or "9999" in p), "9999")
        cloud = next((p for p in parts if p.startswith(("FEW", "SCT", "BKN", "OVC"))), "FEW020")
        wx = next((p for p in parts if any(w in p for w in ["RA", "TS", "SH", "FG", "DZ"])), "")
    except Exception:
        wind, vis, cloud, wx = "09005KT", "9999", "FEW020", ""

    # --- Fusi BMKG + OpenMeteo (prioritas BMKG) ---
    if bmkg_status == "OK":
        wind_final = f"{int(bmkg_data['wind_dir']):03d}{int(bmkg_data['wind_spd']):02d}KT"
        cloud_final = "SCT020" if bmkg_data.get("clouds", 0) > 5 else "FEW020"
        wx_final = bmkg_data.get("wx", "") or wx
    elif openmeteo_status == "OK":
        wind_final = f"{int(openmeteo_data['wind_dir']):03d}{int(openmeteo_data['wind_spd']):02d}KT"
        cloud_final = "SCT020" if openmeteo_data.get("clouds", 0) > 5 else "FEW020"
        wx_final = wx
    else:
        wind_final, cloud_final, wx_final = wind, cloud, wx

    # === Header TAF ===
    taf_header = f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {issue_dt.strftime('%d%H')}/{valid_to.strftime('%d%H')}"

    becmg1_start = issue_dt + timedelta(hours=4)
    becmg1_end = becmg1_start + timedelta(hours=5)
    becmg2_start = issue_dt + timedelta(hours=10)
    becmg2_end = becmg2_start + timedelta(hours=6)

    tafor_lines = [
        taf_header,
        f"{wind_final} {vis} {cloud_final}",
        f"BECMG {becmg1_start.strftime('%d%H')}/{becmg1_end.strftime('%d%H')} 20005KT 8000 -RA SCT025 BKN040",
        f"BECMG {becmg2_start.strftime('%d%H')}/{becmg2_end.strftime('%d%H')} 24005KT 9999 SCT020"
    ]

    trend_start = issue_dt
    trend_end = trend_start + timedelta(hours=1)

    if wx_final:
        trend_lines = [
            f"TEMPO TL{trend_end.strftime('%d%H%M')} 5000 {wx_final} SCT020CB",
            f"BECMG {trend_start.strftime('%d%H%M')}/{trend_end.strftime('%d%H%M')} {wind_final} {vis} {cloud_final}"
        ]
    else:
        trend_lines = ["NOSIG"]

    tafor_html = "<br>".join(tafor_lines)
    trend_html = "<br>".join(trend_lines)

    # === Output ===
    st.success("✅ TAFOR + TREND generation complete!")

    st.subheader("📊 Ringkasan Sumber Data")
    st.write(f"""
    | Sumber | Status |
    |:-------|:--------|
    | BMKG Source | {bmkg_status} |
    | Open Meteo | {openmeteo_status} |
    | OGIMET (METAR) | OK |
    | METAR Input | {metar_status} |
    """)

    st.markdown("### 📡 METAR (Observasi Terakhir)")
    st.markdown(f"""
        <div style='padding:12px;border:2px solid #bbb;border-radius:10px;background-color:#fafafa;'>
            <p style='color:#000;font-weight:700;font-size:16px;font-family:monospace;'>{metar or 'Data tidak tersedia'}</p>
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

    with st.expander("🧠 Debug: fused numeric values (BMKG priority then Open-Meteo)"):
        st.json({"BMKG": bmkg_data, "OpenMeteo": openmeteo_data})
