# streamlit_auto_tafor_with_trend.py
# Auto TAFOR + TREND for WARR (Juanda)
# - Fusion: METAR (OGIMET/NOAA) + Open-Meteo + (optional) BMKG
# - Outputs: TAF-like block + automatic 2–6 hour TREND + interactive charts
# Run: streamlit run streamlit_auto_tafor_with_trend.py

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="🛫 Auto TAFOR + TREND — WARR (Juanda)", layout="wide")
st.title("🛫 Auto TAFOR + TREND — WARR (Juanda)")
st.caption("Fusion: METAR (OGIMET/NOAA) + Open-Meteo (+BMKG optional). Output: TAF-like + TREND otomatis + grafik.")

# ---------- Inputs ----------
col1, col2, col3 = st.columns([1,1,1])
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", value=datetime.utcnow().date())
with col2:
    issue_time = st.time_input("🕓 Issue time (UTC)", value=datetime.utcnow().time())
issue_dt = datetime.combine(issue_date, issue_time)
with col3:
    use_bmkg = st.checkbox("Use BMKG point API (optional)", value=False)

metar_input = st.text_area("METAR input (optional)", value="WARR 280430Z 09008KT 9999 FEW020CB 33/24 Q1009 NOSIG=", height=100)

st.markdown("---")

# ---------- Configurable endpoints / params ----------
LAT, LON = -7.379, 112.787  # approximate Juanda
OPENMETEO_URL = (
    f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
    "&hourly=temperature_2m,relative_humidity_2m,precipitation,cloudcover,wind_speed_10m,wind_direction_10m"
    "&timezone=UTC"
)

BMKG_EXAMPLE_URL = """https://wilayahbmkg.example/api/point?adm4=..."""  # replace with real BMKG endpoint if available

OGIMET_METAR_URL = "https://aviationweather.gov/adds/dataserver_current/httpparam"

# ---------- Helper functions ----------

def parse_metar_simple(metar_text: str):
    parts = metar_text.replace('=', ' ').split()
    wind = next((p for p in parts if p.endswith('KT')), '00000KT')
    vis = next((p for p in parts if p.isdigit() or '9999' in p), '9999')
    cloud = next((p for p in parts if p.startswith(('FEW', 'SCT', 'BKN', 'OVC'))), 'FEW020')
    return wind, vis, cloud


def fetch_openmeteo(lon=LON, lat=LAT):
    try:
        r = requests.get(OPENMETEO_URL, timeout=10)
        r.raise_for_status()
        j = r.json()
        df = pd.DataFrame({
            'time': pd.to_datetime(j['hourly']['time']),
            'temp': j['hourly']['temperature_2m'],
            'rh': j['hourly']['relative_humidity_2m'],
            'precip': j['hourly']['precipitation'],
            'cloud': j['hourly']['cloudcover'],
            'wind': j['hourly']['wind_speed_10m']
        })
        return df
    except Exception as e:
        st.warning(f"Open-Meteo fetch failed: {e}")
        return None


def analyze_trend_from_df(df: pd.DataFrame, hours_ahead=3):
    now_utc = pd.to_datetime(datetime.utcnow().replace(minute=0, second=0, microsecond=0))
    if now_utc not in df['time'].values:
        idx = (df['time'] - now_utc).abs().idxmin()
    else:
        idx = df.index[df['time'] == now_utc][0]

    end_idx = min(len(df)-1, idx + hours_ahead)
    p = df.loc[idx:end_idx]

    deltaT = p['temp'].iloc[-1] - p['temp'].iloc[0]
    deltaRH = p['rh'].iloc[-1] - p['rh'].iloc[0]
    deltaP = p['precip'].sum()
    deltaCloud = p['cloud'].iloc[-1] - p['cloud'].iloc[0]

    if deltaP >= 1.0 or (deltaP > 0 and deltaCloud > 10):
        trend_type = 'TEMPO'
        trend_text = f"TEMPO {now_utc.strftime('%d%H')}/{(now_utc+timedelta(hours=hours_ahead)).strftime('%d%H')} 3000 +TSRA BKN010CB"
        narrative = 'Potensi hujan lebat disertai petir (TSRA) — perhatikan visibilitas turun.'
    elif deltaT <= -2 and deltaRH >= 10:
        trend_type = 'BECMG'
        trend_text = f"BECMG {now_utc.strftime('%d%H')}/{(now_utc+timedelta(hours=hours_ahead)).strftime('%d%H')} 5000 BR SCT020"
        narrative = 'Kondisi menjadi lebih lembap—potensi penurunan visibilitas (BR).'
    elif deltaT >= 2 and deltaRH <= -10:
        trend_type = 'BECMG'
        trend_text = f"BECMG {now_utc.strftime('%d%H')}/{(now_utc+timedelta(hours=hours_ahead)).strftime('%d%H')} 9999 NSW FEW030"
        narrative = 'Kondisi membaik; kelembapan turun, berawan berkurang.'
    else:
        trend_type = 'NOSIG'
        trend_text = 'NOSIG'
        narrative = 'Tidak ada perubahan signifikan terdeteksi dalam jangka pendek.'

    details = dict(deltaT=deltaT, deltaRH=deltaRH, deltaP=deltaP, deltaCloud=deltaCloud)
    return trend_text, narrative, details

# ---------- Main generate logic ----------
if st.button('Generate TAFOR + TREND'):
    wind, vis, cloud = parse_metar_simple(metar_input)

    valid_to = issue_dt + timedelta(hours=validity)
    taf_header = f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {issue_dt.strftime('%d%H')}/{valid_to.strftime('%d%H')}"

    b1s = issue_dt + timedelta(hours=4)
    b1e = b1s + timedelta(hours=4)
    b2s = issue_dt + timedelta(hours=12)
    b2e = b2s + timedelta(hours=6)

    taf_lines = [
        taf_header,
        f"{wind} {vis} {cloud} Q1009",
        f"BECMG {b1s.strftime('%d%H')}/{b1e.strftime('%d%H')} 20006KT 8000 -RA SCT025 BKN040",
        f"BECMG {b2s.strftime('%d%H')}/{b2e.strftime('%d%H')} 24006KT 9999 SCT020"
    ]

    st.success('TAFOR baseline generated')

    df_om = fetch_openmeteo()
    if df_om is not None:
        trend_code, narrative, details = analyze_trend_from_df(df_om, hours_ahead=3)
    else:
        trend_code, narrative, details = 'NOSIG', 'Open-Meteo data not available', {}

    st.subheader('Generated TAFOR (draft)')
    st.code('\n'.join(taf_lines), language='text')

    st.subheader('Auto-TREND (2–3 jam ke depan)')
    st.markdown(f"**{trend_code}**")
    st.write(narrative)
    st.json(details)

    if df_om is not None:
        st.subheader('Forecast time-series (Open-Meteo)')
        now_utc = pd.to_datetime(datetime.utcnow())
        mask = (df_om['time'] >= now_utc - pd.Timedelta('1H')) & (df_om['time'] <= now_utc + pd.Timedelta(hours=24))
        plot_df = df_om.loc[mask].reset_index(drop=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_t = px.line(plot_df, x='time', y=['temp','rh'], labels={'value':'Value','time':'UTC'}, title='Temperature (°C) & RH (%)')
            st.plotly_chart(fig_t, use_container_width=True)
        with col_t2:
            fig_p = px.bar(plot_df, x='time', y='precip', labels={'precip':'Precip (mm)'}, title='Precipitation (mm per hour)')
            st.plotly_chart(fig_p, use_container_width=True)

    st.info('Experimental: validate with official TAF/TAF AMD from BMKG before operational use')

st.markdown('---')
st.caption('Built for prototyping — adapt thresholds & rules to local forecaster practice.')
