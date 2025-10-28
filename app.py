"""
TAFOR Fusion v2.0 - Streamlit app
Features:
 - BMKG point -> adm4 fallback (Sedati Gede)
 - Open-Meteo hourly baseline
 - METAR chain: OGIMET -> NOAA AviationWeather XML -> cached last METAR
 - Fusion heuristic for TAFOR generation with BECMG / TEMPO / FM rules
 - Sidebar thresholds & tuning
 - Plots (plotly) for wind speed/dir, precipitation, cloud fraction
 - Caching of METAR and TAF logs (./cache)

Usage:
 - pip install -r requirements_v2.txt
 - streamlit run TAFOR_Fusion_v2_app.py

Note: This is heuristic fusion (not formal DA). Validate with official sources before operational use.
"""

import streamlit as st
import requests
import pandas as pd
import os
import re
from datetime import datetime, timedelta, date, time
import math
from dateutil import parser as dparser

# -------------------- CONFIG --------------------
CACHE_DIR = "./cache"
if not os.path.isdir(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
METAR_CACHE = os.path.join(CACHE_DIR, "metar_last.txt")
TAF_LOG_CSV = os.path.join(CACHE_DIR, "tafor_log.csv")

LAT = -7.380
LON = 112.786
ADM4_SEDATI_GDE = "35.15.17.2011"
BMKG_POINT_URL = f"https://cuaca.bmkg.go.id/api/df/v1/forecast/point?lat={LAT}&lon={LON}"
BMKG_ADM4_URL = f"https://cuaca.bmkg.go.id/api/df/v1/forecast/adm?adm4={ADM4_SEDATI_GDE}"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
OGIMET_METAR_URL = "https://ogimet.com/display_metars2.php?lang=en&lugar=WARR&tipo=ALL&ord=REV&nil=SI&fmt=txt"
NOAA_METAR_XML = (
    "https://aviationweather.gov/adds/dataserver_current/httpparam?"
    "datasource=metars&requestType=retrieve&format=xml&stationString=WARR&hoursBeforeNow=3"
)

# -------------------- UI --------------------
st.set_page_config(page_title="TAFOR Fusion v2.0 — WARR", layout="wide")
st.title("🛫 TAFOR Fusion v2.0 — WARR (Juanda)")
st.markdown("Fusion BMKG + Open-Meteo + METAR (chain). Operational-style UI with tuning and plots.")

# Sidebar controls (thresholds & options)
st.sidebar.header("Settings & Thresholds")
WIND_SPEED_CHANGE_KT = st.sidebar.slider("Wind speed change threshold (kt)", 2, 12, 6)
WIND_DIR_CHANGE_DEG = st.sidebar.slider("Wind direction change threshold (deg)", 10, 90, 30)
VISIBILITY_THRESHOLD_M = st.sidebar.number_input("Visibility threshold (m)", value=5000)
PRECIP_MIN_MM = st.sidebar.number_input("Precipitation onset threshold (mm)", value=0.3)
CLOUD_FRACTION_THRESHOLD = st.sidebar.slider("Cloud fraction threshold (0-100%)", 0, 100, 40) / 100.0

use_noaa = st.sidebar.checkbox("Include NOAA AviationWeather fallback for METAR", value=True)
cache_metar = st.sidebar.checkbox("Cache METAR locally when fetched", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("Reset cached METAR & logs"):
    if os.path.exists(METAR_CACHE): os.remove(METAR_CACHE)
    if os.path.exists(TAF_LOG_CSV): os.remove(TAF_LOG_CSV)
    st.sidebar.success("Cache cleared")

# Issue time and validity
st.subheader("🕓 Issue time & validity (UTC)")
col1, col2 = st.columns([1, 1])
with col1:
    issue_date = st.date_input("Issue date (UTC)", value=date.today())
with col2:
    default_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0).time()
    issue_time = st.time_input("Issue time (UTC)", value=default_time)
run_dt = datetime.combine(issue_date, issue_time)
hours = st.selectbox("Validity (hours)", [6, 9, 12, 18, 24], index=4)

st.markdown("---")

# -------------------- Helpers & fetchers --------------------

def mps_to_kt(mps):
    return mps * 1.94384 if mps is not None else 0.0

def deg_to_cardinal10(deg):
    try:
        d = int((deg + 5) // 10 * 10) % 360
        return f"{d:03d}"
    except:
        return "000"

def wind_vector(speed_kt, dir_deg):
    rad = math.radians((270 - dir_deg) % 360)
    u = speed_kt * math.cos(rad); v = speed_kt * math.sin(rad)
    return u, v

def vector_to_wind(u, v):
    speed = math.hypot(u, v)
    ang = (270 - math.degrees(math.atan2(v, u))) % 360
    return speed, ang

# METAR chain: OGIMET -> NOAA XML -> cache

def fetch_metar_ogimet():
    try:
        r = requests.get(OGIMET_METAR_URL, timeout=10)
        r.raise_for_status()
        txt = r.text.strip()
        for line in txt.splitlines():
            if line.strip().startswith("WARR"):
                return line.strip()
        return None
    except Exception:
        return None

import xml.etree.ElementTree as ET

def fetch_metar_noaa():
    try:
        r = requests.get(NOAA_METAR_XML, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        metars = root.findall('.//METAR')
        if metars:
            raw = metars[0].findtext('raw_text') or metars[0].findtext('raw_ob') or ''
            return raw.strip()
        return None
    except Exception:
        return None


def cache_metar_write(text):
    try:
        with open(METAR_CACHE, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception:
        pass


def cache_metar_read():
    try:
        if os.path.exists(METAR_CACHE):
            with open(METAR_CACHE, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception:
        pass
    return None

# Basic METAR parse

def parse_metar(metar_text):
    if not metar_text: return None
    wind_dir = None; wind_spd = None
    m = re.search(r" (\d{3}|VRB)(\d{2,3})KT", metar_text)
    if m:
        wind_dir = 0 if m.group(1) == 'VRB' else int(m.group(1)); wind_spd = int(m.group(2))
    vis = None
    mvis = re.search(r" (\d{4}) ", metar_text)
    if mvis:
        try: vis = int(mvis.group(1))
        except: vis = None
    else:
        msm = re.search(r" (\d+\/\d+|\d+)SM", metar_text)
        if msm:
            try:
                val = msm.group(1)
                if '/' in val:
                    num, den = val.split('/')
                    vis = int(float(num)/float(den)*1609.344)
                else:
                    vis = int(float(val)*1609.344)
            except:
                vis = None
    clouds = re.findall(r"(FEW|SCT|BKN|OVC)\d{3}", metar_text)
    wx_tags = [t for t in ["TS","SHRA","RA","DZ","FG","BR","SH"] if t in metar_text]
    return {"raw":metar_text, "wind_dir":wind_dir, "wind_speed_kt":wind_spd, "visibility_m":vis, "clouds":clouds, "wx":wx_tags}

# BMKG fetch (point then adm4)

def fetch_bmkg():
    try:
        r = requests.get(BMKG_POINT_URL, timeout=8)
        j = r.json()
        if j and j.get('data'):
            return {'type':'point','json':j}
    except Exception:
        pass
    try:
        r2 = requests.get(BMKG_ADM4_URL, timeout=8)
        j2 = r2.json()
        if j2 and j2.get('data'):
            return {'type':'adm4','json':j2}
    except Exception:
        pass
    return None

# Open-Meteo fetch

def fetch_openmeteo(lat, lon, hours=24):
    params = {
        'latitude': lat, 'longitude': lon,
        'hourly':'temperature_2m,windspeed_10m,winddirection_10m,precipitation,cloudcover,visibility',
        'forecast_days': max(1, (hours//24)+1), 'timezone':'UTC'
    }
    r = requests.get(OPENMETEO_URL, params=params, timeout=12)
    r.raise_for_status()
    return r.json()

# build fused hourly

def build_fused_hours(open_json, metar=None, hours=24):
    res = []
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    times = open_json['hourly']['time']
    ws = open_json['hourly'].get('windspeed_10m', [])
    wd = open_json['hourly'].get('winddirection_10m', [])
    prec = open_json['hourly'].get('precipitation', [])
    cc = open_json['hourly'].get('cloudcover', [])
    vis = open_json['hourly'].get('visibility', [])
    for i, tstr in enumerate(times):
        t = dparser.isoparse(tstr)
        if t < now: continue
        if (t - now).total_seconds() > hours*3600: break
        try:
            ws_kt = mps_to_kt(ws[i]) if i < len(ws) else 0.0
        except:
            ws_kt = 0.0
        wd_deg = float(wd[i]) if i < len(wd) else 0.0
        precip_mm = float(prec[i]) if i < len(prec) else 0.0
        cloud_frac = (float(cc[i])/100.0) if i < len(cc) else 0.0
        vis_m = int(vis[i]) if i < len(vis) and vis[i] is not None else 10000
        entry = {'time':t, 'wind_speed_kt':round(ws_kt,1), 'wind_dir_deg':round(wd_deg,1),
                 'precip_mm':round(precip_mm,2), 'cloud_frac':round(cloud_frac,2), 'visibility_m':int(vis_m)}
        res.append(entry)
    # bias first hour toward METAR if available
    if metar and res:
        try:
            first = res[0]
            if metar.get('wind_speed_kt') is not None:
                first['wind_speed_kt'] = round((first['wind_speed_kt'] + metar['wind_speed_kt'])/2.0,1)
            if metar.get('wind_dir') is not None and metar.get('wind_speed_kt') is not None:
                u1,v1 = wind_vector(first['wind_speed_kt'], first['wind_dir_deg'])
                u2,v2 = wind_vector(metar['wind_speed_kt'], metar['wind_dir'])
                u = (u1+u2)/2; v = (v1+v2)/2
                spd, ang = vector_to_wind(u,v)
                first['wind_speed_kt'] = round(spd,1); first['wind_dir_deg'] = round(ang,1)
            if metar.get('visibility_m'):
                first['visibility_m'] = int((first['visibility_m'] + metar['visibility_m'])/2)
            if metar.get('clouds'):
                if any(c.startswith('BKN') or c.startswith('OVC') for c in metar['clouds']):
                    first['cloud_frac'] = max(first['cloud_frac'], 0.5)
        except Exception:
            pass
    return res

# detect events

def detect_events(hourly, wind_spd_thresh, wind_dir_thresh, vis_thresh, precip_thresh, cloud_frac_thresh):
    evs = []
    if not hourly: return evs
    base = hourly[0]
    for p in hourly[1:]:
        ws_diff = abs(p['wind_speed_kt'] - base['wind_speed_kt'])
        wd_diff = abs(((p['wind_dir_deg'] - base['wind_dir_deg'] + 180) % 360) - 180)
        vis_drop = p['visibility_m'] < vis_thresh and base['visibility_m'] >= vis_thresh
        precip_onset = p['precip_mm'] >= precip_thresh and base['precip_mm'] < precip_thresh
        cloud_increase = p['cloud_frac'] >= cloud_frac_thresh and base['cloud_frac'] < cloud_frac_thresh
        if ws_diff >= wind_spd_thresh or wd_diff >= wind_dir_thresh:
            evs.append({'type':'BECMG', 'start':p['time'], 'end':p['time']+timedelta(hours=3), 'sample':p,
                        'reason':f"Wind change {base['wind_speed_kt']}kt/{int(base['wind_dir_deg'])}°→{p['wind_speed_kt']}kt/{int(p['wind_dir_deg'])}°"})
            base = p
        elif vis_drop or precip_onset or cloud_increase:
            evs.append({'type':'TEMPO', 'start':p['time'], 'end':p['time']+timedelta(hours=3), 'sample':p,
                        'reason':f"{'Vis drop' if vis_drop else ''}{' Precip onset' if precip_onset else ''}{' Cloud increase' if cloud_increase else ''}"})
    for e in evs:
        if e['type']=='BECMG' and (e['end']-e['start']).total_seconds() <= 3600:
            e['type']='FM'
    return evs

# cloud group

def cloud_group_from_frac(frac):
    if frac >= 0.75: return 'OVC020'
    if frac >= 0.4: return 'BKN030'
    if frac >= 0.1: return 'SCT050'
    return 'SKC'

# format visibility

def format_vis(vis_m):
    return str(int(vis_m)) if vis_m is not None else '9999'

# build taf

def build_taf(issue_time, hourly, events, validity_hours=24):
    if not hourly: return 'TAF not available - no hourly points.'
    now = issue_time.replace(minute=0, second=0, microsecond=0)
    end = now + timedelta(hours=validity_hours)
    header = f"TAF WARR {issue_time.strftime('%d%H%M')}Z {now.strftime('%d%H')}/{end.strftime('%d%H')}"
    p1_end = now + timedelta(hours=6)
    p2_end = now + timedelta(hours=12)
    p3_end = end
    def pick(s,e):
        cands = [p for p in hourly if s <= p['time'] < e]
        if not cands: return hourly[0]
        return cands[len(cands)//2]
    pt1 = pick(now, p1_end); pt2 = pick(p1_end, p2_end); pt3 = pick(p2_end, p3_end)
    lines = [header]
    for (s,e,pt) in [(now,p1_end,pt1),(p1_end,p2_end,pt2),(p2_end,p3_end,pt3)]:
        wind = f"{deg_to_cardinal10(pt['wind_dir_deg'])}{int(round(pt['wind_speed_kt'])):02d}KT"
        vis = format_vis(pt.get('visibility_m',9999))
        wx = ''
        if pt['precip_mm'] >= PRECIP_MIN_MM:
            wx = '-RA' if pt['precip_mm'] < 5 else 'RA'
        cloud = cloud_group_from_frac(pt['cloud_frac'])
        lines.append(f"{s.strftime('%d%H%M')}Z {s.strftime('%d%H')}/{e.strftime('%d%H')} {wind} {vis} {wx} {cloud}".replace('  ',' ').strip())
    event_lines = []
    for ev in events:
        pt = ev['sample']; typ = ev['type']; start = ev['start']; endt = ev['end']
        wind = f"{deg_to_cardinal10(pt['wind_dir_deg'])}{int(round(pt['wind_speed_kt'])):02d}KT"
        vis = format_vis(pt.get('visibility_m',9999))
        wx = '-RA' if pt['precip_mm'] >= PRECIP_MIN_MM else ''
        cloud = cloud_group_from_frac(pt['cloud_frac'])
        event_lines.append(f"{typ} {start.strftime('%d%H%M')}Z/{endt.strftime('%d%H%M')}Z {wind} {vis} {wx} {cloud}".replace('  ',' ').strip())
    detail = lines[1:4]
    if event_lines:
        detail.append(event_lines[0])
    output = '\n'.join([header] + detail[:4])
    return output

# -------------------- Main action --------------------

if st.button('Generate TAFOR v2.0 (Fuse BMKG/OpenMeteo/METAR)'):
    st.info('Fetching BMKG (point -> adm4 fallback) ...')
    bmkg = fetch_bmkg()
    if bmkg:
        st.success(f"BMKG source: {bmkg['type']}")
    else:
        st.warning('BMKG not available. Proceed with Open-Meteo + METAR only.')

    st.info('Fetching Open-Meteo ...')
    try:
        openj = fetch_openmeteo(LAT, LON, hours=hours)
        st.success('Open-Meteo OK')
    except Exception as e:
        st.error(f'Open-Meteo fetch failed: {e}')
        st.stop()

    st.info('Fetching METAR (OGIMET -> NOAA -> cache) ...')
    metar_txt = None
    metar_txt = fetch_metar_ogimet()
    if not metar_txt and use_noaa:
        metar_txt = fetch_metar_noaa()
    if not metar_txt:
        cached = cache_metar_read()
        if cached:
            st.warning('Using cached METAR (last available).')
            metar_txt = cached
    if metar_txt:
        st.success('METAR available')
        st.code(metar_txt)
        metar_parsed = parse_metar(metar_txt)
        st.write('METAR parsed:', metar_parsed)
        if cache_metar and metar_txt:
            cache_metar_write(metar_txt)
    else:
        st.warning('METAR not available via OGIMET/NOAA nor cache.')
        metar_parsed = None

    hourly = build_fused_hours(openj, metar=metar_parsed, hours=hours)
    st.write(f'Built {len(hourly)} hourly fused points (next {hours}h).')

    df_preview = pd.DataFrame([{'time':p['time'], 'ws_kt':p['wind_speed_kt'], 'wd_deg':p['wind_dir_deg'],
                                'precip_mm':p['precip_mm'], 'cloud_frac':p['cloud_frac'], 'vis_m':p['visibility_m']} for p in hourly])
    st.dataframe(df_preview.head(24))

    events = detect_events(hourly, WIND_SPEED_CHANGE_KT, WIND_DIR_CHANGE_DEG, VISIBILITY_THRESHOLD_M, PRECIP_MIN_MM, CLOUD_FRACTION_THRESHOLD)
    if events:
        st.write('Detected change events:')
        for e in events:
            st.write(f"- {e['type']} {e['start']} → {e['reason']}")
    else:
        st.info('No significant change events detected.')

    taf_text = build_taf(run_dt, hourly, events, validity_hours=hours)
    st.markdown('### ✅ GENERATED TAFOR (WARR)')
    st.code(taf_text, language='text')

    # copyable text area + download
    st.text_area('TAFOR (copy)', value=taf_text, height=200)
    st.download_button('Download TAFOR (TXT)', data=taf_text, file_name=f'TAFOR_WARR_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.txt', mime='text/plain')

    # log TAF
    try:
        row = {'issued_at':run_dt.isoformat(), 'tafor':taf_text}
        if os.path.exists(TAF_LOG_CSV):
            df_log = pd.read_csv(TAF_LOG_CSV)
            df_log = df_log.append(row, ignore_index=True)
        else:
            df_log = pd.DataFrame([row])
        df_log.to_csv(TAF_LOG_CSV, index=False)
        st.success('TAF logged to cache/tafor_log.csv')
    except Exception:
        st.warning('Failed to write TAF log (check permissions).')

    # simple plots (windspeed, precip, cloud)
    try:
        import plotly.express as px
        p1 = px.line(df_preview, x='time', y='ws_kt', title='Wind speed (kt)')
        p2 = px.line(df_preview, x='time', y='precip_mm', title='Precipitation (mm)')
        p3 = px.line(df_preview, x='time', y='cloud_frac', title='Cloud fraction')
        st.plotly_chart(p1, use_container_width=True)
        st.plotly_chart(p2, use_container_width=True)
        st.plotly_chart(p3, use_container_width=True)
    except Exception:
        pass

st.markdown('---')
st.caption('TAFOR Fusion v2.0 — heuristic fusion. Tune thresholds in sidebar. Validate with official reports before operational use.')
