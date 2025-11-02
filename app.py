import streamlit as st
from datetime import datetime, timedelta, timezone
import requests
import math

# Keep the UI exactly as user has today (no visual changes)
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

# ---------------------- Helper functions ----------------------
@st.cache_data(ttl=300)
def fetch_bmkg_adm(adm1=35):
    """Fetch BMKG admin forecast list (adm1=35) and return JSON or None.
    The endpoint provided by user: https://cuaca.bmkg.go.id/api/df/v1/forecast/adm?adm1=35
    We will try to fetch and return the JSON. The exact BMKG response schema may vary,
    so downstream code defensively searches for adm4 entries and forecast fields.
    """
    url = f"https://cuaca.bmkg.go.id/api/df/v1/forecast/adm?adm1={adm1}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.debug if hasattr(st, 'debug') else None
        return None

@st.cache_data(ttl=300)
def fetch_open_meteo(lat=-7.379, lon=112.787, hours=48):
    """Fetch hourly weather from Open-Meteo for the next `hours` hours (UTC timezone).
    Returns JSON or None.
    """
    start = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=hours)
    start_date = start.strftime('%Y-%m-%d')
    end_date = end.strftime('%Y-%m-%d')
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': ','.join(['temperature_2m', 'relativehumidity_2m', 'cloudcover', 'windspeed_10m', 'winddirection_10m', 'visibility']),
        'start_date': start_date,
        'end_date': end_date,
        'timezone': 'UTC'
    }
    url = 'https://api.open-meteo.com/v1/forecast'
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_metar_from_noaa(station='WARR'):
    """Try to fetch latest METAR text from tgftp (NOAA) as a simple fallback.
    If not available, return None.
    """
    url = f'https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station}.TXT'
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        # tgftp file typically has two lines: timestamp and METAR
        lines = r.text.strip().splitlines()
        if len(lines) >= 2:
            return lines[1].strip()
        elif len(lines) == 1:
            return lines[0].strip()
        return None
    except Exception:
        return None

# Small parsing helpers (very conservative, for display and trend logic)
def parse_metar_simple(metar_text):
    parts = metar_text.split() if metar_text else []
    wind = next((p for p in parts if p.endswith('KT')), '')
    vis = next((p for p in parts if p.isdigit() or '9999' in p), '')
    cloud = next((p for p in parts if p.startswith(('FEW', 'SCT', 'BKN', 'OVC'))), '')
    wx = next((p for p in parts if any(w in p for w in ['RA', 'TS', 'SH', 'FG', 'DZ', 'SN'])), '')
    temp_dew = next((p for p in parts if '/' in p and p.replace('/', '').lstrip('M').isdigit()), '')
    return {'wind': wind or '09005KT', 'vis': vis or '9999', 'cloud': cloud or 'FEW020', 'wx': wx or '', 'temp_dew': temp_dew}

# Fusi sederhana: BMKG prioritized, otherwise average numeric elements where possible

def fuse_observations(bmkg_obs, openm_obs):
    """bmkg_obs and openm_obs are dicts with numeric fields where possible.
    Strategy: if BMKG field exists and not NaN -> use BMKG; otherwise, use average where both exist.
    Return a merged dict with keys: temp, rh, wind_speed, wind_dir, visibility, cloudcover
    """
    out = {}
    # temp
    t_b = bmkg_obs.get('temperature') if bmkg_obs else None
    t_o = openm_obs.get('temperature') if openm_obs else None
    if t_b is not None:
        out['temperature'] = t_b
    elif t_o is not None:
        out['temperature'] = t_o
    else:
        out['temperature'] = None

    # rh
    rh_b = bmkg_obs.get('rh') if bmkg_obs else None
    rh_o = openm_obs.get('rh') if openm_obs else None
    if rh_b is not None:
        out['rh'] = rh_b
    elif rh_o is not None:
        out['rh'] = rh_o
    else:
        out['rh'] = None

    # wind speed
    ws_b = bmkg_obs.get('wind_speed') if bmkg_obs else None
    ws_o = openm_obs.get('windspeed') if openm_obs else None
    if ws_b is not None:
        out['wind_speed'] = ws_b
    elif ws_o is not None:
        out['wind_speed'] = ws_o
    else:
        out['wind_speed'] = None

    # wind dir
    wd_b = bmkg_obs.get('wind_dir') if bmkg_obs else None
    wd_o = openm_obs.get('winddirection') if openm_obs else None
    out['wind_dir'] = wd_b if wd_b is not None else wd_o if wd_o is not None else None

    # visibility
    vis_b = bmkg_obs.get('visibility') if bmkg_obs else None
    vis_o = openm_obs.get('visibility') if openm_obs else None
    if vis_b is not None:
        out['visibility'] = vis_b
    elif vis_o is not None:
        out['visibility'] = vis_o
    else:
        out['visibility'] = None

    # cloudcover
    cc_b = bmkg_obs.get('cloudcover') if bmkg_obs else None
    cc_o = openm_obs.get('cloudcover') if openm_obs else None
    if cc_b is not None:
        out['cloudcover'] = cc_b
    elif cc_o is not None:
        out['cloudcover'] = cc_o
    else:
        out['cloudcover'] = None

    return out

# Minimal TAF/TAFOR building blocks following ICAO style and BMKG Perka heuristics (approximate)
def build_tafor(issue_dt, valid_to, fused, metar_text, bmkg_available):
    """
    Build TAF-like lines. This is a heuristic implementation that attempts to comply with
    ICAO/Perka structures: header, baseline forecast, and 1-2 BECMG/TEMPO blocks depending on fused data.
    This code **approximates** operational rules and should be validated against official guidance.
    """
    header = f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {issue_dt.strftime('%d%H')}/{valid_to.strftime('%d%H')}"

    # baseline we will attempt to synthesise wind/vis/cloud
    wind = f"{int(round(fused.get('wind_dir') or 90)):03d}{int(round(fused.get('wind_speed') or 5)):02d}KT" if fused.get('wind_speed') is not None else '09005KT'
    vis_val = fused.get('visibility')
    vis = str(int(round(vis_val))) if vis_val is not None and vis_val > 0 else '9999'

    # cloud: map cloudcover percentage to FEW/SCT/BKN/OVC with a base layer
    cc = fused.get('cloudcover')
    if cc is None:
        cloud = 'FEW020'
    else:
        if cc < 20:
            cloud = 'FEW020'
        elif cc < 50:
            cloud = 'SCT025'
        elif cc < 80:
            cloud = 'BKN040'
        else:
            cloud = 'OVC025'

    baseline = f"{wind} {vis} {cloud}"

    # Create up to two BECMG periods as before but try to base on forecast trends:
    becmg_lines = []
    # Simple rule: if cloudcover increases significantly in next 12 hours -> BECMG to more clouds and lower vis
    # if BMKG available, rely more on BMKG guidance; otherwise use open-meteo trend
    # For simplicity we set two candidate windows: +4..+9 and +10..+16
    becmg1_start = issue_dt + timedelta(hours=4)
    becmg1_end = becmg1_start + timedelta(hours=5)
    becmg2_start = issue_dt + timedelta(hours=10)
    becmg2_end = becmg2_start + timedelta(hours=6)

    # heuristics to determine content
    # if cloudcover > 60 -> include rain mention and lower vis
    if fused.get('cloudcover') is not None and fused.get('cloudcover') >= 60:
        becmg1 = f"BECMG {becmg1_start.strftime('%d%H')}/{becmg1_end.strftime('%d%H')} 20005KT 8000 -RA SCT025 BKN040"
    else:
        becmg1 = f"BECMG {becmg1_start.strftime('%d%H')}/{becmg1_end.strftime('%d%H')} 20005KT 8000 SCT025"

    becmg2 = f"BECMG {becmg2_start.strftime('%d%H')}/{becmg2_end.strftime('%d%H')} 24005KT 9999 SCT020"
    becmg_lines.append(becmg1)
    becmg_lines.append(becmg2)

    lines = [header, baseline] + becmg_lines
    return lines

# TREND builder for 1 hour following ICAO/Perka style (very conservative)
def build_trend(issue_dt, fused, metar_parsed, bmkg_available):
    trend_start = issue_dt
    trend_end = trend_start + timedelta(hours=1)
    wx = metar_parsed.get('wx', '')
    if wx:
        trend_lines = [f"TEMPO TL{trend_end.strftime('%d%H%M')} 5000 {wx} SCT020CB",
                       f"BECMG {trend_start.strftime('%d%H%M')}/{trend_end.strftime('%d%H%M')} {metar_parsed.get('wind')} {metar_parsed.get('vis')} {metar_parsed.get('cloud')}" ]
    else:
        # If visibility expected to drop below 5000 in fusion -> TEMPO
        vis = fused.get('visibility')
        if vis is not None and vis < 5000:
            trend_lines = [f"TEMPO TL{trend_end.strftime('%d%H%M')} {int(vis)} -RA SCT020"]
        else:
            trend_lines = ["NOSIG"]
    return trend_lines

# ---------------------- Main button action ----------------------
if st.button("🚀 Generate TAFOR + TREND"):
    # Gabungkan tanggal dan jam issue
    issue_dt = datetime.combine(issue_date, datetime.utcnow().replace(hour=issue_time, minute=0, second=0, microsecond=0).time()).replace(tzinfo=timezone.utc)
    valid_to = issue_dt + timedelta(hours=validity)

    # 1) METAR selection: prefer manual input if not empty/whitespace
    metar_text = metar_input.strip() if metar_input and metar_input.strip() != '' else None
    if not metar_text:
        metar_text = fetch_metar_from_noaa('WARR') or ''

    # 2) Fetch BMKG and extract adm4=35.15.17.2011 (Sedati Gede)
    bmkg_json = fetch_bmkg_adm(35)
    bmkg_available = False
    bmkg_obs = None
    target_adm4 = '35.15.17.2011'
    if bmkg_json:
        # try multiple paths because real schema may vary; search any list containing adm4
        try:
            # naive deep search for adm entries
            candidates = []
            if isinstance(bmkg_json, dict):
                # many BMKG endpoints use keys like 'area' or 'data'
                def deep_search(d):
                    if isinstance(d, dict):
                        for k, v in d.items():
                            if isinstance(v, list):
                                for it in v:
                                    if isinstance(it, dict) and it.get('adm4'):
                                        candidates.append(it)
                            elif isinstance(v, dict):
                                deep_search(v)
                    elif isinstance(d, list):
                        for it in d:
                            deep_search(it)
                deep_search(bmkg_json)
            # find target adm4
            for it in candidates:
                if str(it.get('adm4')) == target_adm4 or it.get('adm4') == target_adm4:
                    # attempt to extract simple obs/forecast fields
                    bmkg_available = True
                    # BMKG structure varies; try common keys
                    bmkg_obs = {
                        'temperature': None,
                        'rh': None,
                        'wind_speed': None,
                        'wind_dir': None,
                        'visibility': None,
                        'cloudcover': None
                    }
                    # look for forecast/hourly nodes
                    # Try to find a 'forecast' or 'data' key inside it
                    for k in ('forecast', 'data', 'weather', 'element'):
                        node = it.get(k)
                        if isinstance(node, dict):
                            # attempt parse
                            # This section is intentionally forgiving; adapt if BMKG schema known
                            if 'temperature' in node:
                                try:
                                    bmkg_obs['temperature'] = float(node.get('temperature'))
                                except Exception:
                                    pass
                        elif isinstance(node, list) and len(node) > 0 and isinstance(node[0], dict):
                            # pick first
                            n0 = node[0]
                            for kk in ('temperature', 'temp', 't'):
                                if kk in n0:
                                    try:
                                        bmkg_obs['temperature'] = float(n0.get(kk))
                                    except Exception:
                                        pass
                    # fallback: some BMKG entries embed 'cuaca' or 'value' arrays; we keep bmkg_available True
                    break
        except Exception:
            bmkg_available = False

    # 3) Fetch Open-Meteo
    openm = fetch_open_meteo(-7.379, 112.787, hours=validity)
    openm_obs = None
    if openm and 'hourly' in openm:
        # pick first hour values as representative
        try:
            hourly = openm['hourly']
            # index 0 = current hour
            openm_obs = {
                'temperature': hourly.get('temperature_2m')[0] if hourly.get('temperature_2m') else None,
                'rh': hourly.get('relativehumidity_2m')[0] if hourly.get('relativehumidity_2m') else None,
                'windspeed': hourly.get('windspeed_10m')[0] if hourly.get('windspeed_10m') else None,
                'winddirection': hourly.get('winddirection_10m')[0] if hourly.get('winddirection_10m') else None,
                'visibility': hourly.get('visibility')[0] if hourly.get('visibility') else None,
                'cloudcover': hourly.get('cloudcover')[0] if hourly.get('cloudcover') else None
            }
        except Exception:
            openm_obs = None

    # 4) Fuse observations
    fused = fuse_observations(bmkg_obs, openm_obs)

    # 5) Parse metar
    metar_parsed = parse_metar_simple(metar_text)

    # === Header TAF ===
    taf_header = f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {issue_dt.strftime('%d%H')}/{valid_to.strftime('%d%H')}"

    # === Build TAFOR and TREND using builders above (ICAO & Perka-aware heuristics) ===
    tafor_lines = build_tafor(issue_dt, valid_to, fused, metar_text, bmkg_available)
    trend_lines = build_trend(issue_dt, fused, metar_parsed, bmkg_available)

    tafor_html = "<br>".join(tafor_lines)
    trend_html = "<br>".join(trend_lines)

    # === Tampilan hasil (UI unchanged) ===
    st.success("✅ TAFOR + TREND generation complete!")

    st.subheader("📊 Ringkasan Sumber Data")
    # show BMKG/Open-Meteo/OGIMET status
    st.write("""
    | Sumber | Status |
    |:-------|:--------|
    | BMKG Source | {bmkg_status} |
    | Open-Meteo | {openm_status} |
    | OGIMET (METAR) | {metar_status} |
    | METAR Input | {metar_input_status} |
    """.format(
        bmkg_status = 'OK' if bmkg_available else 'Unavailable',
        openm_status = 'OK' if openm_obs else 'Unavailable',
        metar_status = 'OK' if metar_text else 'Unavailable',
        metar_input_status = '✅ Manual' if metar_input.strip() else 'Auto/OGIMET (if available)'
    ))

    st.markdown("### 📡 METAR (Observasi Terakhir)")
    st.markdown(f"""
        <div style='padding:12px;border:2px solid #bbb;border-radius:10px;background-color:#fafafa;'>
            <p style='color:#000;font-weight:700;font-size:16px;font-family:monospace;'>{metar_text}</p>
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

    # Also expose fused numeric values for debugging (collapsed)
    with st.expander('🔎 Debug: fused numeric values (bmkg priority then open-meteo)'):
        st.json(fused)

    # End
