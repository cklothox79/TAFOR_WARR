"""
Streamlit app: Auto TAFOR Generator — WARR (Juanda)
Fusion: BMKG (point -> adm4 Sedati Gede) + Open-Meteo + METAR (OGIMET)
Outputs a 3-4 line ICAO-like TAF with BECMG / TEMPO / FM when detected.
"""

import streamlit as st
import requests
import pandas as pd
import re
from datetime import datetime, timedelta, date, time
import math
from dateutil import parser as dparser

st.set_page_config(page_title="TAFOR WARR (Fusion)", layout="wide")

# -------------------------
# CONFIG
# -------------------------
LAT = -7.380
LON = 112.786
ADM4_SEDATI_GDE = "35.15.17.2011"  # fallback adm4
BMKG_POINT_URL = f"https://cuaca.bmkg.go.id/api/df/v1/forecast/point?lat={LAT}&lon={LON}"
BMKG_ADM4_URL = f"https://cuaca.bmkg.go.id/api/df/v1/forecast/adm?adm4={ADM4_SEDATI_GDE}"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
OGIMET_METAR_URL = "https://ogimet.com/display_metars2.php?lang=en&lugar=WARR&tipo=ALL&ord=REV&nil=SI&fmt=txt"

# thresholds (tweakable)
WIND_SPEED_CHANGE_KT = 6
WIND_DIR_CHANGE_DEG = 30
VISIBILITY_THRESHOLD_M = 5000
PRECIP_MIN_MM = 0.3
CLOUD_FRACTION_THRESHOLD = 0.4

st.title("🛫 Auto TAFOR Generator — WARR (Juanda)")
st.markdown("Fusion: **BMKG (point → adm4 Sedati Gede)** + **Open-Meteo** + **METAR (OGIMET)** → output TAFOR (ICAO-like).")

# -------------------------
# UI - Issue time & validity
# -------------------------
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

# -------------------------
# Helpers
# -------------------------
def mps_to_kt(mps): return mps * 1.94384
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

# METAR fetch & parse (OGIMET)
def fetch_metar_ogimet():
    try:
        r = requests.get(OGIMET_METAR_URL, timeout=12)
        r.raise_for_status()
        txt = r.text.strip()
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("WARR"):
                return line
        return None
    except Exception:
        return None

def parse_metar(metar_text):
    if not metar_text:
        return None
    wind_dir = None; wind_spd = None
    m = re.search(r" (\d{3}|VRB)(\d{2,3})KT", metar_text)
    if m:
        wind_dir = 0 if m.group(1) == "VRB" else int(m.group(1))
        wind_spd = int(m.group(2))
    # visibility (meters)
    vis = None
    mvis = re.search(r" (\d{4}) ", metar_text)
    if mvis:
        try: vis = int(mvis.group(1))
        except: vis = None
    else:
        msm = re.search(r" (\d+/\d+|\d+)SM", metar_text)
        if msm:
            try:
                val = msm.group(1)
                if "/" in val:
                    num, den = val.split("/")
                    vis = int(float(num)/float(den)*1609.344)
                else:
                    vis = int(float(val)*1609.344)
            except:
                vis = None
    clouds = re.findall(r"(FEW|SCT|BKN|OVC)\d{3}", metar_text)
    wx_tags = []
    for token in ["TS", "SHRA", "RA", "DZ", "FG", "BR", "SH"]:
        if token in metar_text:
            wx_tags.append(token)
    return {"raw": metar_text, "wind_dir": wind_dir, "wind_speed_kt": wind_spd, "visibility_m": vis, "clouds": clouds, "wx": wx_tags}

# BMKG fetch (point or fallback adm4)
def fetch_bmkg():
    try:
        r = requests.get(BMKG_POINT_URL, timeout=10)
        j = r.json()
        if j and j.get("data"):
            return {"type":"point", "json": j}
    except Exception:
        pass
    try:
        r2 = requests.get(BMKG_ADM4_URL, timeout=10)
        j2 = r2.json()
        if j2 and j2.get("data"):
            return {"type":"adm4", "json": j2}
    except Exception:
        pass
    return None

# Open-Meteo fetch
def fetch_openmeteo(lat, lon, hours=24):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,winddirection_10m,precipitation,cloudcover,visibility",
        "forecast_days": max(1, (hours//24)+1),
        "timezone": "UTC"
    }
    r = requests.get(OPENMETEO_URL, params=params, timeout=12)
    r.raise_for_status()
    return r.json()

# Build fused hourly baseline from Open-Meteo, bias first hour with METAR if present
def build_fused_hours(open_json, metar=None, hours=24):
    res = []
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    times = open_json["hourly"]["time"]
    ws = open_json["hourly"].get("windspeed_10m", [])
    wd = open_json["hourly"].get("winddirection_10m", [])
    prec = open_json["hourly"].get("precipitation", [])
    cc = open_json["hourly"].get("cloudcover", [])
    vis = open_json["hourly"].get("visibility", [])
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
        entry = {"time":t, "wind_speed_kt":round(ws_kt,1), "wind_dir_deg":round(wd_deg,1),
                 "precip_mm":round(precip_mm,2), "cloud_frac":round(cloud_frac,2), "visibility_m":int(vis_m)}
        res.append(entry)
    # bias first hour toward METAR if available
    if metar and res:
        try:
            first = res[0]
            if metar.get("wind_speed_kt") is not None:
                first["wind_speed_kt"] = round((first["wind_speed_kt"] + metar["wind_speed_kt"])/2.0,1)
            if metar.get("wind_dir") is not None and metar.get("wind_speed_kt") is not None:
                u1,v1 = wind_vector(first["wind_speed_kt"], first["wind_dir_deg"])
                u2,v2 = wind_vector(metar["wind_speed_kt"], metar["wind_dir"])
                u = (u1+u2)/2; v = (v1+v2)/2
                spd, ang = vector_to_wind(u,v)
                first["wind_speed_kt"] = round(spd,1); first["wind_dir_deg"] = round(ang,1)
            if metar.get("visibility_m"):
                first["visibility_m"] = int((first["visibility_m"] + metar["visibility_m"])/2)
            # clouds: if METAR shows BKN/OVC, raise cloud_frac
            if metar.get("clouds"):
                if any(c.startswith("BKN") or c.startswith("OVC") for c in metar["clouds"]):
                    first["cloud_frac"] = max(first["cloud_frac"], 0.5)
        except Exception:
            pass
    return res

# detect change events
def detect_events(hourly):
    evs = []
    if not hourly: return evs
    base = hourly[0]
    for p in hourly[1:]:
        ws_diff = abs(p["wind_speed_kt"] - base["wind_speed_kt"])
        wd_diff = abs(((p["wind_dir_deg"] - base["wind_dir_deg"] + 180) % 360) - 180)
        vis_drop = p["visibility_m"] < VISIBILITY_THRESHOLD_M and base["visibility_m"] >= VISIBILITY_THRESHOLD_M
        precip_onset = p["precip_mm"] >= PRECIP_MIN_MM and base["precip_mm"] < PRECIP_MIN_MM
        cloud_increase = p["cloud_frac"] >= CLOUD_FRACTION_THRESHOLD and base["cloud_frac"] < CLOUD_FRACTION_THRESHOLD
        if ws_diff >= WIND_SPEED_CHANGE_KT or wd_diff >= WIND_DIR_CHANGE_DEG:
            evs.append({"type":"BECMG", "start":p["time"], "end":p["time"]+timedelta(hours=3), "sample":p,
                        "reason":f"Wind change {base['wind_speed_kt']}kt/{int(base['wind_dir_deg'])}°→{p['wind_speed_kt']}kt/{int(p['wind_dir_deg'])}°"})
            base = p
        elif vis_drop or precip_onset or cloud_increase:
            evs.append({"type":"TEMPO", "start":p["time"], "end":p["time"]+timedelta(hours=3), "sample":p,
                        "reason":f"{'Vis drop' if vis_drop else ''}{' Precip onset' if precip_onset else ''}{' Cloud increase' if cloud_increase else ''}"})
    # Convert very short BECMG into FM if needed
    for e in evs:
        if e["type"]=="BECMG" and (e["end"]-e["start"]).total_seconds() <= 3600:
            e["type"]="FM"
    return evs

def cloud_group_from_frac(frac):
    if frac >= 0.75: return "OVC020"
    if frac >= 0.4: return "BKN030"
    if frac >= 0.1: return "SCT050"
    return "SKC"

def format_vis(vis_m):
    return str(int(vis_m)) if vis_m is not None else "9999"

# build TAF string
def build_taf(issue_time, hourly, events, validity_hours=24):
    if not hourly: return "TAF not available - no hourly points."
    now = issue_time.replace(minute=0, second=0, microsecond=0)
    end = now + timedelta(hours=validity_hours)
    header = f"TAF WARR {issue_time.strftime('%d%H%M')}Z {now.strftime('%d%H')}/{end.strftime('%d%H')}"
    # split periods: 0-6,6-12,12-end
    p1_end = now + timedelta(hours=6)
    p2_end = now + timedelta(hours=12)
    p3_end = end
    def pick(s,e):
        cands = [p for p in hourly if s <= p["time"] < e]
        if not cands: return hourly[0]
        return cands[len(cands)//2]
    pt1 = pick(now, p1_end)
    pt2 = pick(p1_end, p2_end)
    pt3 = pick(p2_end, p3_end)
    lines = [header]
    for (s,e,pt) in [(now,p1_end,pt1),(p1_end,p2_end,pt2),(p2_end,p3_end,pt3)]:
        wind = f"{deg_to_cardinal10(pt['wind_dir_deg'])}{int(round(pt['wind_speed_kt'])):02d}KT"
        vis = format_vis(pt.get("visibility_m",9999))
        wx = ""
        if pt["precip_mm"] >= PRECIP_MIN_MM:
            wx = "-RA" if pt["precip_mm"] < 5 else "RA"
        cloud = cloud_group_from_frac(pt["cloud_frac"])
        lines.append(f"{s.strftime('%d%H%M')}Z {s.strftime('%d%H')}/{e.strftime('%d%H')} {wind} {vis} {wx} {cloud}".replace("  "," ").strip())
    # include first event if exists (limit detail lines to 4)
    event_lines = []
    for ev in events:
        pt = ev["sample"]
        typ = ev["type"]
        start = ev["start"]; endt = ev["end"]
        wind = f"{deg_to_cardinal10(pt['wind_dir_deg'])}{int(round(pt['wind_speed_kt'])):02d}KT"
        vis = format_vis(pt.get("visibility_m",9999))
        wx = "-RA" if pt["precip_mm"] >= PRECIP_MIN_MM else ""
        cloud = cloud_group_from_frac(pt["cloud_frac"])
        event_lines.append(f"{typ} {start.strftime('%d%H%M')}Z/{endt.strftime('%d%H%M')}Z {wind} {vis} {wx} {cloud}".replace("  "," ").strip())
    detail = lines[1:4]  # three period lines
    if event_lines:
        detail.append(event_lines[0])
    output = "\n".join([header] + detail[:4])
    return output

# -------------------------
# MAIN ACTION
# -------------------------
if st.button("Generate TAFOR (Fuse BMKG/OpenMeteo/METAR)"):
    st.info("Fetching BMKG (point -> adm4 fallback)...")
    bmkg = fetch_bmkg()
    if bmkg:
        st.success(f"BMKG source: {bmkg['type']}")
    else:
        st.warning("BMKG not available (point & adm4). Will proceed with Open-Meteo + METAR only.")

    st.info("Fetching Open-Meteo...")
    try:
        openj = fetch_openmeteo(LAT, LON, hours=hours)
        st.success("Open-Meteo OK")
    except Exception as e:
        st.error(f"Open-Meteo fetch failed: {e}")
        st.stop()

    st.info("Fetching METAR (OGIMET)...")
    metar_txt = fetch_metar_ogimet()
    if metar_txt:
        st.success("METAR fetched (OGIMET)")
        st.code(metar_txt)
        metar_parsed = parse_metar(metar_txt)
        st.write("METAR parsed:", metar_parsed)
    else:
        st.warning("METAR not available via OGIMET.")
        metar_parsed = None

    # build fused hourly list
    hourly = build_fused_hours(openj, metar=metar_parsed, hours=hours)
    st.write(f"Built {len(hourly)} hourly fused points (next {hours}h).")
    df_preview = pd.DataFrame([{"time":p["time"], "ws_kt":p["wind_speed_kt"], "wd_deg":p["wind_dir_deg"],
                                "precip_mm":p["precip_mm"], "cloud_frac":p["cloud_frac"], "vis_m":p["visibility_m"]} for p in hourly])
    st.dataframe(df_preview.head(24))

    # detect events
    events = detect_events(hourly)
    if events:
        st.write("Detected change events:")
        for e in events:
            st.write(f"- {e['type']} {e['start']} → {e['reason']}")
    else:
        st.info("No significant change events detected.")

    taf_text = build_taf(run_dt, hourly, events, validity_hours=hours)
    st.markdown("### ✅ GENERATED TAFOR (WARR)")
    st.code(taf_text, language="text")
    st.download_button("Download TAFOR (TXT)", data=taf_text, file_name=f"TAFOR_WARR_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.txt", mime="text/plain")
    st.success("TAFOR generation complete. Adjust thresholds in the script if you need stricter/looser sensitivity.")

st.markdown("---")
st.caption("Notes: Fusion is heuristic (Open-Meteo baseline, METAR biases nowcast, BMKG as fallback). For operational use, validate against official TAF/METAR and tune thresholds.")
