import streamlit as st
import requests
import pandas as pd
import re
from datetime import datetime, timedelta
import math
from dateutil import parser as dparser

st.set_page_config(page_title="TAFOR WARR (Fusion BMKG + OpenMeteo + METAR)", layout="wide")

# -------------------------
# CONFIG
# -------------------------
LAT = -7.380
LON = 112.786
ADM4_SEDATI_GDE = "35.15.17.2011"  # primary fallback desa Sedati Gede
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

VALIDITY_HOURS = 24

st.title("🛫 Auto TAFOR Generator — WARR (Juanda)")
st.markdown("Fusion: **BMKG (point → adm4 Sedati Gede)** + **Open-Meteo** + **METAR (OGIMET)** → output TAFOR (ICAO-like).")

# UI controls
col1, col2 = st.columns([1, 1])
with col1:
    run_dt = st.datetime_input("Issue time (UTC)", value=datetime.utcnow().replace(minute=0, second=0, microsecond=0))
with col2:
    hours = st.selectbox("Validity hours", [6, 9, 12, 18, 24], index=4)

st.markdown("---")

# -------------------------
# Helpers
# -------------------------
def kph_to_kt(kph): return kph / 1.852
def mps_to_kt(mps): return mps * 1.94384
def deg_to_cardinal10(deg):
    d = int((deg + 5) // 10 * 10) % 360
    return f"{d:03d}"
def wind_vector(speed_kt, dir_deg):
    rad = math.radians((270 - dir_deg) % 360)
    u = speed_kt * math.cos(rad); v = speed_kt * math.sin(rad)
    return u, v
def vector_to_wind(u, v):
    speed = math.hypot(u, v)
    ang = (270 - math.degrees(math.atan2(v, u))) % 360
    return speed, ang

# Parse OGIMET plain text for latest METAR line starting with WARR
def fetch_metar_ogimet():
    try:
        r = requests.get(OGIMET_METAR_URL, timeout=12)
        r.raise_for_status()
        txt = r.text.strip()
        # OGIMET returns possibly multiple lines; find first line starting with WARR
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("WARR"):
                return line
        return None
    except Exception as e:
        return None

# Basic METAR parse (wind, vis, wx, clouds)
def parse_metar(metar_text):
    if not metar_text:
        return None
    # wind: e.g. 09010KT or VRB03KT
    wind_m = re.search(r" (\d{3}|VRB)(\d{2,3})KT", metar_text)
    wind_dir = None; wind_spd = None
    if wind_m:
        try:
            wind_dir = 0 if wind_m.group(1) == "VRB" else int(wind_m.group(1))
            wind_spd = int(wind_m.group(2))
        except:
            pass
    # visibility: metric or SM; try meters like 8000 or 9999, or "1/2SM"
    vis_m = None
    vis_m_m = re.search(r" (\d{4}) ", metar_text)
    if vis_m_m:
        try: vis_m = int(vis_m_m.group(1))
        except: vis_m = None
    else:
        # try SM
        sm_m = re.search(r" (\d+/\d+|\d+)SM", metar_text)
        if sm_m:
            try:
                val = sm_m.group(1)
                if "/" in val:
                    num, den = val.split("/")
                    vis_m = float(num) / float(den) * 1609.344
                else:
                    vis_m = float(val) * 1609.344
            except:
                vis_m = None
    # clouds: collect e.g. FEW020, BKN015, OVC100
    clouds = re.findall(r"(FEW|SCT|BKN|OVC)\d{3}", metar_text)
    # weather phenomenas like -RA, SHRA, TS
    wx = re.findall(r"(-|\+)?(RA|SHRA|TS|DZ|SN|FG|BR|BCFG|SH)?", metar_text)
    # simplify wx by searching common substrings
    wx_tags = []
    for token in ["TS", "SHRA", "RA", "DZ", "FG", "BR", "SH"]:
        if token in metar_text:
            wx_tags.append(token)
    return {"raw": metar_text, "wind_dir": wind_dir, "wind_speed_kt": wind_spd, "visibility_m": vis_m, "clouds": clouds, "wx": wx_tags}

# fetch BMKG point or adm4 fallback
def fetch_bmkg():
    # try point first
    try:
        r = requests.get(BMKG_POINT_URL, timeout=10)
        j = r.json()
        if j and j.get("data"):
            return {"type":"point", "json": j}
    except:
        pass
    # fallback adm4
    try:
        r2 = requests.get(BMKG_ADM4_URL, timeout=10)
        j2 = r2.json()
        if j2 and j2.get("data"):
            return {"type":"adm4", "json": j2}
    except:
        pass
    return None

# fetch open-meteo hourly
def fetch_openmeteo(lat, lon, hours=VALIDITY_HOURS):
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

# Build hourly fused points (list of dicts) for next N hours based on OpenMeteo baseline
def build_fused_hours(open_json, bmkg_json=None, metar=None, hours=VALIDITY_HOURS):
    res = []
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    time_arr = open_json["hourly"]["time"]
    ws_arr = open_json["hourly"].get("windspeed_10m", [])
    wd_arr = open_json["hourly"].get("winddirection_10m", [])
    prec_arr = open_json["hourly"].get("precipitation", [])
    cloud_arr = open_json["hourly"].get("cloudcover", [])
    vis_arr = open_json["hourly"].get("visibility", [])
    for i, tstr in enumerate(time_arr):
        t = dparser.isoparse(tstr)
        if t < now: continue
        if (t - now).total_seconds() > hours * 3600: break
        # default from OpenMeteo
        ws_kt = mps_to_kt(ws_arr[i]) if i < len(ws_arr) else 0.0
        wd_deg = float(wd_arr[i]) if i < len(wd_arr) else 0.0
        precip_mm = float(prec_arr[i]) if i < len(prec_arr) else 0.0
        cloud_frac = (float(cloud_arr[i]) / 100.0) if i < len(cloud_arr) else 0.0
        vis_m = int(vis_arr[i]) if i < len(vis_arr) and vis_arr[i] is not None else 10000
        entry = {"time": t, "wind_speed_kt": round(ws_kt,1), "wind_dir_deg": round(wd_deg,1),
                 "precip_mm": round(precip_mm,2), "cloud_frac": round(cloud_frac,2), "visibility_m": int(vis_m)}
        res.append(entry)
    # apply METAR bias for first hour if available
    if metar:
        try:
            # parse observation time from METAR if possible; assume it's issue hour if not
            obs = None
            # If METAR has Z timestamp, try to extract e.g. 280500Z
            m = re.search(r"(\d{6}Z)", metar.get("raw",""))
            if m:
                # we cannot robustly parse date without month; fallback use now
                obs = now
            else:
                obs = now
            # bias first hour
            if res:
                first = res[0]
                if metar.get("wind_speed_kt") is not None:
                    first["wind_speed_kt"] = round((first["wind_speed_kt"] + metar["wind_speed_kt"]) / 2.0,1)
                if metar.get("wind_dir") is not None and metar.get("wind_speed_kt") is not None:
                    u1,v1 = wind_vector(first["wind_speed_kt"], first["wind_dir_deg"])
                    u2,v2 = wind_vector(metar["wind_speed_kt"], metar["wind_dir"])
                    u = (u1+u2)/2; v = (v1+v2)/2
                    spd, ang = vector_to_wind(u,v)
                    first["wind_speed_kt"] = round(spd,1); first["wind_dir_deg"] = round(ang,1)
                if metar.get("visibility_m"):
                    first["visibility_m"] = int((first["visibility_m"] + metar["visibility_m"]) / 2)
                if metar.get("clouds"):
                    # if METAR shows low clouds, increase cloud_frac a bit
                    if any(c.startswith("BKN") or c.startswith("OVC") for c in metar.get("clouds",[])):
                        first["cloud_frac"] = max(first["cloud_frac"], 0.5)
        except Exception:
            pass
    # Note: BMKG json fusion not deeply implemented — BMKG point/ad4 could be used for corrections if structure known
    return res

# detect change events (simple heuristics)
def detect_events(hourly):
    events = []
    if not hourly: return events
    base = hourly[0]
    for p in hourly[1:]:
        ws_diff = abs(p["wind_speed_kt"] - base["wind_speed_kt"])
        wd_diff = abs(((p["wind_dir_deg"] - base["wind_dir_deg"] + 180) % 360) - 180)
        vis_drop = p["visibility_m"] < VISIBILITY_THRESHOLD_M and base["visibility_m"] >= VISIBILITY_THRESHOLD_M
        precip_onset = p["precip_mm"] >= PRECIP_MIN_MM and base["precip_mm"] < PRECIP_MIN_MM
        cloud_increase = p["cloud_frac"] >= CLOUD_FRACTION_THRESHOLD and base["cloud_frac"] < CLOUD_FRACTION_THRESHOLD
        if ws_diff >= WIND_SPEED_CHANGE_KT or wd_diff >= WIND_DIR_CHANGE_DEG:
            events.append({"type":"BECMG", "start": p["time"], "end": p["time"] + timedelta(hours=3), "sample": p,
                           "reason": f"Wind change {base['wind_speed_kt']}kt/{int(base['wind_dir_deg'])}° → {p['wind_speed_kt']}kt/{int(p['wind_dir_deg'])}°"})
            base = p
        elif vis_drop or precip_onset or cloud_increase:
            events.append({"type":"TEMPO", "start": p["time"], "end": p["time"] + timedelta(hours=3), "sample": p,
                           "reason": f"{'Vis drop' if vis_drop else ''}{' Precip onset' if precip_onset else ''}{' Cloud increase' if cloud_increase else ''}"})
            # don't update base
    # convert short abrupt changes into FM if within 1 hour and abrupt
    for ev in list(events):
        if (ev["end"] - ev["start"]).total_seconds() <= 3600 and ev["type"] == "BECMG":
            ev["type"] = "FM"
    return events

# format cloud group from fraction
def cloud_group_from_frac(frac):
    if frac >= 0.75: return "OVC020"
    if frac >= 0.4: return "BKN030"
    if frac >= 0.1: return "SCT050"
    return "SKC"

# format visibility (meters) into ICAO-readable (use meters)
def format_vis(vis_m):
    if vis_m is None: return "9999"
    return str(int(vis_m))

# build TAF (header + up to 3 period lines + event lines)
def build_taf(issue_time, hourly, events, validity_hours=VALIDITY_HOURS):
    if not hourly: return "TAF not available - no hourly points."
    now = issue_time.replace(minute=0, second=0, microsecond=0)
    end = now + timedelta(hours=validity_hours)
    header = f"TAF WARR {issue_time.strftime('%d%H%M')}Z {now.strftime('%d%H')}/{end.strftime('%d%H')}"
    # period splits: 0-6,6-12,12-validity
    p1_end = now + timedelta(hours=6)
    p2_end = now + timedelta(hours=12)
    p3_end = end
    def pick_point(s,e):
        cands = [p for p in hourly if s <= p["time"] < e]
        if not cands:
            # fallback nearest
            return hourly[0]
        return cands[len(cands)//2]
    pt1 = pick_point(now, p1_end)
    pt2 = pick_point(p1_end, p2_end)
    pt3 = pick_point(p2_end, p3_end)
    lines = [header]
    for (s,e,pt) in [(now,p1_end,pt1),(p1_end,p2_end,pt2),(p2_end,p3_end,pt3)]:
        wind = f"{deg_to_cardinal10(pt['wind_dir_deg'])}{int(round(pt['wind_speed_kt'])):02d}KT"
        vis = format_vis(pt.get("visibility_m",9999))
        wx = ""
        if pt["precip_mm"] >= PRECIP_MIN_MM:
            wx = "-RA" if pt["precip_mm"] < 5 else "RA"
        cloud = cloud_group_from_frac(pt["cloud_frac"])
        lines.append(f"{s.strftime('%d%H%M')}Z {s.strftime('%d%H')}/{e.strftime('%d%H')} {wind} {vis} {wx} {cloud}".replace("  "," ").strip())
    # Insert event lines (append, but limit total detail lines to 4)
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
    # keep header + up to 4 detail lines (3 period lines + max 1 event line or include event lines within 3-4)
    detail = lines[1:]  # 3 main period lines
    # try to place up to 1 event line replacing the least important period if exists
    if event_lines:
        # if only 3 detail lines allowed, append first event line as extra (making total 4)
        detail = detail[:3]  # ensure 3
        detail.append(event_lines[0])
    output = "\n".join([header] + detail[:4])
    return output

# -------------------------
# MAIN ACTION (button)
# -------------------------
if st.button("Generate TAFOR (Fuse BMKG/OpenMeteo/METAR)"):
    # fetch BMKG (point or adm4)
    st.info("Fetching BMKG point/ad4 ...")
    bmkg = fetch_bmkg()
    if bmkg is None:
        st.warning("BMKG unavailable (point & adm4). Proceeding with Open-Meteo + METAR only.")
    else:
        st.success(f"BMKG source: {bmkg.get('type','-')} (fetched)")

    # fetch Open-Meteo
    st.info("Fetching Open-Meteo ...")
    try:
        openj = fetch_openmeteo(LAT, LON, hours=hours)
        st.success("Open-Meteo OK")
    except Exception as e:
        st.error(f"Open-Meteo fetch failed: {e}")
        openj = None

    # fetch METAR from OGIMET
    st.info("Fetching METAR (OGIMET) ...")
    metar_txt = fetch_metar_ogimet()
    if metar_txt:
        st.success("METAR fetched from OGIMET")
        st.code(metar_txt)
        metar_parsed = parse_metar(metar_txt)
        st.write("METAR parsed:", metar_parsed)
    else:
        st.warning("METAR not available via OGIMET.")
        metar_parsed = None

    # build fused hourly
    if openj:
        hourly = build_fused_hours(openj, bmkg_json=(bmkg["json"] if bmkg else None), metar=metar_parsed, hours=hours)
        st.write(f"Built {len(hourly)} hourly fused points (next {hours}h).")
        # show a small DataFrame preview
        df_preview = pd.DataFrame([{"time":p["time"], "ws_kt":p["wind_speed_kt"], "wd_deg":p["wind_dir_deg"],
                                    "precip_mm":p["precip_mm"], "cloud_frac":p["cloud_frac"], "vis_m":p["visibility_m"]} for p in hourly])
        st.dataframe(df_preview.head(24))
    else:
        st.error("Open-Meteo missing — cannot build forecast.")
        st.stop()

    # detect events
    events = detect_events(hourly)
    if events:
        st.write("Detected change events:")
        for e in events:
            st.write(f"- {e['type']} {e['start']} → {e['reason']}")
    else:
        st.info("No significant change events detected.")

    # build TAF
    taf_text = build_taf(run_dt, hourly, events, validity_hours=hours)
    st.markdown("### ✅ GENERATED TAFOR (WARR)")
    st.code(taf_text, language="text")

    st.download_button("Download TAFOR (txt)", data=taf_text, file_name=f"TAFOR_WARR_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.txt", mime="text/plain")

    st.success("TAFOR generation complete. Tweak thresholds in the script if you want more/less sensitivity for BECMG/TEMPO/FM.")

st.markdown("---")
st.caption("Notes: This app uses Open-Meteo as baseline hourly grid, BMKG point/ad4 as fallback local info (if available) and OGIMET for METAR. Fusion here is heuristic (not formal data assimilation). Adjust thresholds for operational tuning.")
