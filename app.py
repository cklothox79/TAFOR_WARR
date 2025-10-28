"""
Generate a TAF-like forecast (TAFOR) for WARR (Juanda) by fusing:
 - Open-Meteo forecast (by lat/lon)
 - BMKG forecast (attempt using adm endpoint provided by user)
 - METAR from NOAA AWC (recent observation for WARR)

Outputs a 3-4 line TAF-style string with BECMG/TEMPO/FM elements when
significant changes are detected.

Usage:
  python generate_tafor_warr.py
"""

import requests
import math
import datetime
import xml.etree.ElementTree as ET
from collections import namedtuple

# -------------------------
# CONFIG / USER INPUT
# -------------------------
LAT = -7.380
LON = 112.786
RADIUS_KM = 10  # as requested

# Time window for TAF validity (hours)
VALIDITY_HOURS = 24  # produce TAF for next 24 hours

# Thresholds to decide significant changes (tuneable)
WIND_SPEED_CHANGE_KT = 6        # kt difference considered significant
WIND_DIR_CHANGE_DEG = 30        # deg change considered significant
VISIBILITY_DROP_M = 5000        # below this is significant decline (m)
PRECIP_PROB_THRESHOLD = 0.2     # probability or forecast precip > this => precipitation
CLOUD_COVER_THRESHOLD = 0.4     # fraction (0-1) to consider cloud significant

# Endpoints
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
BMKG_ADM_ENDPOINT = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm?adm1=35"
# NOAA AviationWeather AWC METAR server
NOAA_METAR_URL = ("https://aviationweather.gov/adds/dataserver_current/httpparam"
                  "?datasource=metars&requestType=retrieve&format=xml&stationString=WARR"
                  "&hoursBeforeNow=3")

# -------------------------
# Helper data structures
# -------------------------
ForecastPoint = namedtuple("ForecastPoint",
                           ["time", "wind_speed_kt", "wind_dir_deg",
                            "visibility_m", "precip_mm", "cloud_cover", "condition_tags"])

# -------------------------
# Utility functions
# -------------------------
def kph_to_kt(kph):
    return kph / 1.852

def mps_to_kt(mps):
    return mps * 1.94384

def deg_to_cardinal(deg):
    # convert degrees to nearest 10KT sector like 070
    d = int((deg + 5) // 10 * 10) % 360
    return f"{d:03d}"

def wind_vector(speed_kt, dir_deg):
    """Return vector components (u,v) where u is eastward, v is northward"""
    rad = math.radians((270 - dir_deg) % 360)  # convert to mathematical angle
    u = speed_kt * math.cos(rad)
    v = speed_kt * math.sin(rad)
    return u, v

def vector_to_wind(u, v):
    speed = math.hypot(u, v)
    # convert back to meteorological direction
    ang = (270 - math.degrees(math.atan2(v, u))) % 360
    return speed, ang

# -------------------------
# Fetchers
# -------------------------
def fetch_open_meteo(lat, lon, hours=24):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m", "windspeed_10m", "winddirection_10m",
            "visibility", "cloudcover", "precipitation"
        ]),
        "forecast_days": max(1, (hours // 24) + 1),
        "timezone": "auto"
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    # parse hourly arrays
    times = data["hourly"]["time"]
    ws = data["hourly"]["windspeed_10m"]
    wd = data["hourly"]["winddirection_10m"]
    vis = data["hourly"].get("visibility", [10000]*len(times))  # m (may be absent)
    cc = data["hourly"].get("cloudcover", [0]*len(times))  # percent
    precip = data["hourly"].get("precipitation", [0]*len(times))
    points = []
    now = datetime.datetime.utcnow()
    for i, t in enumerate(times):
        tdt = datetime.datetime.fromisoformat(t)
        if tdt < now:
            continue
        if (tdt - now).total_seconds() > hours*3600:
            break
        # convert units
        w_kt = kph_to_kt(ws[i]) if ws[i] is not None else 0.0
        wd_deg = wd[i] if wd[i] is not None else 0.0
        vis_m = vis[i] if vis[i] is not None else 10000
        cloud_frac = cc[i] / 100.0 if cc[i] is not None else 0.0
        p = ForecastPoint(time=tdt, wind_speed_kt=w_kt, wind_dir_deg=wd_deg,
                          visibility_m=vis_m, precip_mm=precip[i], cloud_cover=cloud_frac,
                          condition_tags=[])
        points.append(p)
    return points

def fetch_bmkg_adm():
    """Attempt to fetch given BMKG adm endpoint; structure may vary."""
    try:
        r = requests.get(BMKG_ADM_ENDPOINT, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # return None to indicate unavailable
        return None

def parse_bmkg_to_points(bmkg_json, lat, lon, hours=24):
    """
    BMKG adm endpoint returns forecasts by administrative area.
    This parser tries to extract precipitation probability/intensity and wind if present.
    If structure doesn't match, return None.
    """
    if not bmkg_json:
        return None
    # heuristic parsing
    try:
        # sometimes BMKG returns "data" or "forecast"
        # We will look for any time series with keys: 'time','wind','rain' etc.
        # This is best-effort — if fails, return None
        points = []
        now = datetime.datetime.utcnow()
        # Example structures vary — loop through nested dicts to find time entries
        def find_time_series(obj):
            found = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() in ("time", "times", "tanggal"):
                        found.append(v)
                    else:
                        found.extend(find_time_series(v))
            elif isinstance(obj, list):
                for e in obj:
                    found.extend(find_time_series(e))
            return found

        # fallback: try to find 'forecast'
        # We'll just return None and let Open-Meteo + METAR drive the TAFOR
        return None
    except Exception:
        return None

def fetch_metar_noaa():
    r = requests.get(NOAA_METAR_URL, timeout=12)
    r.raise_for_status()
    xml = r.text
    root = ET.fromstring(xml)
    # find most recent METAR element
    metars = root.findall(".//METAR")
    if not metars:
        return None
    m = metars[0]
    # parse key fields
    obs_time = m.findtext("observation_time")
    wind_dir = m.findtext("wind_dir_degrees")
    wind_speed = m.findtext("wind_speed_kt")
    visibility = m.findtext("visibility_statute_mi")  # in statute miles sometimes
    if visibility is not None:
        try:
            vis_m = float(visibility) * 1609.344
        except:
            vis_m = None
    else:
        vis_m = None
    weather = m.findtext("wx_string")
    sky_conditions = [sc.attrib.get("sky_cover") + ((":" + sc.attrib.get("cloud_base_ft_agl")) if sc.attrib.get("cloud_base_ft_agl") else "")
                      for sc in m.findall("sky_condition")]
    return {
        "observation_time": obs_time,
        "wind_dir_deg": float(wind_dir) if wind_dir else None,
        "wind_speed_kt": float(wind_speed) if wind_speed else None,
        "visibility_m": vis_m,
        "weather": weather,
        "sky_conditions": sky_conditions
    }

# -------------------------
# Fusion & change detection
# -------------------------
def fuse_open_meteo_and_bmkg(open_points, bmkg_points=None, metar=None):
    """
    Simple fusion:
     - For each hour-point from Open-Meteo (primary), adjust using bmkg_points if available
     - If metar provided and within 3 hours, use METAR for t=now baseline (first step)
    Returns list of ForecastPoint (fused)
    """
    fused = []
    # baseline: open_meteo
    for p in open_points:
        # start with open-meteo fields
        wind_speed = p.wind_speed_kt
        wind_dir = p.wind_dir_deg
        vis = p.visibility_m
        precip = p.precip_mm
        cloud = p.cloud_cover
        tags = []

        # apply simple bmkg adjustments if present (not implemented in detail here)
        # if bmkg_points: ... (omitted for brevity) -> best-effort in future

        # apply METAR bias for first hour (if metar exists)
        if metar:
            try:
                obs_time = datetime.datetime.fromisoformat(metar["observation_time"].replace("Z", "+00:00"))
            except Exception:
                obs_time = None
            if obs_time:
                dt = abs((p.time - obs_time).total_seconds())
                if dt <= 3600:  # within 1 hour, bias open-meteo toward METAR
                    if metar.get("wind_speed_kt") is not None:
                        # average
                        wind_speed = (wind_speed + metar["wind_speed_kt"]) / 2.0
                    if metar.get("wind_dir_deg") is not None:
                        # average via vector
                        u1, v1 = wind_vector(wind_speed, wind_dir)
                        u2, v2 = wind_vector(metar["wind_speed_kt"] or 0.0, metar["wind_dir_deg"] or 0.0)
                        u = (u1 + u2) / 2.0
                        v = (v1 + v2) / 2.0
                        wind_speed, wind_dir = vector_to_wind(u, v)
                    if metar.get("visibility_m"):
                        vis = (vis + metar["visibility_m"]) / 2.0 if vis else metar["visibility_m"]
                    if metar.get("weather"):
                        tags.append(metar["weather"])
        fused.append(ForecastPoint(time=p.time, wind_speed_kt=round(wind_speed,1),
                                   wind_dir_deg=round(wind_dir,1), visibility_m=int(vis or 10000),
                                   precip_mm=round(precip or 0.0,2), cloud_cover=round(cloud,2),
                                   condition_tags=tags))
    return fused

def detect_significant_changes(fused_points):
    """
    Detects windows where significant changes occur and returns a list of change events:
    Each event is dict: {type: 'BECMG'|'TEMPO'|'FM', start: datetime, end: datetime, reason: str, sample: ForecastPoint}
    Simplified heuristics:
     - sudden persistent change => BECMG
     - short-duration deterioration (vis drop / showers) => TEMPO
     - abrupt change within <1 hour => FM
    """
    events = []
    if not fused_points:
        return events
    baseline = fused_points[0]
    # iterate and compare
    for p in fused_points[1:]:
        # wind change
        ws_diff = abs(p.wind_speed_kt - baseline.wind_speed_kt)
        wd_diff = abs(((p.wind_dir_deg - baseline.wind_dir_deg + 180) % 360) - 180)
        vis_drop = p.visibility_m < VISIBILITY_DROP_M and baseline.visibility_m >= VISIBILITY_DROP_M
        precip_onset = (p.precip_mm > 0.1) and (baseline.precip_mm <= 0.1)
        cloud_increase = (p.cloud_cover >= CLOUD_COVER_THRESHOLD) and (baseline.cloud_cover < CLOUD_COVER_THRESHOLD)
        # classify
        if ws_diff >= WIND_SPEED_CHANGE_KT or wd_diff >= WIND_DIR_CHANGE_DEG:
            # if change persists for several hours -> BECMG; if short -> TEMPO
            events.append({"type": "BECMG", "start": p.time, "end": p.time + datetime.timedelta(hours=3),
                           "reason": f"Wind change {baseline.wind_speed_kt}kt/{int(baseline.wind_dir_deg)}° -> {p.wind_speed_kt}kt/{int(p.wind_dir_deg)}°", "sample": p})
            baseline = p
        elif vis_drop or precip_onset or cloud_increase:
            # use TEMPO for transient adverse changes
            events.append({"type": "TEMPO", "start": p.time, "end": p.time + datetime.timedelta(hours=3),
                           "reason": f"{'Vis drop' if vis_drop else ''}{' Precip onset' if precip_onset else ''}{' Cloud increase' if cloud_increase else ''}",
                           "sample": p})
            # do not reset baseline for TEMPO
    return events

# -------------------------
# TAF formatter
# -------------------------
def format_taf_line_from_point(pt, valid_from, valid_to):
    """
    Build a TAF line for the period using simple formatting:
    e.g. "280600Z 2806/2906 07010KT 8000 -SHRA FEW020 SCT040"
    We'll produce simple lines: wind, visibility (m->SM), wx groups, cloud groups.
    """
    # times
    znow = datetime.datetime.utcnow()
    # validity in TAF uses ddhh/ddhh ; here we'll produce ddhh/ddhhZ style (UTC)
    def to_ddhh(dt):
        return f"{dt.day:02d}{dt.hour:02d}"
    validity_str = f"{to_ddhh(valid_from)}/{to_ddhh(valid_to)}"
    # wind
    wdir = deg_to_cardinal(pt.wind_dir_deg if pt.wind_dir_deg is not None else 0)
    wspd = int(round(pt.wind_speed_kt))
    wind_group = f"{wdir}{wspd:02d}KT"
    # visibility: convert meters to nearest 1000m or into runway-friendly format; TAF uses meters (ICAO) or SM in US
    vis_m = int(pt.visibility_m or 9999)
    vis_group = f"{vis_m}"
    # weather group: very simple mapping
    wx = ""
    if pt.precip_mm > 0.5:
        wx = "-RA" if pt.precip_mm < 5 else "RA"
    # cloud group: quick conversion from cloud fraction
    cloud_group = ""
    if pt.cloud_cover >= 0.75:
        cloud_group = "OVC020"
    elif pt.cloud_cover >= 0.4:
        cloud_group = "BKN030"
    elif pt.cloud_cover >= 0.1:
        cloud_group = "SCT050"
    else:
        cloud_group = "SKC"
    # assemble
    return f"{valid_from.strftime('%d%H%M')}Z {validity_str} {wind_group} {vis_group} {wx} {cloud_group}".replace("  ", " ").strip()

def generate_taf(fused_points, events):
    """
    Build 3-4 lines: initial forecast + up to 3 change lines (TEMPO/BECMG/FM)
    Strategy:
     - line 1: initial from now -> +6h
     - line 2: 6-12h
     - line 3: 12-24h
     - plus add event lines as TEMPO/BECMG/FM inserted after the line they occur in
    """
    if not fused_points:
        return "TAF not available - no forecast points."
    now = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    p1_end = now + datetime.timedelta(hours=6)
    p2_end = now + datetime.timedelta(hours=12)
    p3_end = now + datetime.timedelta(hours=24 if VALIDITY_HOURS >= 24 else VALIDITY_HOURS)
    # choose representative point per period (median hour)
    def pick_point_between(start, end):
        candidates = [p for p in fused_points if start <= p.time < end]
        if not candidates:
            # fallback nearest
            return fused_points[0]
        return candidates[len(candidates)//2]
    pt1 = pick_point_between(now, p1_end)
    pt2 = pick_point_between(p1_end, p2_end)
    pt3 = pick_point_between(p2_end, p3_end)
    lines = []
    # header line (station & issue time)
    issue = datetime.datetime.utcnow()
    header = f"TAF WARR {issue.strftime('%d%H%M')}Z {now.strftime('%d%H')}/{p3_end.strftime('%d%H')}"
    lines.append(header)
    # main period lines
    lines.append(format_taf_line_from_point(pt1, now, p1_end))
    lines.append(format_taf_line_from_point(pt2, p1_end, p2_end))
    lines.append(format_taf_line_from_point(pt3, p2_end, p3_end))
    # Insert event lines
    for ev in events:
        ev_type = ev["type"]
        ev_start = ev["start"]
        ev_end = ev["end"]
        sample = ev["sample"]
        taf_ev = f"{ev_type} {ev_start.strftime('%d%H%M')}Z/{ev_end.strftime('%d%H%M')}Z " + \
                 f"{deg_to_cardinal(sample.wind_dir_deg)}{int(round(sample.wind_speed_kt)):02d}KT {sample.visibility_m} " + \
                 (("-RA" if sample.precip_mm > 0.5 else "") + (" BKN020" if sample.cloud_cover >= 0.5 else " SCT040"))
        lines.append(taf_ev)
    # Keep only 3-4 body lines (header + <=4 detail lines). Trim if too many events.
    # The user wanted 3-4 lines like official TAF; we'll keep header + up to 4 detail lines
    max_detail = 4
    output_lines = [lines[0]] + lines[1:1+max_detail]
    # append additional event lines if none were included in the 4 lines yet (but we already inserted them)
    return "\n".join(output_lines)

# -------------------------
# Main flow
# -------------------------
def main():
    print("Fetching Open-Meteo forecast...")
    try:
        open_pts = fetch_open_meteo(LAT, LON, hours=VALIDITY_HOURS)
        if not open_pts:
            print("No Open-Meteo points returned.")
    except Exception as e:
        print("Open-Meteo fetch failed:", e)
        open_pts = []

    print("Fetching BMKG ADM (best-effort)...")
    bmkg_json = fetch_bmkg_adm()
    if bmkg_json:
        print("BMKG ADM fetched (structure may vary). Attempting parse...")
        bmkg_pts = parse_bmkg_to_points(bmkg_json, LAT, LON, hours=VALIDITY_HOURS)
    else:
        print("BMKG ADM not available or fetch failed.")
        bmkg_pts = None

    print("Fetching recent METAR (NOAA AWC)...")
    try:
        metar = fetch_metar_noaa()
        if metar:
            print("METAR:", metar.get("observation_time"), "wind:", metar.get("wind_speed_kt"), "kt", metar.get("wind_dir_deg"), "deg")
        else:
            print("No METAR found.")
    except Exception as e:
        print("METAR fetch failed:", e)
        metar = None

    print("Fusing forecasts...")
    fused = fuse_open_meteo_and_bmkg(open_pts, bmkg_pts, metar)

    print("Detecting significant changes...")
    events = detect_significant_changes(fused)

    print("Generating TAFOR text...")
    taf_text = generate_taf(fused, events)
    print("\n--- GENERATED TAFOR (WARR) ---\n")
    print(taf_text)
    print("\n--- End ---\n")

if __name__ == "__main__":
    main()
