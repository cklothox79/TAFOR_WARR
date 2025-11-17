# app.py
"""
TAFOR Fusion Pro — Operational v2.9 (WARR / Sedati Gede)
- Minor updates:
  * Enforce per-line TAF (Format A)
  * Sort BECMG / TEMPO lines ascending by start time
  * Short RMK: "RMK AUTO FUSION (BMKG, ECMWF, ICON & GFS)"
- Keeps: auto-weighting, CB/TS logic, dry-run, log viewer, download TAF (.txt) & logs.
- IMPORTANT: ALWAYS validate final TAFOR manually per ICAO Annex 3 & Perka BMKG.
"""

import os
import io
import json
import logging
from datetime import datetime, timedelta, time as dtime
import math
import re

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Basic config
# -----------------------
st.set_page_config(page_title="TAFOR Fusion Pro — Operational v2.9 (WARR)", layout="centered")
st.title("🛫 TAFOR Fusion Pro — Operational (WARR / Sedati Gede) — v2.9")
st.caption("Location: Sedati Gede (ADM4=35.15.17.2011). Fusi BMKG + Open-Meteo + METAR realtime")

# folders
os.makedirs("output", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# constants
LAT, LON = -7.379, 112.787
ICAO_STATION = "WARR"            # TAF header station
TAF_FILENAME_STATION = "WAII"    # filename station token as requested
ADM4 = "35.15.17.2011"
REFRESH_TTL = 600  # seconds
# default base weights (percent)
BASE_WEIGHTS_PERC = {"bmkg": 45.0, "ecmwf": 45.0, "icon": 5.0, "gfs": 5.0}
# fallback user-defaults (fractions)
DEFAULT_WEIGHTS = {"bmkg": 0.45, "ecmwf": 0.25, "icon": 0.15, "gfs": 0.15}

# HTTP session
session = requests.Session()
session.headers.update({"User-Agent": "TAFOR-Fusion-Pro/2.9"})

# -----------------------
# UI Inputs
# -----------------------
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", datetime.utcnow().date())
with col2:
    jam_penting = [0, 3, 6, 9, 12, 15, 18, 21]
    default_hour = min(jam_penting, key=lambda j: abs(j - datetime.utcnow().hour))
    issue_time = st.selectbox("🕓 Issue time (UTC)", jam_penting, index=jam_penting.index(default_hour))
with col3:
    validity = st.number_input("🕐 Validity (hours)", min_value=6, max_value=36, value=24, step=6)

st.markdown("### ⚙️ Ensemble weights (BMKG priority)")
wcols = st.columns(4)
bmkg_w = wcols[0].number_input("BMKG", 0.0, 1.0, value=DEFAULT_WEIGHTS["bmkg"], step=0.05, key="bmkg_w")
ecmwf_w = wcols[1].number_input("ECMWF", 0.0, 1.0, value=DEFAULT_WEIGHTS["ecmwf"], step=0.05, key="ecmwf_w")
icon_w = wcols[2].number_input("ICON", 0.0, 1.0, value=DEFAULT_WEIGHTS["icon"], step=0.05, key="icon_w")
gfs_w = wcols[3].number_input("GFS", 0.0, 1.0, value=DEFAULT_WEIGHTS["gfs"], step=0.05, key="gfs_w")
_sumw = bmkg_w + ecmwf_w + icon_w + gfs_w
if _sumw <= 0:
    _sumw = 1.0
manual_weights = {"bmkg": bmkg_w / _sumw, "ecmwf": ecmwf_w / _sumw, "icon": icon_w / _sumw, "gfs": gfs_w / _sumw}
st.caption(f"Normalized manual weights: {manual_weights}")

# Auto-weight toggle
auto_weight = st.checkbox("🔁 Auto-weight ensembles (use default priorities and redistribute if some unavailable)", value=True)

st.divider()

# Extra controls
col_a, col_b = st.columns([1,2])
with col_a:
    dry_run = st.checkbox("🔎 Dry-run (don't save files)", value=False)
with col_b:
    validate_strict = st.checkbox("✅ Stricter TAF validation (ICAO/Perka-like rules)", value=True)

st.markdown("## Actions")
st.write("Tekan tombol untuk generate TAFOR (fusion). Pastikan koneksi ke API tersedia untuk hasil real-time.")

# -----------------------
# Helpers
# -----------------------
def wind_to_uv(speed, deg):
    if speed is None or deg is None or (isinstance(speed, float) and np.isnan(speed)) or (isinstance(deg, float) and np.isnan(deg)):
        return np.nan, np.nan
    theta = math.radians((270.0 - deg) % 360.0)
    return speed * math.cos(theta), speed * math.sin(theta)

def uv_to_wind(u, v):
    try:
        spd = math.sqrt(u * u + v * v)
        theta = math.degrees(math.atan2(v, u))
        deg = (270.0 - theta) % 360.0
        return spd, deg
    except Exception:
        return np.nan, np.nan

def safe_to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def safe_int(x, default=0):
    try:
        return int(round(float(x)))
    except Exception:
        return default

def weighted_mean(vals, ws):
    if not vals or not ws:
        return np.nan
    arr = np.array(vals, dtype=float)
    w = np.array(ws[:len(arr)], dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return np.nan
    w_mask = w[mask]
    if w_mask.sum() == 0:
        return float(np.nanmean(arr[mask]))
    return float((arr[mask] * w_mask).sum() / w_mask.sum())

def fmt_bytes(n):
    try:
        n = float(n)
    except Exception:
        return "0 B"
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def sanitize_taf_text_lines(lines):
    """
    Accepts a list of TAF lines and returns a clean string with one entry per line.
    Preserves line breaks, strips extra spaces per line.
    """
    clean = []
    for l in lines:
        if not isinstance(l, str):
            continue
        s = re.sub(r'\s+', ' ', l).strip()
        if s:
            clean.append(s)
    return "\n".join(clean)

# -----------------------
# Auto-weighting utility
# -----------------------
def compute_auto_weights(availability):
    """
    availability: dict with keys 'bmkg','ecmwf','icon','gfs' boolean True/False
    base percentages defined in BASE_WEIGHTS_PERC
    returns dict of normalized fractional weights summing to 1.0
    """
    active = {k: v for k, v in BASE_WEIGHTS_PERC.items() if availability.get(k, False)}
    if not active:
        return {"bmkg": 0.0, "ecmwf": 0.0, "icon": 0.0, "gfs": 0.0}
    total_active = sum(active.values())
    normalized = {k: (v / total_active) for k, v in active.items()}
    final = {
        "bmkg": normalized.get("bmkg", 0.0),
        "ecmwf": normalized.get("ecmwf", 0.0),
        "icon": normalized.get("icon", 0.0),
        "gfs": normalized.get("gfs", 0.0)
    }
    return final

# -----------------------
# TS / cloud / weather helpers
# -----------------------
def detect_cb_from_model_flag(flag):
    if not flag:
        return False
    try:
        s = str(flag).upper()
        if "CB" in s or "CUMUL" in s or "TCU" in s:
            return True
    except Exception:
        pass
    return False

def normalize_weather(wx):
    if not wx:
        return []
    s = str(wx).upper()
    s = s.replace(",", " ")
    patterns = ["+TSRA", "-TSRA", "TSRA", "TSGR", "+TS", "-TS", "VCTS", "TS",
                "+RA", "-RA", "SHRA", "RA", "SH", "DZ", "BR", "FG", "BC", "DR", "BL"]
    tokens = []
    words = re.split(r'\s+', s)
    for w in words:
        for p in patterns:
            if p in w and p not in tokens:
                tokens.append(p)
    for p in patterns:
        if p in s and p not in tokens:
            tokens.append(p)
    return tokens

def parse_metar_for_ts(metar_text):
    if not metar_text:
        return False
    return bool(re.search(r'(^|\s)(V?TS|TSRA|TSGR|\+TS|-TS)(\s|$)', metar_text.upper()))

def should_include_ts(tokens, cb_present, metar_has_ts=False):
    ts_codes = ["+TSRA", "-TSRA", "TSRA", "TSGR", "+TS", "-TS", "TS", "VCTS"]
    has_ts_token = any(t in ts_codes for t in tokens)
    if (cb_present and has_ts_token) or (metar_has_ts and has_ts_token):
        return True
    return False

def infer_precip_from_probs_and_conditions(prob_row, cc, rh):
    pop = 0.0
    if prob_row is not None and "PoP_precip" in prob_row:
        try:
            pop = float(prob_row.get("PoP_precip", 0.0))
        except Exception:
            pop = 0.0
    try:
        ccv = float(cc) if cc is not None and not pd.isna(cc) else 0.0
        rhv = float(rh) if rh is not None and not pd.isna(rh) else 0.0
    except Exception:
        ccv = 0.0; rhv = 0.0
    if pop >= 0.7 or (ccv >= 80 and rhv >= 85):
        return True
    return False

# -----------------------
# Fetchers (cached)
# -----------------------
@st.cache_data(ttl=REFRESH_TTL)
def fetch_bmkg(adm4=ADM4, local_fallback="JSON_BMKG.txt"):
    url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm"
    params = {"adm1": "35", "adm2": "35.15", "adm3": "35.15.17", "adm4": adm4}
    try:
        r = session.get(url, params=params, timeout=15, verify=True)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning("BMKG API failed: %s", e)
        if os.path.exists(local_fallback):
            with open(local_fallback, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

@st.cache_data(ttl=REFRESH_TTL)
def fetch_openmeteo(model):
    base = f"https://api.open-meteo.com/v1/{model}"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,windspeed_10m,winddirection_10m,visibility",
        "forecast_days": 2,
        "timezone": "UTC"
    }
    try:
        r = session.get(base, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning("Open-Meteo %s failed: %s", model, e)
        return None

@st.cache_data(ttl=REFRESH_TTL)
def fetch_metar_ogimet(station=ICAO_STATION):
    try:
        og = session.get(f"https://ogimet.com/display_metars2.php?lang=en&icao={station}", timeout=10)
        if og.ok:
            text = og.text
            lines = [ln.strip() for ln in text.splitlines() if station in ln]
            if lines:
                last = lines[-1]
                last = re.sub("<[^<]+?>", "", last)
                idx = last.find(station)
                if idx >= 0:
                    return " ".join(last[idx:].split())
    except Exception as e:
        logging.warning("OGIMET failed: %s", e)
    try:
        r = session.get(f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station}.TXT", timeout=10)
        if r.ok:
            lines = r.text.strip().splitlines()
            return lines[-1].strip()
    except Exception as e:
        logging.warning("NOAA METAR fallback failed: %s", e)
    return None

# -----------------------
# Parsers & converters
# -----------------------
def bmkg_cuaca_to_df(cuaca):
    if not cuaca:
        return pd.DataFrame()
    records = []
    if isinstance(cuaca, dict):
        records = [cuaca]
    elif isinstance(cuaca, list):
        for item in cuaca:
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        records.append(sub)
    times, tvals, rhvals, tccvals, wsvals, wdvals, visvals, cloudtype = [], [], [], [], [], [], [], []
    for rec in records:
        dt = rec.get("datetime") or rec.get("time") or rec.get("jamCuaca") or rec.get("date") or rec.get("valid_time")
        t0 = None
        if isinstance(dt, str):
            try:
                t0 = pd.to_datetime(dt.replace("Z", "+00:00"), utc=True)
            except Exception:
                try:
                    t0 = pd.to_datetime(dt)
                except Exception:
                    t0 = None
        elif isinstance(dt, (int, float)):
            try:
                t0 = pd.to_datetime(dt, unit="s", utc=True)
            except Exception:
                t0 = None
        if t0 is None:
            continue
        try:
            times.append(t0.tz_convert("UTC").tz_localize(None))
        except Exception:
            try:
                times.append(t0.tz_localize(None))
            except Exception:
                times.append(pd.to_datetime(t0))
        tvals.append(safe_to_float(rec.get("t") or rec.get("temp") or rec.get("temperature")))
        rhvals.append(safe_to_float(rec.get("hu") or rec.get("rh") or rec.get("humidity")))
        tccvals.append(safe_to_float(rec.get("tcc") or rec.get("cloud") or rec.get("cloud_cover")))
        wsvals.append(safe_to_float(rec.get("ws") or rec.get("wind_speed")))
        wdvals.append(safe_to_float(rec.get("wd_deg") or rec.get("wind_dir") or rec.get("wind_direction")))
        visvals.append(rec.get("vs_text") or rec.get("visibility") or np.nan)
        cloudtype.append(rec.get("cloud_type") or rec.get("ew") or rec.get("cb") or None)
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame({
        "time": times,
        "T_BMKG": tvals,
        "RH_BMKG": rhvals,
        "CC_BMKG": tccvals,
        "WS_BMKG": wsvals,
        "WD_BMKG": wdvals,
        "VIS_BMKG": visvals,
        "CLOUDTYPE_BMKG": cloudtype
    })
    df = df.sort_values("time").reset_index(drop=True)
    return df

def openmeteo_json_to_df(j, tag):
    if not j or "hourly" not in j:
        return None
    h = j["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"])})
    df[f"T_{tag}"] = h.get("temperature_2m")
    df[f"RH_{tag}"] = h.get("relative_humidity_2m")
    df[f"CC_{tag}"] = h.get("cloud_cover")
    df[f"WS_{tag}"] = h.get("windspeed_10m")
    df[f"WD_{tag}"] = h.get("winddirection_10m")
    df[f"VIS_{tag}"] = h.get("visibility", [np.nan] * len(df))
    return df

# -----------------------
# Merge, fuse & validate rules
# -----------------------
def align_hourly(dfs):
    normalized = []
    for d in dfs:
        if d is None:
            continue
        if "time" in d.columns:
            d["time"] = pd.to_datetime(d["time"], errors="coerce")
            d = d.dropna(subset=["time"])
            try:
                d["time"] = d["time"].dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception:
                try:
                    d["time"] = d["time"].dt.tz_localize(None)
                except Exception:
                    pass
            normalized.append(d)
    if not normalized:
        return None
    base = normalized[0][["time"]].copy()
    for d in normalized[1:]:
        base = pd.merge(base, d, on="time", how="outer")
    base = base.sort_values("time").reset_index(drop=True)
    return base

def fuse_ensemble(df_merged, weights, hours=24):
    rows = []
    now = pd.to_datetime(datetime.utcnow()).floor("H")
    df_merged = df_merged.sort_values("time").reset_index(drop=True)
    df_merged = df_merged[df_merged["time"] >= now].head(hours)
    for _, r in df_merged.iterrows():
        T_vals, RH_vals, CC_vals, VIS_vals = [], [], [], []
        u_vals, v_vals = [], []
        w_list = []
        cb_flag = False
        # BMKG
        if weights.get("bmkg", 0) > 0:
            t = r.get("T_BMKG"); rh = r.get("RH_BMKG"); cc = r.get("CC_BMKG")
            ws = r.get("WS_BMKG"); wd = r.get("WD_BMKG"); vis = r.get("VIS_BMKG")
            if not pd.isna(t): T_vals.append(t)
            if not pd.isna(rh): RH_vals.append(rh)
            if not pd.isna(cc): CC_vals.append(cc)
            try:
                VIS_vals.append(float(vis))
            except Exception:
                pass
            if not pd.isna(ws) and not pd.isna(wd):
                u, v = wind_to_uv(ws, wd); u_vals.append(u); v_vals.append(v)
            if detect_cb_from_model_flag(r.get("CLOUDTYPE_BMKG")):
                cb_flag = True
            w_list.append(weights["bmkg"])
        # other models
        for model in ["ecmwf", "icon", "gfs"]:
            tag = model.upper()
            wt = weights.get(model, 0)
            if wt <= 0:
                continue
            t = r.get(f"T_{tag}"); rh = r.get(f"RH_{tag}"); cc = r.get(f"CC_{tag}")
            ws = r.get(f"WS_{tag}"); wd = r.get(f"WD_{tag}"); vis = r.get(f"VIS_{tag}")
            if not pd.isna(t): T_vals.append(t)
            if not pd.isna(rh): RH_vals.append(rh)
            if not pd.isna(cc): CC_vals.append(cc)
            try:
                VIS_vals.append(float(vis))
            except Exception:
                pass
            if not pd.isna(ws) and not pd.isna(wd):
                u, v = wind_to_uv(ws, wd); u_vals.append(u); v_vals.append(v)
            w_list.append(wt)
        if not w_list:
            continue
        T_f = weighted_mean(T_vals, w_list)
        RH_f = weighted_mean(RH_vals, w_list)
        CC_f = weighted_mean(CC_vals, w_list)
        VIS_f = weighted_mean(VIS_vals, w_list)
        U_f = weighted_mean(u_vals, w_list) if u_vals else np.nan
        V_f = weighted_mean(v_vals, w_list) if v_vals else np.nan
        WS_f, WD_f = uv_to_wind(U_f, V_f)
        rows.append({
            "time": r["time"], "T": T_f, "RH": RH_f, "CC": CC_f, "VIS": VIS_f, "WS": WS_f, "WD": WD_f, "CB": cb_flag
        })
    return pd.DataFrame(rows)

# -----------------------
# Probabilities & flags
# -----------------------
def compute_probabilities(df_merged, models_list=["GFS", "ECMWF", "ICON", "BMKG"]):
    probs = []
    for _, r in df_merged.iterrows():
        votes = 0
        nm = 0
        temps = []
        for src in models_list:
            nm += 1
            t = r.get(f"T_{src}") if src != "BMKG" else r.get("T_BMKG")
            rh = r.get(f"RH_{src}") if src != "BMKG" else r.get("RH_BMKG")
            cc = r.get(f"CC_{src}") if src != "BMKG" else r.get("CC_BMKG")
            if t is not None:
                temps.append(safe_to_float(t))
            try:
                if (safe_to_float(cc) >= 80) and (safe_to_float(rh) >= 85):
                    votes += 1
            except Exception:
                pass
        prob = votes / nm if nm > 0 else 0.0
        spread = float(np.nanstd([x for x in temps if not pd.isna(x)])) if temps else np.nan
        probs.append({"time": r["time"], "PoP_precip": prob, "T_spread": spread})
    return pd.DataFrame(probs)

# -----------------------
# TAF building with new TS logic (PER-LINE output, sorted)
# -----------------------
def tcc_to_cloud_label(cc):
    if pd.isna(cc):
        return "FEW020"
    try:
        c = float(cc)
    except Exception:
        return "FEW020"
    if c < 25: return "FEW020"
    elif c < 50: return "SCT025"
    elif c < 85: return "BKN030"
    else: return "OVC030"

def build_taf_from_fused(df_fused, df_merged_for_flags, metar, issue_dt, validity, df_probs=None):
    """
    Returns (taf_lines_list, signif_times_list, taf_text_str)
    taf_lines_list is a list where each element is one TAF line (no combined changes on same line).
    BECMG / TEMPO lines will be sorted by start time (ascending).
    """
    taf_lines = []
    issue_header = issue_dt.strftime("%d%H%MZ")
    valid_to = issue_dt + timedelta(hours=validity)
    valid_period = f"{issue_dt.strftime('%d%H')}/{valid_to.strftime('%d%H')}"
    header = f"TAF {ICAO_STATION} {issue_header} {valid_period}"
    taf_lines.append(header)

    if df_fused is None or df_fused.empty:
        taf_lines.append("00000KT 9999 FEW020")
        taf_lines.append("NOSIG")
        taf_lines.append("RMK AUTO FUSION (BMKG, ECMWF, ICON & GFS)")
        taf_text = sanitize_taf_text_lines(taf_lines)
        return taf_lines, [], taf_text

    # base state (initial forecast line)
    first = df_fused.iloc[0]
    wd0 = safe_int(first.WD, 90)
    ws0 = safe_int(first.WS, 5)
    vis0 = safe_int(first.VIS, 9999)
    cloud0 = tcc_to_cloud_label(first.CC)
    taf_lines.append(f"{wd0:03d}{ws0:02d}KT {vis0:04d} {cloud0}")

    WIND_ANGLE_THRESHOLD = 60
    WIND_SPEED_DELTA = 10  # kt
    WIND_SIGNIFICANT_SPEED = 25  # kt

    becmg_entries = []  # will store tuples (sort_key_int, line)
    tempo_entries = []
    signif_times = []

    metar_has_ts = parse_metar_for_ts(metar)

    for i in range(1, len(df_fused)):
        prev = df_fused.iloc[i - 1]
        curr = df_fused.iloc[i]
        tstart = prev["time"].strftime("%d%H")
        tend = curr["time"].strftime("%d%H")

        # compute integer sort key (ddHH -> ddHH as int)
        try:
            sort_key = int(tstart)
        except Exception:
            # fallback to timestamp
            sort_key = int(curr["time"].strftime("%Y%m%d%H"))

        # wind checks
        wd_prev = safe_to_float(prev.WD) if not pd.isna(prev.WD) else np.nan
        wd_curr = safe_to_float(curr.WD) if not pd.isna(curr.WD) else np.nan
        ws_prev = safe_to_float(prev.WS) if not pd.isna(prev.WS) else np.nan
        ws_curr = safe_to_float(curr.WS) if not pd.isna(curr.WS) else np.nan

        wd_diff = np.nan
        if not pd.isna(wd_prev) and not pd.isna(wd_curr):
            wd_diff = abs((wd_curr - wd_prev + 180) % 360 - 180)

        ws_diff = np.nan
        if not pd.isna(ws_prev) and not pd.isna(ws_curr):
            ws_diff = abs(ws_curr - ws_prev)

        sig_wind = False
        if (not pd.isna(wd_diff) and not pd.isna(ws_diff)):
            if (wd_diff >= WIND_ANGLE_THRESHOLD and ws_diff >= WIND_SPEED_DELTA):
                sig_wind = True
        if (not pd.isna(ws_curr)) and (ws_curr >= WIND_SIGNIFICANT_SPEED):
            sig_wind = True

        # cloud significant only if CB appears/disappears
        prev_cb = bool(prev.get("CB", False))
        curr_cb = bool(curr.get("CB", False))
        sig_cloud = False
        if (not prev_cb) and curr_cb:
            sig_cloud = True
        elif prev_cb and (not curr_cb):
            sig_cloud = True

        # precipitation detection
        prob_row = None
        if df_probs is not None and curr["time"] in list(df_probs["time"]):
            prob_row = df_probs[df_probs["time"] == curr["time"]].iloc[0].to_dict()
        precip_flag = infer_precip_from_probs_and_conditions(prob_row, curr.CC, curr.RH)

        # tokens from metar (prioritize metar for TS tokens)
        metar_tokens = normalize_weather(metar) if metar else []
        include_ts = should_include_ts(metar_tokens, curr_cb, metar_has_ts=metar_has_ts)

        # Build per-line statements (each entry is own line), but store with sort key
        if sig_wind:
            becmg_line = f"BECMG {tstart}/{tend} {safe_int(curr.WD):03d}{safe_int(curr.WS):02d}KT {safe_int(curr.VIS or 9999):04d} {tcc_to_cloud_label(curr.CC)}"
            becmg_entries.append((sort_key, becmg_line))
            signif_times.append(curr["time"])

        if sig_cloud and curr_cb:
            becmg_line = f"BECMG {tstart}/{tend} {safe_int(curr.WD):03d}{safe_int(curr.WS):02d}KT {safe_int(curr.VIS or 9999):04d} CB"
            becmg_entries.append((sort_key, becmg_line))
            signif_times.append(curr["time"])

        if precip_flag:
            if include_ts:
                tempo_line = f"TEMPO {tstart}/{tend} 4000 +TSRA SCT020CB"
            else:
                tempo_line = f"TEMPO {tstart}/{tend} 4000 -RA"
            tempo_entries.append((sort_key, tempo_line))
            signif_times.append(curr["time"])

    # sort entries by start time ascending, then append each as its own line
    if becmg_entries:
        becmg_entries_sorted = sorted(becmg_entries, key=lambda x: x[0])
        for _, line in becmg_entries_sorted:
            taf_lines.append(line)
    if tempo_entries:
        tempo_entries_sorted = sorted(tempo_entries, key=lambda x: x[0])
        for _, line in tempo_entries_sorted:
            taf_lines.append(line)

    if not becmg_entries and not tempo_entries:
        taf_lines.append("NOSIG")

    source_marker = "METAR+MODEL FUSION" if metar else "MODEL FUSION"
    taf_lines.append(f"RMK AUTO FUSION (BMKG, ECMWF, ICON & GFS)")

    # sanitize & join preserving per-line structure
    taf_text = sanitize_taf_text_lines(taf_lines)
    return taf_lines, sorted(list(set(signif_times))), taf_text

# -----------------------
# Export helpers
# -----------------------
def make_taf_filename(issue_dt):
    return f"TAFOR-{TAF_FILENAME_STATION}-{issue_dt.strftime('%Y%m%d')}-{issue_dt.strftime('%H%M')}Z.txt"

def make_log_filenames(issue_dt):
    day = issue_dt.strftime("%Y%m%d")
    return f"LOG-{day}.txt", f"LOG-{day}.log"

def export_results_files(df_fused, df_probs, taf_lines, taf_text, issue_dt, dry_run=False):
    stamp = issue_dt.strftime("%Y%m%d_%H%M")
    taf_fname = make_taf_filename(issue_dt)
    taf_path = os.path.join("output", taf_fname)
    if not dry_run:
        with open(taf_path, "w", encoding="utf-8") as f:
            f.write(taf_text + "\n")
    csv_fname = f"output/fused_{stamp}.csv"
    json_fname = f"output/fused_{stamp}.json"
    if not dry_run:
        df_fused.to_csv(csv_fname, index=False)
        with open(json_fname, "w", encoding="utf-8") as f:
            json.dump({
                "issued_at": issue_dt.isoformat(),
                "taf_lines": taf_lines,
                "fused": df_fused.to_dict(orient="records"),
                "probabilities": df_probs.to_dict(orient="records") if df_probs is not None else []
            }, f, default=str, ensure_ascii=False, indent=2)
    return taf_path, csv_fname, json_fname

# -----------------------
# MAIN ACTION
# -----------------------
if st.button("🚀 Generate Operational TAFOR (Fusion)"):

    issue_dt = datetime.combine(issue_date, dtime(hour=issue_time, minute=0, second=0))
    st.info("📡 Fetching BMKG / Open-Meteo / METAR ... (please wait)")

    # fetch data
    bmkg_raw = fetch_bmkg()
    gfs_json = fetch_openmeteo("gfs")
    ecmwf_json = fetch_openmeteo("ecmwf")
    icon_json = fetch_openmeteo("icon")
    metar = fetch_metar_ogimet(ICAO_STATION)

    # determine availability
    availability = {
        "bmkg": bool(bmkg_raw),
        "ecmwf": bool(ecmwf_json),
        "icon": bool(icon_json),
        "gfs": bool(gfs_json)
    }

    # compute effective weights
    if auto_weight:
        eff_weights = compute_auto_weights(availability)
        show_weights_frac = {k: round(v, 3) for k, v in eff_weights.items()}
        st.info(f"Auto-weight active — effective weights (fractions): {show_weights_frac}")
    else:
        eff_weights = manual_weights
        st.info(f"Manual weights active — effective weights (fractions): {eff_weights}")

    st.success("✅ Data fetched (or fallback used). Processing fusion...")

    # parse model dfs
    df_gfs = openmeteo_json_to_df(gfs_json, "GFS")
    df_ecmwf = openmeteo_json_to_df(ecmwf_json, "ECMWF")
    df_icon = openmeteo_json_to_df(icon_json, "ICON")
    # BMKG parsing: attempt common structures
    df_bmkg = None
    try:
        if isinstance(bmkg_raw, dict) and bmkg_raw.get("data"):
            first = bmkg_raw["data"][0]
            cuaca = first.get("cuaca")
            df_bmkg = bmkg_cuaca_to_df(cuaca)
        else:
            df_bmkg = bmkg_cuaca_to_df(bmkg_raw)
    except Exception:
        df_bmkg = bmkg_cuaca_to_df(bmkg_raw)

    df_merged = align_hourly([df_gfs, df_ecmwf, df_icon, df_bmkg])
    if df_merged is None:
        st.error("No model data available to fuse.")
        st.stop()

    # fuse using eff_weights (note: our fuse expects weights keyed by model lower-case)
    df_fused = fuse_ensemble(df_merged, eff_weights, hours=validity)
    if df_fused is None or df_fused.empty:
        st.error("Fusion failed / empty result.")
        st.stop()

    # best-effort RH fill
    if "RH" not in df_fused.columns:
        rh_cols = [c for c in df_merged.columns if c.startswith("RH_")]
        if rh_cols:
            vals = df_merged[rh_cols].mean(axis=1)
            df_fused["RH"] = vals.values[:len(df_fused)]

    df_probs = compute_probabilities(df_merged)

    taf_lines, signif_times, taf_text = build_taf_from_fused(df_fused, df_merged, metar, issue_dt, validity, df_probs=df_probs)

    taf_path, csv_path, json_path = export_results_files(df_fused, df_probs, taf_lines, taf_text, issue_dt, dry_run=dry_run)

    # DISPLAY summary with availability
    st.subheader("📊 Source summary & availability")
    st.write({
        "BMKG ADM4 (Sedati Gede 35.15.17.2011)": "OK" if availability["bmkg"] else "Unavailable",
        "GFS": "OK" if availability["gfs"] else "Unavailable",
        "ECMWF": "OK" if availability["ecmwf"] else "Unavailable",
        "ICON": "OK" if availability["icon"] else "Unavailable",
        "METAR (OGIMET/NOAA)": "OK" if metar else "Unavailable"
    })

    st.markdown("### ℹ️ Effective ensemble weights (fractions)")
    st.write({k: round(v, 3) for k, v in eff_weights.items()})

    st.markdown("### 📡 METAR (Realtime OGIMET/NOAA)")
    st.code(metar or "Not available")

    st.markdown("### 📝 Generated TAFOR (Operational)")
    # show with <pre> to preserve per-line formatting
    st.markdown(f"<pre>{taf_text}</pre>", unsafe_allow_html=True)
    valid_to = issue_dt + timedelta(hours=validity)
    st.caption(f"Issued at {issue_dt:%d%H%MZ}, Valid {issue_dt:%d%H}/{valid_to:%d%H} UTC — Dry-run: {dry_run}")

    # Plot fused variables
    st.markdown("### 📈 Fused (sample)")
    fig, ax = plt.subplots(figsize=(9, 4))
    if not df_fused.empty:
        ax.plot(df_fused["time"], df_fused["T"], label="T (°C)")
        if "RH" in df_fused.columns:
            ax.plot(df_fused["time"], df_fused["RH"], label="RH (%)")
        ax.plot(df_fused["time"], df_fused["CC"], label="Cloud (%)")
        ax.plot(df_fused["time"], df_fused["WS"], label="Wind (kt)")
        for t in signif_times:
            ax.axvline(t, linestyle="--", alpha=0.6)
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=35)
    st.pyplot(fig)

    st.markdown("### 🔢 Probabilistic Metrics (sample)")
    st.dataframe(df_probs.head(24))

    # -----------------------
    # DOWNLOAD TAFOR final (.txt)
    # -----------------------
    st.markdown("### 💾 Download TAFOR Final")
    st.caption("Unduh hasil TAFOR final. Ingat: validasi manual tetap WAJIB sebelum rilis operasional.")
    taf_filename = make_taf_filename(issue_dt)
    taf_bytes = taf_text.encode("utf-8")
    st.download_button(label=f"⬇️ Download TAFOR (.txt) — {taf_filename}", data=taf_bytes, file_name=taf_filename, mime="text/plain")

    # -----------------------
    # LOGGING: create entry and save daily logs (if not dry-run)
    # -----------------------
    pop_max = round((df_probs["PoP_precip"].max() if not df_probs.empty else 0.0) * 100, 1)
    wind_max = round(df_fused["WS"].max() if not df_fused.empty else 0.0, 1)
    rh_max = round(df_fused["RH"].max() if "RH" in df_fused.columns and not df_fused.empty else 0.0, 1)
    cc_max = round(df_fused["CC"].max() if not df_fused.empty else 0.0, 1)

    alerts = []
    if pop_max >= 70:
        alerts.append(f"⚠️ High PoP ({pop_max}%) — possible heavy RA/TS")
    if wind_max >= 25:
        alerts.append(f"💨 High wind: {wind_max} kt")
    if (rh_max >= 90) and (cc_max >= 85):
        alerts.append("🌫️ High RH & cloud cover — possible low visibility / convective cloud")

    log_day = issue_dt.strftime("%Y%m%d")
    log_txt_name, log_log_name = make_log_filenames(issue_dt)
    log_path_txt = os.path.join("logs", log_txt_name)
    log_path_log = os.path.join("logs", log_log_name)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "issue_time": f"{issue_dt:%d%H%MZ}",
        "validity": validity,
        "metar": metar or "",
        "taf_text": taf_text,
        "pop_max_pct": pop_max,
        "wind_max_kt": wind_max,
        "rh_max_pct": rh_max,
        "cc_max_pct": cc_max,
        "alerts": "; ".join(alerts),
        "sources_ok": json.dumps({
            "bmkg": availability["bmkg"],
            "gfs": availability["gfs"],
            "ecmwf": availability["ecmwf"],
            "icon": availability["icon"],
            "metar": bool(metar)
        }),
        "effective_weights": json.dumps({k: round(v, 4) for k, v in eff_weights.items()})
    }

    log_line_human = f"[{log_entry['timestamp']}] ISSUE {log_entry['issue_time']} VALID {validity}h | alerts: {log_entry['alerts']} | sources_ok: {log_entry['sources_ok']} | weights: {log_entry['effective_weights']}\nTAF:\n{taf_text}\n---\n"
    if not dry_run:
        with open(log_path_txt, "a", encoding="utf-8") as f:
            f.write(log_line_human)
        with open(log_path_log, "a", encoding="utf-8") as f:
            f.write(log_line_human)

    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.info("✅ No significant alerts detected — conditions stable.")

    # -----------------------
    # LOG VIEWER (terminal-like)
    # -----------------------
    st.markdown("### 🔍 Log Viewer (terminal-like)")
    log_files = sorted([fn for fn in os.listdir("logs") if fn.startswith("LOG-")], reverse=True)
    if not log_files:
        st.info("No logs available yet.")
    else:
        chosen = st.selectbox("Pilih file log untuk dilihat (terbaru)", log_files, index=0)
        if chosen:
            with open(os.path.join("logs", chosen), "r", encoding="utf-8") as f:
                content = f.read()
            st.code(content, language=None)
            with open(os.path.join("logs", chosen), "rb") as f:
                log_bytes = f.read()
            st.download_button(label=f"⬇️ Download log (.txt)", data=log_bytes, file_name=chosen, mime="text/plain")
            if chosen.endswith(".txt"):
                mirror = chosen.replace(".txt", ".log")
            else:
                mirror = chosen + ".log"
            st.download_button(label=f"⬇️ Download log (.log)", data=log_bytes, file_name=mirror, mime="text/plain")

    with st.expander("🔧 Debug: raw BMKG JSON"):
        st.write(bmkg_raw)

    st.success("✅ Operational TAFOR (fusion) created (dry-run=%s), TAFOR & logs tersedia untuk diunduh. SELALU VALIDASI MANUAL sebelum publikasi operasional." % dry_run)
