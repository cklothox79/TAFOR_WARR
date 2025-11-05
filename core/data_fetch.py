import requests
import pandas as pd

def fetch_bmkg(json_url):
    try:
        df = pd.read_json(json_url)
        df["source"] = "BMKG"
        return df
    except Exception as e:
        print(f"[BMKG] Error: {e}")
        return pd.DataFrame()

def fetch_openmeteo(lat, lon, model="gfs"):
    base_url = f"https://api.open-meteo.com/v1/{model}"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m", "precipitation"],
        "forecast_days": 2,
        "timezone": "auto"
    }
    try:
        r = requests.get(base_url, params=params, timeout=10)
        data = r.json()
        df = pd.DataFrame(data["hourly"])
        df["source"] = model.upper()
        return df
    except Exception as e:
        print(f"[{model}] Error: {e}")
        return pd.DataFrame()

def fetch_all(lat, lon, bmkg_json):
    dfs = [
        fetch_bmkg(bmkg_json),
        fetch_openmeteo(lat, lon, "gfs"),
        fetch_openmeteo(lat, lon, "icon"),
        fetch_openmeteo(lat, lon, "ecmwf")
    ]
    return pd.concat(dfs, ignore_index=True)
