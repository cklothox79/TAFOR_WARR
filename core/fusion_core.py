import pandas as pd
import numpy as np

# Bobot model
WEIGHTS = {
    "BMKG": 0.5,
    "ECMWF": 0.2,
    "ICON": 0.15,
    "GFS": 0.15
}

def weighted_fusion(df):
    """Gabungkan semua data model berdasarkan bobot sumber."""
    if df.empty:
        return pd.DataFrame()
    
    variables = ["temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m", "precipitation"]
    fused = []
    for time in sorted(df["time"].unique()):
        sub = df[df["time"] == time]
        row = {"time": time}
        for var in variables:
            total, wsum = 0, 0
            for _, r in sub.iterrows():
                w = WEIGHTS.get(r["source"], 0)
                if not pd.isna(r[var]):
                    total += r[var] * w
                    wsum += w
            row[var] = total / wsum if wsum else np.nan
        fused.append(row)
    return pd.DataFrame(fused)
