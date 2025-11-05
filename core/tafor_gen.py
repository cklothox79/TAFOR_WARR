import pandas as pd

def tafor_generate(fused_df):
    """
    Generator TAF sederhana: membaca nilai rata-rata & kondisi dominan.
    Bisa dikembangkan ke versi probabilistik nantinya.
    """
    if fused_df.empty:
        return "TAF AUTO: Data tidak tersedia"

    t_mean = fused_df["temperature_2m"].mean()
    rh_mean = fused_df["relative_humidity_2m"].mean()
    ws_max = fused_df["wind_speed_10m"].max()
    cc_mean = fused_df["cloud_cover"].mean()
    pop_mean = fused_df["precipitation"].mean()

    taf = f"""
TAF AUTO WARR {fused_df['time'].iloc[0][:10]}
TEMP {t_mean:.1f}C RH {rh_mean:.0f}% WIND {ws_max:.1f}KT
CLOUD {cc_mean:.0f}% PRECIP {pop_mean:.2f}mm
"""
    return taf.strip()
