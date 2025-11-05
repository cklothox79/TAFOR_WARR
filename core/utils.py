from datetime import datetime

def utc_to_wib(utc_time_str):
    try:
        dt = datetime.fromisoformat(utc_time_str)
        wib_hour = (dt.hour + 7) % 24
        return f"{wib_hour:02d}:{dt.minute:02d}"
    except Exception:
        return utc_time_str
