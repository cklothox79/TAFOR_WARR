from datetime import datetime

def utc_to_wib(utc_time_str):
    try:
        dt = datetime.fromisoformat(utc_time_str)
        wib = dt.hour + 7
        return f"{wib%24:02d}:{dt.minute:02d}"
    except:
        return utc_time_str
