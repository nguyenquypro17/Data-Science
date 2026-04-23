import pandas as pd
from pathlib import Path

# Hanoi city center 
LAT = 21.0285
LON = 105.8542

START_DATE = "2013-01-01"
END_DATE = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
WEATHER_CHUNK_DAYS = 20
LOCAL_TIMEZONE = "Asia/Ho_Chi_Minh" 

BASE_DIR = Path(".")
RAW_AIR_DIR = BASE_DIR / "data" / "raw" / "air_quality"
RAW_WEATHER_DIR = BASE_DIR / "data" / "raw" / "weather"
PROCESSED_DIR = BASE_DIR / "data" / "processed"