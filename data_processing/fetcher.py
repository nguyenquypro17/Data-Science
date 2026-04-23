import time
import requests
import pandas as pd
from config import LAT, LON, WEATHER_CHUNK_DAYS, LOCAL_TIMEZONE

# Hàm này dùng để chia nhỏ khoảng thời gian dài thành các chunk nhỏ hơn để tránh lỗi timeout hoặc quá tải API khi fetch dữ liệu thời tiết lịch sử
def date_chunks(start_date: str, end_date: str, chunk_days: int):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    cur = start
    step = pd.Timedelta(days=chunk_days - 1)
    while cur <= end:
        chunk_end = min(cur + step, end)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + pd.Timedelta(days=1)

# Hàm này thực hiện request GET đến API và trả về kết quả dưới dạng dict. Nó có cơ chế retry với backoff để đảm bảo độ ổn định khi gặp lỗi mạng hoặc lỗi server tạm thời.
def get_json(url: str, params: dict, timeout: int = 90, retries: int = 4) -> dict:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        resp = None
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            detail = resp.text if resp is not None else ""
            try:
                detail = resp.json().get("reason", detail) if resp is not None else detail
            except ValueError:
                pass

            status_code = resp.status_code if resp is not None else None
            last_error = requests.HTTPError(f"{exc}. API detail: {detail}")
            retriable = status_code is not None and status_code >= 500

            if not retriable or attempt == retries:
                raise last_error from exc
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                raise

        time.sleep(min(2 * attempt, 6))

    if last_error is not None:
        raise last_error

    raise RuntimeError("Unexpected request flow in get_json")

def hourly_payload_to_df(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["hourly"])
    # Chuyển đổi từ UTC sang giờ Hà Nội chuẩn xác trước khi tính các feature ngày/tháng
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
    return df.rename(columns={"time": "datetime"})

def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    base_params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi,european_aqi",
        "timezone": "UTC", # Lấy gốc UTC để xử lý đồng nhất
    }
    try:
        payload = get_json(url, {**base_params, "domains": "cams_europe"})
    except requests.HTTPError:
        payload = get_json(url, base_params)
    return hourly_payload_to_df(payload)

def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    parts = []
    for chunk_start, chunk_end in date_chunks(start_date, end_date, WEATHER_CHUNK_DAYS):
        params = {
            "latitude": LAT, "longitude": LON,
            "start_date": chunk_start, "end_date": chunk_end,
            "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,surface_pressure,precipitation,rain,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "wind_speed_unit": "ms", "timezone": "UTC",
        }
        parts.append(hourly_payload_to_df(get_json(url, params)))
    return pd.concat(parts, ignore_index=True)