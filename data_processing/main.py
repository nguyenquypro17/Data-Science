import pandas as pd
from config import START_DATE, END_DATE, RAW_AIR_DIR, RAW_WEATHER_DIR, PROCESSED_DIR
from fetcher import fetch_air_quality, fetch_weather
from cleaner import clean_and_merge
import features 
from sklearn.preprocessing import StandardScaler 

def setup_directories():
    for p in [RAW_AIR_DIR, RAW_WEATHER_DIR, PROCESSED_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def month_ranges(start_date: str, end_date: str):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    cur = start.replace(day=1)
    while cur <= end:
        month_start = max(cur, start)
        month_end = min(cur + pd.offsets.MonthEnd(0), end)
        yield month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")
        cur = cur + pd.offsets.MonthBegin(1)

def main():
    # Khởi tạo thư mục trước khi chạy pipeline
    setup_directories()

    # air_parts = []
    # weather_parts = []

    # # 1. Fetching Data
    # for start_date, end_date in month_ranges(START_DATE, END_DATE):
    #     print(f"Fetching {start_date} -> {end_date}")
    #     aq = fetch_air_quality(start_date, end_date)
    #     wt = fetch_weather(start_date, end_date)
        
    #     air_parts.append(aq)
    #     weather_parts.append(wt)
        
    #     month_tag = pd.Timestamp(start_date).strftime("%Y_%m")
    #     aq.to_csv(RAW_AIR_DIR / f"air_quality_{month_tag}.csv", index=False)
    #     wt.to_csv(RAW_WEATHER_DIR / f"weather_{month_tag}.csv", index=False)

    # aq_all = pd.concat(air_parts, ignore_index=True)
    # wt_all = pd.concat(weather_parts, ignore_index=True)

    # --- ĐỌC DỮ LIỆU AIR QUALITY ---
    air_files = sorted(RAW_AIR_DIR.glob("air_quality_*.csv"))
    if not air_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong {RAW_AIR_DIR}")
        
    air_parts = [pd.read_csv(f) for f in air_files]
    aq_all = pd.concat(air_parts, ignore_index=True)
    
    # --- ĐỌC DỮ LIỆU WEATHER ---
    weather_files = sorted(RAW_WEATHER_DIR.glob("weather_*.csv"))
    if not weather_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong {RAW_WEATHER_DIR}")
        
    weather_parts = [pd.read_csv(f) for f in weather_files]
    wt_all = pd.concat(weather_parts, ignore_index=True)

    print("Parsing datetime columns...")
    aq_all["datetime"] = pd.to_datetime(aq_all["datetime"])
    wt_all["datetime"] = pd.to_datetime(wt_all["datetime"])

    # --- 2. CLEAN & MERGE (Sửa lỗi Leakage trong cleaner.py nếu cần) ---
    print("Cleaning and Merging...")
    clean_df = clean_and_merge(aq_all, wt_all)

    # --- 3. FEATURE ENGINEERING (Bao gồm Cyclical & Target Encoding) ---
    print("Building Features...")
    full_df = features.build_features(clean_df)

    # --- 4. 🆕 FIX 5: DROPNA TRƯỚC KHI SPLIT & SCALE ---
    # Phải xóa NaN trước để đảm bảo phân phối dữ liệu (Mean/Std) của Scaler chính xác
    print("Dropping NaN before scaling...")
    critical_cols = ["pm25_lag168", "pm25_rolling_7d", "target_pm25_t+24"]
    full_df = full_df.dropna(subset=critical_cols).reset_index(drop=True)

    # --- 5. CHIA TẬP DỮ LIỆU THEO THỜI GIAN (Tránh Leakage) ---
    print("Splitting data (Time-based)...")
    train_size = int(len(full_df) * 0.8)
    # Tách riêng biệt hoàn toàn
    train_df = full_df.iloc[:train_size].copy()
    test_df = full_df.iloc[train_size:].copy()

    # --- 6. 🆕 FIX: SCALING AN TOÀN ---
    print("Scaling features...")
    # Loại bỏ các cột không được scale (Target Class đã được mã hóa số ở features.py nên không cần loại bỏ nếu muốn scale)
    exclude_cols = ['datetime', 'target_pm25_t+1', 'target_pm25_t+6', 'target_pm25_t+24',
                    'target_aqi_class_t+1', 'target_aqi_class_t+6', 'target_aqi_class_t+24']
    
    feature_cols = [c for c in full_df.select_dtypes(include=['number']).columns if c not in exclude_cols]
    
    scaler = StandardScaler()
    
    # 🆕 FIT CHỈ TRÊN TRAIN
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    
    # 🆕 TRANSFORM TRÊN TEST (Dùng Mean/Std của Train)
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Gộp lại để lưu file Full
    full_df_final = pd.concat([train_df, test_df], axis=0)

    # --- 7. LƯU KẾT QUẢ ---
    print("Saving processed datasets...")
    full_df_final.to_csv(PROCESSED_DIR / "hanoi_air_ml_ready_full.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "hanoi_air_ml_ready_train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "hanoi_air_ml_ready_test.csv", index=False)

    print(f"\n✅ Pipeline hoàn tất!")
    print(f"Dữ liệu huấn luyện: {len(train_df):,} dòng.")
    print(f"Dữ liệu kiểm thử: {len(test_df):,} dòng.")

if __name__ == "__main__":
    main()