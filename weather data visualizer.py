from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

OUTPUT_DIR = Path("./weather_analysis_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILENAME = "weather_1.csv"
if not Path(CSV_FILENAME).exists():
    raise FileNotFoundError(f"Expected file '{CSV_FILENAME}' in the current working directory.")

# --- Load ---
df = pd.read_csv(CSV_FILENAME)
print("Loaded CSV. Columns:")
for c in df.columns:
    print(" -", c)

# --- Robust column detection by substring ---
col_candidates = {c: c.lower().strip() for c in df.columns}

def detect_by_substrings(subs):
    for col, lc in col_candidates.items():
        for s in subs:
            if s in lc:
                return col
    return None

date_col = detect_by_substrings(["date", "time", "observation", "timestamp"])
temp_col = detect_by_substrings(["temp", "temperature", "°c", "tmean"])
min_temp_col = detect_by_substrings(["min_temp", "mintemp", "minimum", "tmin", "min temp"])
max_temp_col = detect_by_substrings(["max_temp", "maxtemp", "maximum", "tmax", "max temp"])
rain_col = detect_by_substrings(["rain", "precip", "precipitation"])
humidity_col = detect_by_substrings(["humid", "humidity", "rel_humidity", "rh", "%"])

print("\nDetected (best-effort):")
print(" date_col:", date_col)
print(" temp_col:", temp_col)
print(" min_temp_col:", min_temp_col)
print(" max_temp_col:", max_temp_col)
print(" rain_col:", rain_col)
print(" humidity_col:", humidity_col)

# If nothing for date, fallback to first column
if date_col is None:
    date_col = list(df.columns)[0]
    print("No obvious date column — using first column:", date_col)

# --- Parse datetime robustly ---
def try_parse_dates(series):
    # try several strategies, return parsed series
    s = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
    if s.notna().sum() > 0:
        return s
    s = pd.to_datetime(series, errors='coerce', dayfirst=True, infer_datetime_format=True)
    if s.notna().sum() > 0:
        return s
    # try common formats
    fmts = ["%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M", "%m/%d/%Y", "%d/%m/%Y"]
    for f in fmts:
        s = pd.to_datetime(series, format=f, errors='coerce')
        if s.notna().sum() > 0:
            return s
    # last resort: try parsing each element individually (slower)
    parsed = pd.Series([pd.to_datetime(str(x), errors='coerce', dayfirst=True) for x in series])
    return parsed

df[date_col] = try_parse_dates(df[date_col])
n_valid_dates = df[date_col].notna().sum()
print(f"\nParsed dates: {n_valid_dates}/{len(df)} valid.")

# drop rows where date couldn't be parsed
initial_len = len(df)
df = df[df[date_col].notna()].copy()
dropped = initial_len - len(df)
print(f"Dropped {dropped} rows with invalid dates.")

# --- Standardize numeric columns (create canonical names if possible) ---
def to_numeric_col(name, canonical):
    if name is None:
        return False
    if name in df.columns:
        df[canonical] = pd.to_numeric(df[name], errors='coerce')
        return True
    return False

created_temp = False
if temp_col and temp_col in df.columns:
    created_temp = to_numeric_col(temp_col, "temperature")

# if mean not present but min+max present compute mean
if not created_temp and min_temp_col and max_temp_col and min_temp_col in df.columns and max_temp_col in df.columns:
    df["min_temperature"] = pd.to_numeric(df[min_temp_col], errors='coerce')
    df["max_temperature"] = pd.to_numeric(df[max_temp_col], errors='coerce')
    df["temperature"] = df[["min_temperature", "max_temperature"]].mean(axis=1)
    created_temp = True
else:
    # individually create min/max if present
    if min_temp_col and min_temp_col in df.columns:
        df["min_temperature"] = pd.to_numeric(df[min_temp_col], errors='coerce')
    if max_temp_col and max_temp_col in df.columns:
        df["max_temperature"] = pd.to_numeric(df[max_temp_col], errors='coerce')

created_rain = to_numeric_col(rain_col, "rainfall")
created_humidity = to_numeric_col(humidity_col, "humidity")
# fallback defaults
if "rainfall" not in df.columns:
    df["rainfall"] = 0.0
if "humidity" not in df.columns:
    df["humidity"] = np.nan

# set index
df.sort_values(by=date_col, inplace=True)
df.set_index(date_col, inplace=True)

# fill numeric missing values: ffill,bfill, mean
for c in ["temperature", "min_temperature", "max_temperature", "rainfall", "humidity"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].ffill().bfill()
        if df[c].isna().any():
            df[c].fillna(df[c].mean(skipna=True), inplace=True)

# export cleaned
cleaned_path = OUTPUT_DIR / "cleaned_weather_1.csv"
df.to_csv(cleaned_path)
print("\nCleaned data exported to:", cleaned_path)

# --- Build aggregation dict dynamically only for columns that exist ---
agg_map = {}
if "temperature" in df.columns:
    agg_map["temperature"] = ['mean', 'min', 'max', 'std']
if "rainfall" in df.columns:
    agg_map["rainfall"] = ['sum', 'mean', 'std']
if "humidity" in df.columns:
    agg_map["humidity"] = ['mean', 'min', 'max', 'std']

if not agg_map:
    raise ValueError("No numeric columns detected for aggregation (temperature/rainfall/humidity).")

# daily / monthly / yearly
daily = df.resample('D').agg(agg_map)
daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
daily.to_csv(OUTPUT_DIR / "daily_summary.csv")
print("Daily summary saved.")

monthly = df.resample('M').agg(agg_map)
monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
monthly.to_csv(OUTPUT_DIR / "monthly_summary.csv")
print("Monthly summary saved.")

yearly = df.resample('Y').agg(agg_map)
yearly.columns = ['_'.join(col).strip() for col in yearly.columns.values]
yearly.to_csv(OUTPUT_DIR / "yearly_summary.csv")
print("Yearly summary saved.")

# --- Plots (only create plots if required series exist) ---
plt.close('all')
# daily temperature line
if 'temperature_mean' in daily.columns:
    plt.figure(figsize=(10,4))
    plt.plot(daily.index, daily['temperature_mean'])
    plt.title("Daily Mean Temperature")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "daily_mean_temperature.png")
    plt.close()
    print("Saved: daily_mean_temperature.png")

# monthly rainfall bar
if 'rainfall_sum' in monthly.columns:
    plt.figure(figsize=(10,4))
    months = monthly.index.strftime('%Y-%m')
    plt.bar(months, monthly['rainfall_sum'])
    plt.xticks(rotation=45, ha='right')
    plt.title("Monthly Rainfall Totals")
    plt.xlabel("Month")
    plt.ylabel("Rainfall (sum)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_rainfall_totals.png")
    plt.close()
    print("Saved: monthly_rainfall_totals.png")

# scatter humidity vs temp (daily)
if 'temperature_mean' in daily.columns and 'humidity_mean' in daily.columns:
    plt.figure(figsize=(6,6))
    plt.scatter(daily['temperature_mean'], daily['humidity_mean'])
    plt.title("Humidity vs Temperature (daily means)")
    plt.xlabel("Temperature (mean)")
    plt.ylabel("Humidity (mean)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "humidity_vs_temperature_scatter.png")
    plt.close()
    print("Saved: humidity_vs_temperature_scatter.png")

# combined monthly (temp & rainfall)
if 'temperature_mean' in monthly.columns and 'rainfall_sum' in monthly.columns:
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(monthly.index, monthly['temperature_mean'], label='Temp (mean)')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Monthly Mean Temperature")
    ax1.set_xticks(monthly.index)
    ax1.set_xticklabels(monthly.index.strftime('%Y-%m'), rotation=45, ha='right')
    ax2 = ax1.twinx()
    ax2.bar(monthly.index, monthly['rainfall_sum'], alpha=0.3, label='Rainfall (sum)')
    ax2.set_ylabel("Monthly Rainfall (sum)")
    plt.title("Monthly Mean Temperature and Rainfall (combined)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_temp_rainfall_combined.png")
    plt.close()
    print("Saved: monthly_temp_rainfall_combined.png")

# --- Group by month and season ---
df_month = df.copy()
df_month['month'] = df_month.index.month
grouped_by_month = df_month.groupby('month').agg({
    k: (['mean','min','max','std'] if k=='temperature' else (['sum','mean'] if k=='rainfall' else ['mean','min','max']))
    for k in ['temperature','rainfall','humidity'] if k in df_month.columns
})
if not grouped_by_month.empty:
    grouped_by_month.columns = ['_'.join(col).strip() for col in grouped_by_month.columns.values]
    grouped_by_month.to_csv(OUTPUT_DIR / "grouped_by_month.csv")
    print("Saved: grouped_by_month.csv")

def month_to_season(m):
    if m in [12,1,2]: return 'DJF'
    if m in [3,4,5]: return 'MAM'
    if m in [6,7,8]: return 'JJA'
    return 'SON'

df_season = df.copy()
df_season['season'] = df_season.index.month.map(month_to_season)
grouped_by_season = df_season.groupby('season').agg({
    k: (['mean','min','max','std'] if k=='temperature' else (['sum','mean'] if k=='rainfall' else ['mean','min','max']))
    for k in ['temperature','rainfall','humidity'] if k in df_season.columns
})
if not grouped_by_season.empty:
    grouped_by_season.columns = ['_'.join(col).strip() for col in grouped_by_season.columns.values]
    grouped_by_season.to_csv(OUTPUT_DIR / "grouped_by_season.csv")
    print("Saved: grouped_by_season.csv")

# --- Report ---
report_path = OUTPUT_DIR / "report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Weather Analysis Report\n\n")
    f.write(f"Source file: {CSV_FILENAME}\n\n")
    f.write(f"Rows after cleaning: {len(df)}\n\n")
    if len(df)>0:
        f.write(f"Date range: {df.index.min().date()} to {df.index.max().date()}\n\n")
    f.write("Files generated:\n\n")
    for p in sorted(OUTPUT_DIR.iterdir()):
        f.write(f"- {p.name}\n")
    f.write("\nAutomatic insights:\n\n")
    try:
        if 'temperature_max' in daily.columns:
            hottest = daily['temperature_max'].idxmax().date()
            f.write(f"- Hottest day (daily max): {hottest}\n")
        if 'rainfall_sum' in monthly.columns:
            wettest = monthly['rainfall_sum'].idxmax().strftime('%Y-%m')
            f.write(f"- Wettest month: {wettest}\n")
    except Exception:
        f.write("- Some insights could not be computed.\n")
print("\nReport written to", report_path)
print("\nAll done — outputs are in:", OUTPUT_DIR.resolve())
