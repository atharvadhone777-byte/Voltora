"""Preprocessing for UCI Household Electric Power Consumption (minute-level, 1 household, France).
Aggregation policy: minute -> hourly. Power/voltage/intensity = mean; sub-metering (Wh) = sum.
Missing policy: '?' -> NaN; duplicate timestamps dropped; hourly gaps <= 6h linearly interpolated;
longer gaps stay NaN and any window touching NaN is skipped (never silently filled)."""
import numpy as np, pandas as pd

TARGET = "Global_active_power"
BASE = [TARGET, "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
FEATURES = BASE + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
SUM_COLS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

def load_hourly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", na_values=["?", ""], low_memory=False)
    df["ts"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["ts"]).drop_duplicates("ts").sort_values("ts").set_index("ts")
    df = df[BASE].apply(pd.to_numeric, errors="coerce")
    agg = {c: ("sum" if c in SUM_COLS else "mean") for c in BASE}
    h = df.resample("h").agg(agg, min_count=1)   # all-NaN hours stay NaN
    return h.interpolate(limit=6, limit_area="inside")

def add_time_features(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    h["hour_sin"] = np.sin(2*np.pi*h.index.hour/24); h["hour_cos"] = np.cos(2*np.pi*h.index.hour/24)
    h["dow_sin"] = np.sin(2*np.pi*h.index.dayofweek/7); h["dow_cos"] = np.cos(2*np.pi*h.index.dayofweek/7)
    return h[FEATURES]

def chrono_split(h, train=0.70, val=0.15):
    n = len(h); a, b = int(n*train), int(n*(train+val))
    return h.iloc[:a], h.iloc[a:b], h.iloc[b:]

def make_windows(scaled: np.ndarray, lookback: int, horizon: int = 24):
    """Windows are built inside a single split, so no future leakage across split boundaries."""
    X, y = [], []
    for i in range(lookback, len(scaled) - horizon + 1):
        w, t = scaled[i-lookback:i], scaled[i:i+horizon, 0]
        if np.isnan(w).any() or np.isnan(t).any(): continue
        X.append(w); y.append(t)
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")
