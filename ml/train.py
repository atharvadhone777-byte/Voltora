"""python train.py --data household_power_consumption.txt --out model_artifacts [--lookback 168]
Kaggle: add dataset uciml/electric-power-consumption-data-set; path is
/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt"""
import argparse, json, os, numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from preprocessing import *

def smape(y, p): return float(np.mean(2*np.abs(p-y)/(np.abs(y)+np.abs(p)+1e-8))*100)
def report(y, p): return {"mae": float(np.mean(np.abs(y-p))), "rmse": float(np.sqrt(np.mean((y-p)**2))), "smape": smape(y, p)}

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", default="model_artifacts")
ap.add_argument("--lookback", type=int, default=168); ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); H = 24
tf.keras.utils.set_random_seed(a.seed); os.makedirs(a.out, exist_ok=True)

h = add_time_features(load_hourly(a.data))
tr, va, te = chrono_split(h)
sc = StandardScaler().fit(tr.dropna().values)           # train-only fit
S = lambda d: sc.transform(d.values)
Xtr, ytr = make_windows(S(tr), a.lookback, H); Xva, yva = make_windows(S(va), a.lookback, H)
Xte, yte = make_windows(S(te), a.lookback, H)
print("windows", Xtr.shape, Xva.shape, Xte.shape)

m = tf.keras.Sequential([tf.keras.layers.Input((a.lookback, len(FEATURES))),
    tf.keras.layers.LSTM(64, return_sequences=True), tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32), tf.keras.layers.Dense(H)])
m.compile("adam", "mse")
cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
      tf.keras.callbacks.ModelCheckpoint(f"{a.out}/model.keras", save_best_only=True)]
m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=a.epochs, batch_size=a.batch, callbacks=cb)
m = tf.keras.models.load_model(f"{a.out}/model.keras")

mu, sd = sc.mean_[0], sc.scale_[0]
inv = lambda v: v*sd + mu                                  # back to kW
p, y = inv(m.predict(Xte, batch_size=256)), inv(yte)
last24 = inv(Xte[:, -H:, 0])                               # persistence baseline: repeat last 24h
metrics = {"lstm": report(y, p), "persistence_24h": report(y, last24), "unit": "kW (hourly mean)",
           "n_test_windows": int(len(y)), "lookback": a.lookback, "horizon": H}
json.dump(metrics, open(f"{a.out}/metrics.json", "w"), indent=2)
json.dump({"features": FEATURES, "target": TARGET, "lookback": a.lookback, "horizon": H,
           "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
           "sum_cols": SUM_COLS, "trained": True}, open(f"{a.out}/config.json", "w"), indent=2)
np.savez(f"{a.out}/test_sample.npz", actual=y[:200, 0], predicted=p[:200, 0])
print(json.dumps(metrics, indent=2))
