# ============================================================
# Zaman Serisi – SARIMA (Lead Time Forecast)
# ============================================================

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


# ------------------------------------------------------------
# 1) AYARLAR
# ------------------------------------------------------------
TARGET_COL = "lead_time"
TIME_COL = "time"

SEASONAL_PERIOD = 24          # saatlik veri → günlük mevsimsellik
TEST_RATIO = 0.2              # %20 test


# ------------------------------------------------------------
# 2) VERİYİ YÜKLE
# ------------------------------------------------------------
DATA_PATH = Path("data/raw/production_sim.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError("data/raw/production_sim.csv bulunamadı")

df = pd.read_csv(DATA_PATH, parse_dates=[TIME_COL])
df = df.sort_values(TIME_COL).reset_index(drop=True)

y = df[TARGET_COL].astype(float)


# ------------------------------------------------------------
# 3) TRAIN / TEST AYIR
# ------------------------------------------------------------
n_total = len(y)
n_test = int(n_total * TEST_RATIO)

y_train = y.iloc[:-n_test]
y_test  = y.iloc[-n_test:]

time_test = df[TIME_COL].iloc[-n_test:]


# ------------------------------------------------------------
# 4) SARIMA MODELİ
# ------------------------------------------------------------
# SARIMA(1,1,1)(1,1,1,24)
model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)


# ------------------------------------------------------------
# 5) TAHMİN
# ------------------------------------------------------------
y_pred = results.forecast(steps=len(y_test))
y_pred = np.maximum(y_pred, 0)  # negatif lead_time koruması


# ------------------------------------------------------------
# 6) METRİKLER
# ------------------------------------------------------------
mape = mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

metrics = {
    "model": "SARIMA",
    "order": [1, 1, 1],
    "seasonal_order": [1, 1, 1, SEASONAL_PERIOD],
    "MAPE": float(mape),
    "RMSE": float(rmse),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test))
}


# ------------------------------------------------------------
# 7) ÇIKTILARI KAYDET
# ------------------------------------------------------------
OUT_BASE = Path("zs_time_series/outputs")
OUT_FORE = OUT_BASE / "forecasts"
OUT_MET  = OUT_BASE / "metrics"
OUT_PLOT = OUT_BASE / "plots"

for p in [OUT_FORE, OUT_MET, OUT_PLOT]:
    p.mkdir(parents=True, exist_ok=True)

# Tahmin CSV
forecast_df = pd.DataFrame({
    "time": time_test.values,
    "y_true": y_test.values,
    "y_pred": y_pred.values
})

forecast_df.to_csv(
    OUT_FORE / "sarima_lead_time_forecast.csv",
    index=False
)

# Metrikler JSON
with open(OUT_MET / "sarima_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# Grafik
plt.figure(figsize=(10, 4))
plt.plot(df[TIME_COL].iloc[-200:], y.iloc[-200:], label="Gerçek", linewidth=2)
plt.plot(forecast_df["time"], forecast_df["y_pred"], label="SARIMA Tahmin", linewidth=2)
plt.title("Lead Time – SARIMA Forecast")
plt.xlabel("Time")
plt.ylabel("Lead Time")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PLOT / "sarima_forecast.png", dpi=150)
plt.close()


# ------------------------------------------------------------
# 8) ÖZET
# ------------------------------------------------------------
print(" SARIMA ZAMAN SERİSİ TAMAMLANDI")
print(f"MAPE : {mape:.4f}")
print(f"RMSE : {rmse:.4f}")
print("Çıktılar:")
print(" - zs_time_series/outputs/forecasts/sarima_lead_time_forecast.csv")
print(" - zs_time_series/outputs/metrics/sarima_metrics.json")
print(" - zs_time_series/outputs/plots/sarima_forecast.png")
