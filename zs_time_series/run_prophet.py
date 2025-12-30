# ============================================================
# Zaman Serisi – PROPHET (Lead Time Forecast)
# ============================================================

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


# ------------------------------------------------------------
# 1) AYARLAR
# ------------------------------------------------------------
TARGET_COL = "lead_time"
TIME_COL = "time"
HORIZON = 24                  # 24 saat ileri tahmin


# ------------------------------------------------------------
# 2) VERİYİ YÜKLE
# ------------------------------------------------------------
DATA_PATH = Path("data/raw/production_sim.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError("data/raw/production_sim.csv bulunamadı")

df = pd.read_csv(DATA_PATH, parse_dates=[TIME_COL])
df = df.sort_values(TIME_COL).reset_index(drop=True)

ts = df[[TIME_COL, TARGET_COL]].rename(
    columns={TIME_COL: "ds", TARGET_COL: "y"}
)


# ------------------------------------------------------------
# 3) TRAIN / TEST
# ------------------------------------------------------------
train = ts.iloc[:-HORIZON]
test  = ts.iloc[-HORIZON:]


# ------------------------------------------------------------
# 4) PROPHET MODELİ
# ------------------------------------------------------------
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=False,
    yearly_seasonality=False
)

model.fit(train)

future = model.make_future_dataframe(periods=HORIZON, freq="H")
forecast = model.predict(future)


# ------------------------------------------------------------
# 5) TAHMİN & METRİKLER
# ------------------------------------------------------------
y_true = test["y"].values
y_pred = forecast["yhat"].values[-HORIZON:]
y_pred = np.maximum(y_pred, 0)   # negatif lead_time koruması

mape = mean_absolute_percentage_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))


# ------------------------------------------------------------
# 6) TIKANMA RİSK SKORU
# ------------------------------------------------------------
threshold = ts["y"].quantile(0.85)

forecast["risk_score"] = np.clip(
    (forecast["yhat"] - threshold) / threshold,
    0, 1
)


# ------------------------------------------------------------
# 7) ÇIKTILAR
# ------------------------------------------------------------
OUT_BASE = Path("zs_time_series/outputs")
OUT_FORE = OUT_BASE / "forecasts"
OUT_MET  = OUT_BASE / "metrics"
OUT_PLOT = OUT_BASE / "plots"

for p in [OUT_FORE, OUT_MET, OUT_PLOT]:
    p.mkdir(parents=True, exist_ok=True)

# Tahmin CSV
forecast[["ds", "yhat", "risk_score"]].to_csv(
    OUT_FORE / "prophet_lead_time_forecast.csv",
    index=False
)

# Metrikler
pd.DataFrame([{
    "model": "Prophet",
    "MAPE": mape,
    "RMSE": rmse
}]).to_csv(
    OUT_MET / "prophet_metrics.csv",
    index=False
)

# Grafik – Tahmin
fig1 = model.plot(forecast)
fig1.savefig(OUT_PLOT / "prophet_forecast.png", dpi=150)

# Grafik – Trend & Mevsimsellik
fig2 = model.plot_components(forecast)
fig2.savefig(OUT_PLOT / "prophet_components.png", dpi=150)

plt.close("all")


# ------------------------------------------------------------
# 8) ÖZET
# ------------------------------------------------------------
print(" PROPHET ZAMAN SERİSİ TAMAMLANDI")
print(f"MAPE : {mape:.4f}")
print(f"RMSE : {rmse:.4f}")
print("Çıktılar:")
print(" - zs_time_series/outputs/forecasts/prophet_lead_time_forecast.csv")
print(" - zs_time_series/outputs/metrics/prophet_metrics.csv")
print(" - zs_time_series/outputs/plots/prophet_forecast.png")
print(" - zs_time_series/outputs/plots/prophet_components.png")
