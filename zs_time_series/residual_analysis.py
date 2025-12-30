# ============================================================
# Zaman Serisi – Residual & ACF/PACF Analizi
# ============================================================

from __future__ import annotations
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ------------------------------------------------------------
# 1) VERİLERİ YÜKLE
# ------------------------------------------------------------
DATA_PATH = Path("data/raw/production_sim.csv")
FORECAST_PATH = Path(
    "zs_time_series/outputs/forecasts/sarima_lead_time_forecast.csv"
)

df = pd.read_csv(DATA_PATH, parse_dates=["time"])
fc = pd.read_csv(FORECAST_PATH, parse_dates=["time"])

# Residual = gerçek - tahmin
residuals = fc["y_true"] - fc["y_pred"]


# ------------------------------------------------------------
# 2) ÇIKTI KLASÖRLERİ
# ------------------------------------------------------------
OUT_PLOT = Path("zs_time_series/outputs/plots")
OUT_PLOT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 3) RESIDUAL ZAMAN SERİSİ
# ------------------------------------------------------------
plt.figure(figsize=(10, 3))
plt.plot(fc["time"], residuals)
plt.axhline(0, linestyle="--", color="black")
plt.title("SARIMA Residuals (y_true - y_pred)")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(OUT_PLOT / "sarima_residuals.png", dpi=150)
plt.close()


# ------------------------------------------------------------
# 4) ACF
# ------------------------------------------------------------
fig_acf = plot_acf(residuals, lags=40)
fig_acf.tight_layout()
fig_acf.savefig(OUT_PLOT / "sarima_residuals_acf.png", dpi=150)
plt.close(fig_acf)


# ------------------------------------------------------------
# 5) PACF
# ------------------------------------------------------------
fig_pacf = plot_pacf(residuals, lags=40, method="ywm")
fig_pacf.tight_layout()
fig_pacf.savefig(OUT_PLOT / "sarima_residuals_pacf.png", dpi=150)
plt.close(fig_pacf)


# ------------------------------------------------------------
# 6) ÖZET
# ------------------------------------------------------------
print(" Residual analizi tamamlandı")
print("Çıktılar:")
print(" - zs_time_series/outputs/plots/sarima_residuals.png")
print(" - zs_time_series/outputs/plots/sarima_residuals_acf.png")
print(" - zs_time_series/outputs/plots/sarima_residuals_pacf.png")
