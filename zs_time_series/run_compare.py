# ============================================================
# SARIMA vs PROPHET – Model Karşılaştırma
# ============================================================

from pathlib import Path
import json
import pandas as pd


# ------------------------------------------------------------
# 1) PATHLER
# ------------------------------------------------------------
BASE = Path("zs_time_series/outputs")
MET_DIR = BASE / "metrics"
OUT_DIR = BASE / "metrics"

SARIMA_MET = MET_DIR / "sarima_metrics.json"
PROPHET_MET = MET_DIR / "prophet_metrics.csv"


# ------------------------------------------------------------
# 2) SARIMA METRİKLER
# ------------------------------------------------------------
with open(SARIMA_MET, "r", encoding="utf-8") as f:
    sarima = json.load(f)

sarima_row = {
    "model": "SARIMA",
    "MAPE": sarima["MAPE"],
    "RMSE": sarima["RMSE"]
}


# ------------------------------------------------------------
# 3) PROPHET METRİKLER
# ------------------------------------------------------------
prophet_df = pd.read_csv(PROPHET_MET)
prophet_row = {
    "model": "Prophet",
    "MAPE": prophet_df.loc[0, "MAPE"],
    "RMSE": prophet_df.loc[0, "RMSE"]
}


# ------------------------------------------------------------
# 4) KARŞILAŞTIRMA TABLOSU
# ------------------------------------------------------------
compare_df = pd.DataFrame([sarima_row, prophet_row])

compare_df.to_csv(
    OUT_DIR / "model_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 5) ÖZET
# ------------------------------------------------------------
print(" MODEL KARŞILAŞTIRMASI TAMAMLANDI")
print(compare_df)
print("Çıktı:")
print(" - zs_time_series/outputs/metrics/model_comparison.csv")
