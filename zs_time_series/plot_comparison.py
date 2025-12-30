# ============================================================
# SARIMA vs PROPHET – Görsel Karşılaştırma
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# 1) VERİYİ YÜKLE
# ------------------------------------------------------------
DATA_PATH = Path("zs_time_series/outputs/metrics/model_comparison.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError("Önce run_compare.py çalıştırılmalı")

df = pd.read_csv(DATA_PATH)


# ------------------------------------------------------------
# 2) MAPE GRAFİĞİ
# ------------------------------------------------------------
plt.figure(figsize=(5, 4))
plt.bar(df["model"], df["MAPE"])
plt.title("Model Karşılaştırması – MAPE")
plt.ylabel("MAPE")
plt.tight_layout()
plt.savefig(
    "zs_time_series/outputs/plots/mape_comparison.png",
    dpi=150
)
plt.close()


# ------------------------------------------------------------
# 3) RMSE GRAFİĞİ
# ------------------------------------------------------------
plt.figure(figsize=(5, 4))
plt.bar(df["model"], df["RMSE"])
plt.title("Model Karşılaştırması – RMSE")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig(
    "zs_time_series/outputs/plots/rmse_comparison.png",
    dpi=150
)
plt.close()


# ------------------------------------------------------------
# 4) ÖZET
# ------------------------------------------------------------
print(" KARŞILAŞTIRMA GRAFİKLERİ OLUŞTURULDU")
print("Çıktılar:")
print(" - zs_time_series/outputs/plots/mape_comparison.png")
print(" - zs_time_series/outputs/plots/rmse_comparison.png")
