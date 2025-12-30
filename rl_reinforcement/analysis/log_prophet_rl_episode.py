import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PATH
# ============================================================
CSV_PATH = Path(
    "rl_reinforcement/analysis/outputs/early_warning_episode_log.csv"
)
assert CSV_PATH.exists(), "CSV bulunamadı!"

df = pd.read_csv(CSV_PATH)

# ============================================================
# X AXIS (KRİTİK KISIM)
# ============================================================
t = df["t"].values              # <-- BURASI ÖNEMLİ
y_true = df["lead_time_true"]
y_hat  = df["lead_time_hat"]
actions = df["action"]

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(12, 5))

# Lead time çizgileri
plt.plot(t, y_true, label="Gerçek Lead Time", linewidth=2)
plt.plot(t, y_hat, label="Prophet Tahmini", linewidth=2)

# Aksiyon scatter
colors = {0: "blue", 1: "green", 2: "red"}
labels = {0: "action=decrease", 1: "action=hold", 2: "action=increase"}

for a in [0, 1, 2]:
    mask = actions == a
    plt.scatter(
        t[mask],
        y_hat[mask],
        color=colors[a],
        label=labels[a],
        s=60,
        alpha=0.7
    )

plt.title("Erken Uyarı: Prophet Tahmini + RL Aksiyonu")
plt.xlabel("t (episode step)")
plt.ylabel("Lead Time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

OUT = Path("rl_reinforcement/analysis/plots/early_warning_episode.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150)
plt.show()

print(" Grafik üretildi:", OUT)
