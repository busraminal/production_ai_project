import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LOG = Path("rl_reinforcement/analysis/outputs/early_warning_episode_log.csv")
OUT = Path("rl_reinforcement/analysis/plots/early_warning_episode.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LOG)

plt.figure(figsize=(10,4))
plt.plot(df["t"], df["lead_time_true"], label="Gerçek Lead Time")
plt.plot(df["t"], df["lead_time_hat"], label="Prophet Tahmini")

# aksiyonları renkli işaretle
for a, c, lab in [(0,"#1f77b4","decrease"), (1,"#2ca02c","hold"), (2,"#d62728","increase")]:
    idx = df["action"] == a
    plt.scatter(df.loc[idx,"t"], df.loc[idx,"lead_time_true"], s=12, c=c, label=f"action={lab}")

plt.xlabel("t")
plt.ylabel("Lead Time")
plt.title("Erken Uyarı: Prophet Tahmini + RL Aksiyonu")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
plt.close()

print(" Grafik üretildi:", OUT)
