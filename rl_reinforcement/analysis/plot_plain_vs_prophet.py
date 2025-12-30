# ============================================================
# Plain RL vs Prophet-Integrated RL – Bar Chart
# ============================================================

import matplotlib.pyplot as plt

# Ortalama return değerleri (deney çıktılarından)
labels = ["Plain RL", "Prophet + RL"]
avg_returns = [70.4985, 76.0897]

# Grafik
plt.figure(figsize=(5, 4))
plt.bar(labels, avg_returns)
plt.ylabel("Average Episode Return")
plt.title("Plain RL vs Prophet-Integrated RL")

# Değerleri bar üstüne yaz
for i, v in enumerate(avg_returns):
    plt.text(i, v + 0.5, f"{v:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(
    "rl_reinforcement/analysis/plain_vs_prophet_rl.png",
    dpi=150
)
plt.close()

print(" Plain vs Prophet RL bar chart oluşturuldu")
print("Çıktı:")
print(" - rl_reinforcement/analysis/plain_vs_prophet_rl.png")
