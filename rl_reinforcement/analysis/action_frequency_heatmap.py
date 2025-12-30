# ============================================================
# Action Frequency Heatmap – PPO action distribution over (Lead, Queue)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from stable_baselines3 import PPO
from rl_reinforcement.env.production_env import ProductionLineEnv

DATA_PATH = "data/raw/production_sim.csv"
MODEL_PATH = "rl_reinforcement/outputs/ppo/policies/ppo_final"

OUT_DIR = Path("rl_reinforcement/outputs/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load
# -------------------------
env = ProductionLineEnv(DATA_PATH)
model = PPO.load(MODEL_PATH, env=env)
df = pd.read_csv(DATA_PATH)

# -------------------------
# Grid definition (bins)
# -------------------------
n_lead_bins = 12
n_queue_bins = 12

lead_edges = np.linspace(df["lead_time"].min(), df["lead_time"].max(), n_lead_bins + 1)
queue_edges = np.linspace(df["queue_length"].min(), df["queue_length"].max(), n_queue_bins + 1)

# action counts per bin
counts = np.zeros((n_queue_bins, n_lead_bins, 3), dtype=np.int64)

mean_operator = df["operator_load"].mean()
mean_machine = df["machine_status"].mean()

# Sample points per bin to estimate action frequency
samples_per_bin = 60

rng = np.random.default_rng(42)

for qi in range(n_queue_bins):
    q_low, q_high = queue_edges[qi], queue_edges[qi + 1]
    for li in range(n_lead_bins):
        l_low, l_high = lead_edges[li], lead_edges[li + 1]

        # random samples inside that bin
        qs = rng.uniform(q_low, q_high, size=samples_per_bin)
        ls = rng.uniform(l_low, l_high, size=samples_per_bin)

        for q, lt in zip(qs, ls):
            state = np.array([q, mean_operator, mean_machine, lt], dtype=np.float32)
            obs = env._normalize(state).astype(np.float32)

            # stochastic policy: true distribution feel
            action, _ = model.predict(obs, deterministic=False)
            counts[qi, li, int(action)] += 1

# convert to percentages
freq = counts / np.maximum(counts.sum(axis=2, keepdims=True), 1)

action_names = ["Decrease", "Hold", "Increase"]
for a in range(3):
    Z = freq[:, :, a]  # queue x lead

    plt.figure(figsize=(10, 6))
    plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[lead_edges[0], lead_edges[-1], queue_edges[0], queue_edges[-1]],
    )
    plt.colorbar(label=f"P(action={action_names[a]})")
    plt.xlabel("Lead Time")
    plt.ylabel("Queue Length")
    plt.title(f"PPO Action Frequency Heatmap: {action_names[a]}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ppo_action_freq_{action_names[a].lower()}.png", dpi=160)
    plt.close()

# single “dominant action” map
dominant = np.argmax(freq, axis=2)

plt.figure(figsize=(10, 6))
plt.imshow(
    dominant,
    origin="lower",
    aspect="auto",
    extent=[lead_edges[0], lead_edges[-1], queue_edges[0], queue_edges[-1]],
    vmin=0, vmax=2,
)
cbar = plt.colorbar(ticks=[0, 1, 2])
cbar.ax.set_yticklabels(action_names)
plt.xlabel("Lead Time")
plt.ylabel("Queue Length")
plt.title("PPO Dominant Action Map (by frequency)")
plt.tight_layout()
plt.savefig(OUT_DIR / "ppo_dominant_action_map.png", dpi=160)
plt.close()

print(" Action frequency heatmaps üretildi:")
print(" - rl_reinforcement/outputs/analysis/ppo_action_freq_decrease.png")
print(" - rl_reinforcement/outputs/analysis/ppo_action_freq_hold.png")
print(" - rl_reinforcement/outputs/analysis/ppo_action_freq_increase.png")
print(" - rl_reinforcement/outputs/analysis/ppo_dominant_action_map.png")
