# ============================================================
# PPO Action Probability Maps (Entropy-Aware Explainability)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from stable_baselines3 import PPO
from rl_reinforcement.env.production_env import ProductionLineEnv

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DATA_PATH = "data/raw/production_sim.csv"
MODEL_PATH = "rl_reinforcement/outputs/ppo/policies/ppo_final"
OUT_DIR = Path("rl_reinforcement/outputs/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
env = ProductionLineEnv(DATA_PATH)
model = PPO.load(MODEL_PATH, env=env)
df = pd.read_csv(DATA_PATH)

# Grid
lead_vals = np.linspace(df["lead_time"].min(), df["lead_time"].max(), 35)
queue_vals = np.linspace(df["queue_length"].min(), df["queue_length"].max(), 35)

# ------------------------------------------------------------
# Scenario (NORMAL – yeterli)
# ------------------------------------------------------------
op = float(df["operator_load"].mean())
ms = float(df["machine_status"].mean())

# ------------------------------------------------------------
# Probability tensors
# ------------------------------------------------------------
P_dec = np.zeros((len(queue_vals), len(lead_vals)))
P_hold = np.zeros_like(P_dec)
P_inc = np.zeros_like(P_dec)

# ------------------------------------------------------------
# Compute probabilities
# ------------------------------------------------------------
for i, q in enumerate(queue_vals):
    for j, lt in enumerate(lead_vals):
        raw_state = np.array([q, op, ms, lt], dtype=np.float32)
        obs = env._normalize(raw_state)
        obs_t = torch.tensor(obs).unsqueeze(0)

        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.cpu().numpy()[0]

        P_dec[i, j] = probs[0]
        P_hold[i, j] = probs[1]
        P_inc[i, j] = probs[2]

# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------
def plot_prob(Z, title, fname):
    plt.figure(figsize=(9, 6))
    plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[lead_vals.min(), lead_vals.max(),
                queue_vals.min(), queue_vals.max()],
        vmin=0.0, vmax=1.0,
        cmap="viridis"
    )
    plt.colorbar(label="Probability")
    plt.xlabel("Lead Time")
    plt.ylabel("Queue Length")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=160)
    plt.close()

# ------------------------------------------------------------
# Save plots
# ------------------------------------------------------------
plot_prob(P_dec, "PPO Policy – P(Decrease)", "ppo_prob_decrease.png")
plot_prob(P_hold, "PPO Policy – P(Hold)", "ppo_prob_hold.png")
plot_prob(P_inc, "PPO Policy – P(Increase)", "ppo_prob_increase.png")

print(" Action probability maps generated:")
print(" - ppo_prob_decrease.png")
print(" - ppo_prob_hold.png")
print(" - ppo_prob_increase.png")
