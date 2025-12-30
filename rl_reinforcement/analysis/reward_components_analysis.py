# ============================================================
# Reward Components Analysis (NORMALIZED - FIXED)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from rl_reinforcement.env.production_env import ProductionLineEnv

DATA_PATH = "data/raw/production_sim.csv"
MODEL_PATH = "rl_reinforcement/outputs/ppo/policies/ppo_final"

# Reward weights (env ile birebir)
ALPHA = 8.0
BETA  = 1.2
QUEUE_TARGET_N = 0.5       # normalize ~50
QUEUE_PENALTY_W = 0.2

# ------------------------------------------------------------
# Load env + model
# ------------------------------------------------------------
env = ProductionLineEnv(DATA_PATH)
model = PPO.load(MODEL_PATH, env=env)

# ------------------------------------------------------------
# Run ONE episode
# ------------------------------------------------------------
obs, _ = env.reset()
records = []

for _ in range(24):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, _, truncated, info = env.step(action)
    records.append(info)
    if truncated:
        break

df = pd.DataFrame(records)
print("COLUMNS:", df.columns.tolist())

# ------------------------------------------------------------
# Reward components (NORMALIZED SPACE)
# ------------------------------------------------------------
components = {
    "delay": -df["delay_n"].mean(),
    "throughput": ALPHA * df["throughput_n"].mean(),
    "energy": -BETA * df["energy_n"].mean(),
}

if "queue_n" in df.columns:
    components["queue_penalty"] = -QUEUE_PENALTY_W * np.mean(
        np.maximum(df["queue_n"] - QUEUE_TARGET_N, 0.0)
    )

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.bar(components.keys(), components.values())
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Mean Reward Component Contributions (Normalized)")
plt.ylabel("Contribution")
plt.tight_layout()
plt.show()
