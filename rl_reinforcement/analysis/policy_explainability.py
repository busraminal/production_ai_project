# ============================================================
# Policy Explainability – State → Action Analysis
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from stable_baselines3 import PPO
from rl_reinforcement.env.production_env import ProductionLineEnv


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
DATA_PATH = "data/raw/production_sim.csv"
MODEL_PATH = "rl_reinforcement/outputs/ppo/policies/ppo_final"
OUT_DIR = Path("rl_reinforcement/outputs/explainability")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD ENV & MODEL
# ------------------------------------------------------------
env = ProductionLineEnv(DATA_PATH)
model = PPO.load(MODEL_PATH, env=env)


# ------------------------------------------------------------
# STATE GRID (only lead_time varies)
# other states fixed at mean
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

means = df[[
    "queue_length",
    "operator_load",
    "machine_status",
    "lead_time"
]].mean().values

lead_vals = np.linspace(
    df["lead_time"].min(),
    df["lead_time"].max(),
    100
)

actions = []

for lt in lead_vals:
    state = means.copy()
    state[3] = lt  # lead_time index

    # normalize using env logic
    norm_state = env._normalize(state)
    action, _ = model.predict(norm_state, deterministic=False)
    actions.append(action)


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(lead_vals, actions, marker="o", linestyle="-")
plt.yticks([0, 1, 2], ["Decrease", "Hold", "Increase"])
plt.xlabel("Lead Time")
plt.ylabel("Selected Action")
plt.title("PPO Policy Explainability: Lead Time → Action")
plt.tight_layout()
plt.savefig(OUT_DIR / "ppo_policy_explainability.png", dpi=150)
plt.close()


print(" Explainability çıktısı üretildi")
print(" - ppo_policy_explainability.png")
