# ============================================================
# 2D Policy Explainability – Lead Time x Queue Length
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
df = pd.read_csv(DATA_PATH)

# sabit tutulan değişkenler
mean_operator = float(df["operator_load"].mean())
mean_machine  = float(df["machine_status"].mean())

# ------------------------------------------------------------
# GRID (Lead Time × Queue Length)
# ------------------------------------------------------------
lead_vals = np.linspace(df["lead_time"].min(), df["lead_time"].max(), 40)
queue_vals = np.linspace(df["queue_length"].min(), df["queue_length"].max(), 40)

Z = np.zeros((len(queue_vals), len(lead_vals)), dtype=np.int32)

for i, q in enumerate(queue_vals):
    for j, lt in enumerate(lead_vals):

        # -------- RAW STATE (q ve lt grid'den geliyor) --------
        raw_state = np.array(
            [q, mean_operator, mean_machine, lt],
            dtype=np.float32
        )

        # -------- NORMALIZE --------
        obs = env._normalize(raw_state).astype(np.float32)

        # SB3: (n_env, obs_dim) bekler
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)

        Z[i, j] = int(action[0])

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))

im = plt.imshow(
    Z,
    origin="lower",
    aspect="auto",
    extent=[lead_vals.min(), lead_vals.max(), queue_vals.min(), queue_vals.max()],
    vmin=0,
    vmax=2,
    cmap="viridis",
)

cbar = plt.colorbar(ticks=[0, 1, 2])
cbar.ax.set_yticklabels(["Decrease", "Hold", "Increase"])

plt.xlabel("Lead Time")
plt.ylabel("Queue Length")
plt.title("PPO Policy Explainability (2D)\nLead Time × Queue Length")

plt.tight_layout()
plt.savefig(OUT_DIR / "ppo_policy_explainability_2d.png", dpi=150)
plt.close()

print("2D Explainability üretildi")
print("- rl_reinforcement/outputs/explainability/ppo_policy_explainability_2d.png")
