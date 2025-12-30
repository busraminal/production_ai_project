# ============================================================
# PPO Policy – Scenario Based Explainability Maps
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
OUT_DIR = Path("rl_reinforcement/outputs/explainability/scenarios")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
env = ProductionLineEnv(DATA_PATH)
model = PPO.load(MODEL_PATH, env=env)
df = pd.read_csv(DATA_PATH)


# ------------------------------------------------------------
# GRID (Lead × Queue)
# ------------------------------------------------------------
lead_vals = np.linspace(df["lead_time"].min(), df["lead_time"].max(), 50)
queue_vals = np.linspace(df["queue_length"].min(), df["queue_length"].max(), 50)


# ------------------------------------------------------------
# SCENARIOS
# ------------------------------------------------------------
SCENARIOS = {
    "low_congestion": {
        "operator_load": df["operator_load"].quantile(0.25),
        "machine_status": df["machine_status"].quantile(0.25),
        "title": "Low Congestion"
    },
    "normal_operation": {
        "operator_load": df["operator_load"].mean(),
        "machine_status": df["machine_status"].mean(),
        "title": "Normal Operation"
    },
    "high_congestion": {
        "operator_load": df["operator_load"].quantile(0.90),
        "machine_status": df["machine_status"].quantile(0.95),
        "title": "High Congestion"
    }
}


# ------------------------------------------------------------
# GENERATE MAPS
# ------------------------------------------------------------
for key, sc in SCENARIOS.items():

    Z = np.zeros((len(queue_vals), len(lead_vals)))

    for i, q in enumerate(queue_vals):
        for j, lt in enumerate(lead_vals):
            state = np.array([
                q,
                sc["operator_load"],
                sc["machine_status"],
                lt
            ], dtype=np.float32)

            obs = env._normalize(state)
            action, _ = model.predict(obs, deterministic=True)
            Z[i, j] = action

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[
            lead_vals.min(),
            lead_vals.max(),
            queue_vals.min(),
            queue_vals.max()
        ],
        cmap="viridis"
    )

    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Decrease", "Hold", "Increase"])

    plt.xlabel("Lead Time")
    plt.ylabel("Queue Length")
    plt.title(
        f"PPO Policy Scenario Map (Deterministic)\n"
        f"{sc['title']} | op={sc['operator_load']:.2f}, machine={sc['machine_status']:.2f}"
    )

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ppo_policy_map_{key}.png", dpi=150)
    plt.close()

    print(f" Scenario generated: {key}")

print("\nTüm senaryo policy map'leri üretildi.")
