# ============================================================
# Scenario Explainability – PPO Policy Maps (FINAL & SAFE)
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
# Load env + model
# ------------------------------------------------------------
env = ProductionLineEnv(DATA_PATH)
model = PPO.load(MODEL_PATH, env=env)

df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# Grid (RAW values)
# ------------------------------------------------------------
lead_vals = np.linspace(df["lead_time"].min(), df["lead_time"].max(), 40)
queue_vals = np.linspace(df["queue_length"].min(), df["queue_length"].max(), 40)

# ------------------------------------------------------------
# Scenario definitions
# ------------------------------------------------------------
def scenario_params(name: str):
    op_mean = float(df["operator_load"].mean())
    ms_mean = float(df["machine_status"].mean())

    if name == "normal":
        return op_mean, ms_mean
    if name == "high_congestion":
        return min(op_mean + 0.12, 0.95), max(ms_mean - 0.08, 0.65)
    if name == "machine_degradation":
        return op_mean, max(ms_mean - 0.18, 0.55)

    raise ValueError("Unknown scenario")

# ------------------------------------------------------------
# Build policy map
# ------------------------------------------------------------
def build_action_map(operator_load, machine_status):
    Z = np.zeros((len(queue_vals), len(lead_vals)), dtype=int)

    for i, q in enumerate(queue_vals):
        for j, lt in enumerate(lead_vals):
            raw_state = np.array(
                [q, operator_load, machine_status, lt],
                dtype=np.float32,
            )

            obs = env._normalize(raw_state).reshape(1, -1)
            action, _ = model.predict(obs, deterministic=True)
            Z[i, j] = int(action)

    return Z

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
scenarios = ["normal", "high_congestion", "machine_degradation"]

QUEUE_THRESHOLD = 65.0
LEAD_THRESHOLD = 0.6 * df["lead_time"].max()

for sc in scenarios:
    op, ms = scenario_params(sc)
    Z = build_action_map(op, ms)

    plt.figure(figsize=(9, 6))

    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[
            lead_vals.min(), lead_vals.max(),
            queue_vals.min(), queue_vals.max(),
        ],
        vmin=0,
        vmax=2,
        cmap="viridis",
    )

    # ---------- Threshold çizgileri ----------
    plt.axhline(
        y=QUEUE_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Queue Threshold (65)",
    )

    plt.axvline(
        x=LEAD_THRESHOLD,
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Lead Time Threshold",
    )

    # ---------- Colorbar ----------
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Decrease", "Hold", "Increase"])

    plt.xlabel("Lead Time")
    plt.ylabel("Queue Length")
    plt.title(
        f"PPO Policy Map – {sc.replace('_',' ').title()}\n"
        f"operator={op:.2f}, machine={ms:.2f}"
    )

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ppo_policy_map_{sc}.png", dpi=160)
    plt.close()

    dominant = np.bincount(Z.flatten()).argmax()
    print(f"[{sc}] Dominant action:",
          ["Decrease", "Hold", "Increase"][dominant])

print("\nScenario policy maps generated:")
for sc in scenarios:
    print(f" - {OUT_DIR / f'ppo_policy_map_{sc}.png'}")
