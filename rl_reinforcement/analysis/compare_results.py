# ============================================================
# PPO vs DQN vs Baseline Comparison
# ============================================================

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# 1) PATHS
# ------------------------------------------------------------
BASELINE_PATH = Path("rl_reinforcement/outputs/baselines/baseline_results.csv")
PPO_PATH = Path("rl_reinforcement/outputs/ppo/metrics/ppo_metrics.json")
DQN_PATH = Path("rl_reinforcement/outputs/dqn/metrics/dqn_metrics.json")

OUT_DIR = Path("rl_reinforcement/outputs/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2) LOAD RESULTS
# ------------------------------------------------------------
baseline_df = pd.read_csv(BASELINE_PATH)

with open(PPO_PATH, "r", encoding="utf-8") as f:
    ppo = json.load(f)

with open(DQN_PATH, "r", encoding="utf-8") as f:
    dqn = json.load(f)

ppo_row = {
    "policy": "ppo",
    "avg_reward": ppo["avg_reward"],
    "std_reward": ppo["std_reward"],
    "avg_episode_length": 24
}

dqn_row = {
    "policy": "dqn",
    "avg_reward": dqn["avg_reward"],
    "std_reward": dqn["std_reward"],
    "avg_episode_length": 24
}

comparison_df = pd.concat(
    [baseline_df, pd.DataFrame([ppo_row, dqn_row])],
    ignore_index=True
)


# ------------------------------------------------------------
# 3) SAVE CSV
# ------------------------------------------------------------
comparison_df.to_csv(
    OUT_DIR / "comparison_results.csv",
    index=False
)


# ------------------------------------------------------------
# 4) PLOT
# ------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.bar(
    comparison_df["policy"],
    comparison_df["avg_reward"]
)
plt.axhline(
    comparison_df.loc[comparison_df["policy"] == "greedy", "avg_reward"].values[0],
    linestyle="--",
    label="Greedy Baseline"
)

plt.title("Policy Comparison – Average Reward")
plt.ylabel("Average Reward")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "policy_comparison.png", dpi=150)
plt.close()


# ------------------------------------------------------------
# 5) SUMMARY
# ------------------------------------------------------------
print(" Karşılaştırma tamamlandı")
print(comparison_df)
print("Çıktılar:")
print(" - comparison_results.csv")
print(" - policy_comparison.png")
