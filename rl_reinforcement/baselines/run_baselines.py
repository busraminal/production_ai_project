# ============================================================
# Baseline Policies: Random & Greedy
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

from rl_reinforcement.env.production_env import ProductionLineEnv


DATA_PATH = "data/raw/production_sim.csv"
OUT_DIR = Path("rl_reinforcement/outputs/baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODES = 50
LEAD_THRESHOLD = 0.6  # normalize lead_time eşiği


# ------------------------------------------------------------
# Yardımcı: episode çalıştır
# ------------------------------------------------------------
def run_episode(env, policy_fn):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    return total_reward, steps


# ------------------------------------------------------------
# Random Policy
# ------------------------------------------------------------
def random_policy(obs):
    return np.random.randint(0, 3)


# ------------------------------------------------------------
# Greedy Policy (heuristic)
# lead_time yüksek → hız artır
# ------------------------------------------------------------
def greedy_policy(obs):
    lead_time = obs[3]  # normalized
    if lead_time > LEAD_THRESHOLD:
        return 2  # increase
    elif lead_time < 0.3:
        return 0  # decrease
    else:
        return 1  # hold


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
env = ProductionLineEnv(DATA_PATH)

results = []

for name, policy in [
    ("random", random_policy),
    ("greedy", greedy_policy),
]:
    rewards = []
    lengths = []

    for _ in range(EPISODES):
        r, l = run_episode(env, policy)
        rewards.append(r)
        lengths.append(l)

    results.append({
        "policy": name,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_episode_length": float(np.mean(lengths)),
    })

df_results = pd.DataFrame(results)
df_results.to_csv(OUT_DIR / "baseline_results.csv", index=False)

print(" Baseline değerlendirmesi tamamlandı")
print(df_results)
