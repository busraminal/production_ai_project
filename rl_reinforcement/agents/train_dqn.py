# ============================================================
# DQN Training Script
# ============================================================

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from rl_reinforcement.env.production_env import ProductionLineEnv


# ------------------------------------------------------------
# PATHS & PARAMS
# ------------------------------------------------------------
DATA_PATH = "data/raw/production_sim.csv"

OUT_BASE = Path("rl_reinforcement/outputs/dqn")
OUT_POL  = OUT_BASE / "policies"
OUT_MET  = OUT_BASE / "metrics"
OUT_PLOT = OUT_BASE / "plots"
for p in [OUT_POL, OUT_MET, OUT_PLOT]:
    p.mkdir(parents=True, exist_ok=True)

TOTAL_TIMESTEPS = 80_000
EVAL_FREQ = 8_000


# ------------------------------------------------------------
# ENV
# ------------------------------------------------------------
env = ProductionLineEnv(DATA_PATH)
eval_env = ProductionLineEnv(DATA_PATH)


# ------------------------------------------------------------
# DQN MODEL
# ------------------------------------------------------------
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=2_000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=1_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=2,
)


# ------------------------------------------------------------
# CALLBACK
# ------------------------------------------------------------
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(OUT_POL),
    log_path=str(OUT_MET),
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
)


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback
)

model.save(str(OUT_POL / "dqn_final"))


# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------
def run_episode(env, model):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return total_reward


EPISODES = 30
rewards = [run_episode(env, model) for _ in range(EPISODES)]

metrics = {
    "algorithm": "DQN",
    "avg_reward": float(np.mean(rewards)),
    "std_reward": float(np.std(rewards)),
    "episodes": EPISODES,
    "timesteps": TOTAL_TIMESTEPS,
}

with open(OUT_MET / "dqn_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

plt.figure(figsize=(8,4))
plt.plot(rewards, marker="o")
plt.title("DQN Evaluation – Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.tight_layout()
plt.savefig(OUT_PLOT / "dqn_episode_rewards.png", dpi=150)
plt.close()

print(" DQN eğitimi tamamlandı")
print("Ortalama ödül:", metrics["avg_reward"])
