# ============================================================
# PPO Training Script
# ============================================================

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from rl_reinforcement.env.production_env import ProductionLineEnv


# ------------------------------------------------------------
# 1) PATHS & PARAMS
# ------------------------------------------------------------
DATA_PATH = "data/raw/production_sim.csv"

OUT_BASE = Path("rl_reinforcement/outputs/ppo")
OUT_POL  = OUT_BASE / "policies"
OUT_MET  = OUT_BASE / "metrics"
OUT_PLOT = OUT_BASE / "plots"

for p in [OUT_POL, OUT_MET, OUT_PLOT]:
    p.mkdir(parents=True, exist_ok=True)

TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 5_000


# ------------------------------------------------------------
# 2) ENVIRONMENTS
# ------------------------------------------------------------
env = ProductionLineEnv(DATA_PATH)
eval_env = ProductionLineEnv(DATA_PATH)


# ------------------------------------------------------------
# 3) PPO MODEL
# ------------------------------------------------------------
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,               # daha uzun rollout
    batch_size=256,
    gamma=0.99,                       # kısa episode için ideal
    gae_lambda=0.95,     # <-- dalgalanmayı azaltır
    ent_coef=0.08,              # keşfi artır 
    clip_range=0.2,
    verbose=1,
)


# ------------------------------------------------------------
# 4) CALLBACK (Evaluation)
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
# 5) TRAIN
# ------------------------------------------------------------
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback
)


# ------------------------------------------------------------
# 6) SAVE FINAL POLICY
# ------------------------------------------------------------
FINAL_MODEL_PATH = OUT_POL / "ppo_final"
model.save(str(FINAL_MODEL_PATH))


# ------------------------------------------------------------
# 7) EVALUATE TRAINED POLICY
# ------------------------------------------------------------
def run_episode(env, model):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    return total_reward, steps


EPISODES = 30
rewards = []

for _ in range(EPISODES):
    r, l = run_episode(env, model)
    rewards.append(r)

metrics = {
    "algorithm": "PPO",
    "avg_reward": float(np.mean(rewards)),
    "std_reward": float(np.std(rewards)),
    "episodes": EPISODES,
    "timesteps": TOTAL_TIMESTEPS,
}

with open(OUT_MET / "ppo_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------
# 8) PLOT
# ------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(rewards, marker="o")
plt.title("PPO Evaluation – Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.tight_layout()
plt.savefig(OUT_PLOT / "ppo_episode_rewards.png", dpi=150)
plt.close()


# ------------------------------------------------------------
# 9) SUMMARY
# ------------------------------------------------------------
print(" PPO eğitimi tamamlandı")
print("Ortalama ödül:", metrics["avg_reward"])
print("Çıktılar:")
print(" - Policy  :", OUT_POL)
print(" - Metrics :", OUT_MET / "ppo_metrics.json")
print(" - Plot    :", OUT_PLOT / "ppo_episode_rewards.png")

model.save("rl_reinforcement/outputs/ppo/policies/ppo_model")
