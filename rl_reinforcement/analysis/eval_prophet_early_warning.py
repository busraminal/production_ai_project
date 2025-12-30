import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

from rl_reinforcement.env.production_env import ProductionLineEnv

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DATA = Path("data/raw/production_sim.csv")
PROPHET_FORE = Path("zs_time_series/outputs/forecasts/prophet_lead_time_forecast.csv")

OUT_CSV = Path("rl_reinforcement/analysis/outputs/early_warning_episode_log.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("outputs/ppo_model.zip")

# ------------------------------------------------------------
# ENV
# ------------------------------------------------------------
env = ProductionLineEnv(
    data_path=BASE_DATA,
    prophet_forecast_path=PROPHET_FORE
)

model = PPO.load(MODEL_PATH, env=env)

# ------------------------------------------------------------
# EPISODE LOG
# ------------------------------------------------------------
rows = []

obs, info = env.reset()
done = False
truncated = False
t = 0

while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    rows.append({
        "t": t,
        "lead_time_true": info["lead_time_true"],
        "lead_time_hat": info["lead_time_hat"],
        "action": int(action),
        "reward": float(reward),
    })

    t += 1

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(" Episode log yazıldı:", OUT_CSV)
print(df.head())
print("t unique:", df["t"].unique())
