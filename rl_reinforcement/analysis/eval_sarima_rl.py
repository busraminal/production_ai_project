import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from pathlib import Path

from rl_reinforcement.env.production_env import ProductionLineEnv

# -------------------------------------------------
# 1) Verileri yükle
# -------------------------------------------------
BASE_DATA = Path("data/raw/production_sim.csv")
SARIMA_FORECAST = Path(
    "zs_time_series/outputs/forecasts/sarima_lead_time_forecast.csv"
)

df = pd.read_csv(BASE_DATA, parse_dates=["time"])
sarima_df = pd.read_csv(SARIMA_FORECAST, parse_dates=["time"])

# -------------------------------------------------
# 2) lead_time → SARIMA tahmini ile değiştir
# -------------------------------------------------
df = df.merge(
    sarima_df[["time", "y_pred"]],
    on="time",
    how="left"
)

# SARIMA tahmini olmayan yerlerde gerçek değer
df["lead_time"] = df["y_pred"].fillna(df["lead_time"])
df = df.drop(columns=["y_pred"])

TMP_PATH = Path("data/raw/_tmp_sarima_env.csv")
df.to_csv(TMP_PATH, index=False)

# -------------------------------------------------
# 3) Ortam + model
# -------------------------------------------------
env = ProductionLineEnv(data_path=TMP_PATH)
model = PPO.load("outputs/ppo_model.zip", env=env)

# -------------------------------------------------
# 4) Episode evaluation
# -------------------------------------------------
def evaluate(n_episodes=5):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, trunc = False, False
        total_r = 0.0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(action)
            total_r += r

        returns.append(total_r)

    return returns, float(np.mean(returns))


rets, avg = evaluate()

print("SARIMA + RL Episode returns:", rets)
print("SARIMA + RL AVG return:", avg)

# -------------------------------------------------
# 5) Temizlik
# -------------------------------------------------
TMP_PATH.unlink(missing_ok=True)
