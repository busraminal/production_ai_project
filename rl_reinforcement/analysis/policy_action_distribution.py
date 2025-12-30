import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO

from rl_reinforcement.env.production_env import ProductionLineEnv

BASE_DATA = Path("data/raw/production_sim.csv")
PPO_PATH = Path("outputs/ppo_model.zip")

PROPHET_FORE = Path("zs_time_series/outputs/forecasts/prophet_lead_time_forecast.csv")
SARIMA_FORE = Path("zs_time_series/outputs/forecasts/sarima_lead_time_forecast.csv")

OUT_DIR = Path("rl_reinforcement/analysis/outputs")
PLOT_DIR = Path("rl_reinforcement/analysis/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def make_sarima_tmp_csv():
    """Env'e dokunmadan SARIMA tahmini lead_time olarak kullanmak için geçici CSV üret."""
    df = pd.read_csv(BASE_DATA, parse_dates=["time"])
    sar = pd.read_csv(SARIMA_FORE, parse_dates=["time"])

    df = df.merge(sar[["time", "y_pred"]], on="time", how="left")
    df["lead_time"] = df["y_pred"].fillna(df["lead_time"])
    df = df.drop(columns=["y_pred"])

    tmp_path = Path("data/raw/_tmp_sarima_env.csv")
    df.to_csv(tmp_path, index=False)
    return tmp_path


def count_actions(env, model, n_episodes=20):
    counts = np.zeros(3, dtype=np.int64)

    for _ in range(n_episodes):
        obs, _ = env.reset()
        term, trunc = False, False

        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            a = int(action)
            counts[a] += 1
            obs, r, term, trunc, info = env.step(action)

    total = counts.sum()
    probs = counts / (total + 1e-8)
    return counts, probs


def main(n_episodes=20):
    model = PPO.load(PPO_PATH)

    # -----------------------------
    # 1) Plain
    # -----------------------------
    env_plain = ProductionLineEnv(BASE_DATA)
    c_plain, p_plain = count_actions(env_plain, model, n_episodes=n_episodes)

    # -----------------------------
    # 2) Prophet
    # -----------------------------
    env_prophet = ProductionLineEnv(
        BASE_DATA,
        prophet_forecast_path=PROPHET_FORE
    )
    c_prophet, p_prophet = count_actions(env_prophet, model, n_episodes=n_episodes)

    # -----------------------------
    # 3) SARIMA (tmp csv)
    # -----------------------------
    tmp = make_sarima_tmp_csv()
    env_sarima = ProductionLineEnv(tmp)
    c_sarima, p_sarima = count_actions(env_sarima, model, n_episodes=n_episodes)
    tmp.unlink(missing_ok=True)

    # -----------------------------
    # Save table
    # -----------------------------
    df = pd.DataFrame({
        "scenario": ["plain", "sarima", "prophet"],
        "action0_decrease": [p_plain[0], p_sarima[0], p_prophet[0]],
        "action1_hold":     [p_plain[1], p_sarima[1], p_prophet[1]],
        "action2_increase": [p_plain[2], p_sarima[2], p_prophet[2]],
        "steps_total":      [int(c_plain.sum()), int(c_sarima.sum()), int(c_prophet.sum())]
    })

    out_csv = OUT_DIR / "policy_action_distribution.csv"
    df.to_csv(out_csv, index=False)

    # -----------------------------
    # Plot (grouped bar)
    # -----------------------------
    labels = ["decrease(0)", "hold(1)", "increase(2)"]
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(7, 4))
    plt.bar(x - width, [p_plain[0],  p_plain[1],  p_plain[2]],  width, label="Plain")
    plt.bar(x,         [p_sarima[0], p_sarima[1], p_sarima[2]], width, label="SARIMA")
    plt.bar(x + width, [p_prophet[0],p_prophet[1],p_prophet[2]],width, label="Prophet")

    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Action Probability")
    plt.title(f"Policy Action Distribution (n_episodes={n_episodes})")
    plt.legend()
    plt.tight_layout()

    out_png = PLOT_DIR / "policy_action_distribution.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(" Policy action distribution hazır")
    print("CSV :", out_csv)
    print("PNG :", out_png)
    print(df)


if __name__ == "__main__":
    main(n_episodes=30)
