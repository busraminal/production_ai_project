from rl_reinforcement.env.production_env import ProductionLineEnv

DATA_PATH = "data/raw/production_sim.csv"

if __name__ == "__main__":
    env = ProductionLineEnv(DATA_PATH)
    obs, _ = env.reset()
    print("Initial obs:", obs)

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("reward:", reward, "info:", info)
        if terminated or truncated:
            break
