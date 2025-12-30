from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

# ------------------------------------------------------------
# OPTIONAL: Prophet forecast loader
# ------------------------------------------------------------
try:
    from zs_time_series.load_prophet_forecast import ProphetLeadTimeLoader
except Exception:
    ProphetLeadTimeLoader = None


# ============================================================
# Reward Shaping (STABLE PPO)
# ============================================================
ALPHA = 6.0
BETA = 1.3

HOLD_PENALTY = 0.35
SWITCH_PENALTY = 0.15

QUEUE_TARGET_RAW = 50.0
QUEUE_PENALTY_W = 0.15

LOW_QUEUE_INC_THRESHOLD = 30.0
HIGH_QUEUE_THRESHOLD = 65.0

EPISODE_LENGTH = 24


class ProductionLineEnv(gym.Env):
    """
    State  : [queue_length, operator_load, machine_status, lead_time_hat] (normalized)
    Action : 0 = decrease, 1 = hold, 2 = increase
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: str | Path,
        prophet_forecast_path: str | Path | None = None,
    ):
        super().__init__()

        # --------------------------------------------------
        # Load data
        # --------------------------------------------------
        self.df = pd.read_csv(data_path, parse_dates=["time"]).reset_index(drop=True)

        self.state_cols = [
            "queue_length",
            "operator_load",
            "machine_status",
            "lead_time",
        ]

        # normalization bounds
        self.state_min = self.df[self.state_cols].min().values.astype(np.float32)
        self.state_max = self.df[self.state_cols].max().values.astype(np.float32)

        self.queue_min = float(self.state_min[0])
        self.queue_max = float(self.state_max[0])
        self.lead_min = float(self.state_min[3])
        self.lead_max = float(self.state_max[3])

        self.throughput_max = float(self.df["throughput"].max())
        self.energy_max = float(self.df["energy_consumption"].max())

        # --------------------------------------------------
        # Spaces
        # --------------------------------------------------
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # --------------------------------------------------
        # Episode state
        # --------------------------------------------------
        self.current_step = 0
        self.episode_step = 0
        self.prev_action = 1

        self._queue = None
        self._lead = None  # lead_time_hat

        # --------------------------------------------------
        # Prophet forecast (optional)
        # --------------------------------------------------
        self.prophet = None
        if prophet_forecast_path is not None and ProphetLeadTimeLoader is not None:
            self.prophet = ProphetLeadTimeLoader(prophet_forecast_path)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _normalize(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_min) / (self.state_max - self.state_min + 1e-8)

    def _get_obs(self):
        row = self.df.loc[self.current_step]
        op = float(row["operator_load"])
        ms = float(row["machine_status"])

        state = np.array(
            [self._queue, op, ms, self._lead], dtype=np.float32
        )
        return self._normalize(state).astype(np.float32)

    # ---------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.df) - EPISODE_LENGTH - 1
        self.current_step = int(self.np_random.integers(0, max_start))
        self.episode_step = 0
        self.prev_action = 1

        row = self.df.loc[self.current_step]
        self._queue = float(row["queue_length"])

        # -------- LEAD TIME INITIALIZATION --------
        if self.prophet is not None:
            ts = row["time"]
            lead_hat = self.prophet.get(ts)
            self._lead = (
                float(lead_hat)
                if lead_hat is not None
                else float(row["lead_time"])
            )
        else:
            self._lead = float(row["lead_time"])

        return self._get_obs(), {}

    def step(self, action: int):
        # -------- END OF DATA GUARD --------
        if self.current_step >= len(self.df) - 1:
            obs = self._get_obs()
            return obs, 0.0, True, False, {"reason": "eof"}

        action = int(action)
        row = self.df.loc[self.current_step]

        # exogenous
        op = float(row["operator_load"])
        ms = float(row["machine_status"])
        base_throughput = float(row["throughput"])
        base_energy = float(row["energy_consumption"])

        # internal
        queue = float(self._queue)
        lead = float(self._lead)

        throughput = base_throughput
        energy = base_energy
        delay = lead

        # ---------------- Dynamics ----------------
        if action == 0:  # decrease
            throughput *= 0.90
            energy *= 0.92
            delay *= 1.10
            queue += 3.0

        elif action == 2:  # increase
            throughput *= 1.10
            energy *= 1.22
            delay *= 0.93
            queue -= 5.0

            if queue < LOW_QUEUE_INC_THRESHOLD:
                energy *= 1.20
                delay *= 1.12

        # clip
        queue = float(np.clip(queue, self.queue_min, self.queue_max))
        delay = float(np.clip(delay, self.lead_min, self.lead_max))

        self._queue = queue
        self._lead = delay

        # ---------------- Normalization ----------------
        delay_n = delay / (self.lead_max + 1e-8)
        queue_n = queue / (self.queue_max + 1e-8)
        throughput_n = throughput / (self.throughput_max + 1e-8)
        energy_n = energy / (self.energy_max + 1e-8)

        # ---------------- Reward ----------------
        hold_penalty = HOLD_PENALTY if action == 1 else 0.0
        switch_penalty = SWITCH_PENALTY if action != self.prev_action else 0.0

        queue_penalty = QUEUE_PENALTY_W * max(
            queue - QUEUE_TARGET_RAW, 0.0
        ) / (self.queue_max + 1e-8)

        increase_penalty = 0.0
        if action == 2:
            if queue < LOW_QUEUE_INC_THRESHOLD:
                increase_penalty = 1.5
            elif queue < QUEUE_TARGET_RAW:
                increase_penalty = 0.6
            else:
                increase_penalty = 0.2

        hold_bonus = 0.0
        if action == 1 and abs(queue - QUEUE_TARGET_RAW) < 5.0:
            hold_bonus = 2.0

        increase_bonus = 0.0
        if action == 2 and (queue > HIGH_QUEUE_THRESHOLD or delay_n > 0.65):
            increase_bonus = 0.8

        decrease_bonus = 0.0
        if action == 0 and queue < 30 and delay_n < 0.35:
            decrease_bonus = 0.8

        reward = (
            -1.4 * delay_n
            + ALPHA * throughput_n
            - BETA * energy_n
            - hold_penalty
            - switch_penalty
            - queue_penalty
            - increase_penalty
            + hold_bonus
            + increase_bonus
            + decrease_bonus
        )

        self.prev_action = action
        self.current_step += 1
        self.episode_step += 1

        terminated = False
        truncated = self.episode_step >= EPISODE_LENGTH

        obs = (
            self._get_obs()
            if not (terminated or truncated)
            else np.zeros(4, dtype=np.float32)
        )

        info = {
            "delay_n": delay_n,
            "queue_n": queue_n,
            "throughput_n": throughput_n,
            "energy_n": energy_n,
            "action": action,
            "reward": float(reward),
            "operator_load": op,
            "machine_status": ms,
            "lead_hat_used": self.prophet is not None,
            "lead_time_true": float(row["lead_time"]),   # ham veri
            "lead_time_hat": float(self._lead),           # Prophet / SARIMA sonrası kullanılan lead
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(
            f"t={self.current_step} | queue={self._queue:.1f} | lead_hat={self._lead:.2f}"
        )
