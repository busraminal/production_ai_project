# ============================================================
# Prophet Lead Time Forecast Loader
# ============================================================

import pandas as pd
from pathlib import Path


class ProphetLeadTimeLoader:
    def __init__(self, csv_path: str | Path):
        self.df = pd.read_csv(csv_path, parse_dates=["ds"])
        self.df = self.df.sort_values("ds").reset_index(drop=True)
        self.df.set_index("ds", inplace=True)

    def get(self, timestamp):
        """
        timestamp: pd.Timestamp
        returns: float lead_time_hat
        """
        if timestamp in self.df.index:
            return float(self.df.loc[timestamp, "yhat"])
        else:
            # fallback: nearest previous
            past = self.df[self.df.index <= timestamp]
            if len(past) > 0:
                return float(past.iloc[-1]["yhat"])
            return None
