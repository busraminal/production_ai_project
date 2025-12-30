# ============================================================
# Digital Twin – Hourly Production Data Generator
# ============================================================
# Bu modül:
# - Saatlik dijital ikiz verisi üretir
# - ZS ve RL için ortak veri kaynağıdır
# - Gerçek veri ile birebir şema uyumludur
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


# ------------------------------------------------------------
# 1) SENARYO PARAMETRELERİ
# ------------------------------------------------------------
SCENARIOS = {
    "normal": {
        "queue_mu": 50,
        "queue_sigma": 10,
        "operator_load_base": 0.55,
        "machine_status_base": 0.95,
    },
    "high_demand": {
        "queue_mu": 80,
        "queue_sigma": 15,
        "operator_load_base": 0.70,
        "machine_status_base": 0.92,
    },
    "fatigue": {
        "queue_mu": 65,
        "queue_sigma": 12,
        "operator_load_base": 0.80,
        "machine_status_base": 0.90,
    },
    "breakdown": {
        "queue_mu": 90,
        "queue_sigma": 20,
        "operator_load_base": 0.65,
        "machine_status_base": 0.70,
    },
}


# ------------------------------------------------------------
# 2) TEK EPISODE (24 SAAT) ÜRETİMİ
# ------------------------------------------------------------
def simulate_one_day(
    start_time: datetime,
    scenario: str,
    station_id: str = "A",
    seed: int | None = None,
) -> pd.DataFrame:
    """
    24 saatlik (1 episode) dijital ikiz simülasyonu.
    """

    if seed is not None:
        np.random.seed(seed)

    params = SCENARIOS[scenario]

    records = []
    current_queue = max(
        np.random.normal(params["queue_mu"], params["queue_sigma"]), 5
    )

    for h in range(24):
        time = start_time + timedelta(hours=h)

        # Operatör yükü (0–1)
        operator_load = np.clip(
            params["operator_load_base"] + np.random.normal(0, 0.05), 0.3, 0.95
        )

        # Makine durumu (0–1)
        machine_status = np.clip(
            params["machine_status_base"] + np.random.normal(0, 0.05), 0.4, 1.0
        )

        # Throughput (adet/saat)
        throughput = max(
            2.0 * machine_status * (1.0 - 0.6 * operator_load)
            + np.random.normal(0, 0.2),
            0.5,
        )

        # Kuyruk güncelleme
        arrivals = np.random.poisson(throughput * 1.2)
        current_queue = max(current_queue + arrivals - throughput, 0)

        # Lead time (Little’s Law benzeri)
        lead_time = max(current_queue / max(throughput, 0.5), 0.5)

        # Enerji tüketimi (kWh)
        energy_consumption = (
            1.0 + 0.8 * throughput + 0.5 * operator_load
        )

        records.append(
            {
                "time": time,
                "station_id": station_id,
                "queue_length": float(current_queue),
                "operator_load": float(operator_load),
                "machine_status": float(machine_status),
                "lead_time": float(lead_time),
                "throughput": float(throughput),
                "energy_consumption": float(energy_consumption),
                "scenario": scenario,
            }
        )

    return pd.DataFrame(records)


# ------------------------------------------------------------
# 3) ÇOK GÜNLÜK VERİ SETİ
# ------------------------------------------------------------
def generate_dataset(
    days: int = 30,
    scenario: str = "normal",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Çok günlük (days × 24 saat) veri seti üretir.
    """

    start_time = datetime(2024, 1, 1)
    all_days = []

    for d in range(days):
        day_df = simulate_one_day(
            start_time=start_time + timedelta(days=d),
            scenario=scenario,
            seed=seed + d,
        )
        all_days.append(day_df)

    return pd.concat(all_days, ignore_index=True)


# ------------------------------------------------------------
# 4) MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_PATH = Path("data/raw/production_sim.csv")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(
        days=60,          # 60 gün → ZS ve RL için yeterli
        scenario="normal" # senaryo daha sonra değiştirilebilir
    )

    df.to_csv(OUTPUT_PATH, index=False)
    print(f" Dijital ikiz veri üretildi: {OUTPUT_PATH}")
    print("Kolonlar:", list(df.columns))
