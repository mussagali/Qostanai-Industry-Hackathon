from __future__ import annotations

import numpy as np
import pandas as pd


STATION_TYPES = [
    "Body", "Body", "Body", "Body", "Body", "Body",
    "Paint", "Paint", "Paint", "Paint", "Paint", "Paint",
    "Final", "Final", "Final", "Final", "Final", "Final",
    "Final", "Final", "Final", "Final", "Final", "Final",
    "Final", "Final", "Final", "Final", "Final", "Final",
]

def initial_line(n_stations: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    station_ids = [f"S{i:02d}" for i in range(1, n_stations + 1)]
    types = (STATION_TYPES * ((n_stations // len(STATION_TYPES)) + 1))[:n_stations]

    sensor_level = rng.choice(
        ["rich", "partial", "manual"],
        size=n_stations,
        p=[0.65, 0.25, 0.10],
    )

    df = pd.DataFrame({
        "station_id": station_ids,
        "station_type": types,
        "sensor_level": sensor_level,
        "cycle_time": rng.normal(48, 4, n_stations).round(1),
        "temperature": rng.normal(57, 5, n_stations).round(1),
        "vibration": rng.normal(2.3, 0.35, n_stations).round(2),
        "torque": rng.normal(45, 4, n_stations).round(1),
        "queue_length": rng.integers(0, 5, n_stations),
        "throughput": rng.normal(72, 5, n_stations).round(1),
        "downtime_min": rng.uniform(0, 2, n_stations).round(1),
    })

    return apply_sensor_gaps(df, rng)

def apply_sensor_gaps(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()

    partial = out["sensor_level"].eq("partial")
    manual = out["sensor_level"].eq("manual")

    for idx in out.index[partial]:
        missing = rng.choice(["temperature", "vibration", "torque"])
        out.loc[idx, missing] = np.nan

    for idx in out.index[manual]:
        out.loc[idx, ["temperature", "vibration", "torque"]] = np.nan

    return out

def simulate_step(
    df: pd.DataFrame,
    step: int,
    scenario: str = "Normal",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + step * 17)
    out = df.copy()

    # Base process noise
    out["cycle_time"] = (out["cycle_time"] + rng.normal(0, 0.8, len(out))).clip(35, 95).round(1)
    out["queue_length"] = (out["queue_length"] + rng.integers(-1, 2, len(out))).clip(0, 25).astype(int)
    out["throughput"] = (out["throughput"] + rng.normal(0, 1.2, len(out))).clip(35, 90).round(1)
    out["downtime_min"] = (out["downtime_min"] + rng.uniform(-0.2, 0.35, len(out))).clip(0, 12).round(1)

    for col, noise, low, high, decimals in [
        ("temperature", 0.6, 35, 100, 1),
        ("vibration", 0.10, 0.5, 9.0, 2),
        ("torque", 0.7, 20, 80, 1),
    ]:
        mask = out[col].notna()
        out.loc[mask, col] = (
            out.loc[mask, col] + rng.normal(0, noise, mask.sum())
        ).clip(low, high).round(decimals)

    if scenario == "Developing bottleneck":
        # Degrade Station 14 progressively
        idx = out.index[out["station_id"].eq("S14")]
        if len(idx):
            i = idx[0]
            severity = min(step, 12)
            out.loc[i, "cycle_time"] = min(48 + severity * 2.8, 88)
            out.loc[i, "queue_length"] = min(2 + severity * 2, 25)
            out.loc[i, "throughput"] = max(72 - severity * 2.6, 38)
            out.loc[i, "downtime_min"] = min(0.5 + severity * 0.65, 10)
            if pd.notna(out.loc[i, "vibration"]):
                out.loc[i, "vibration"] = min(2.3 + severity * 0.32, 7.2)
            if pd.notna(out.loc[i, "temperature"]):
                out.loc[i, "temperature"] = min(57 + severity * 1.5, 90)

    elif scenario == "High defect risk":
        idx = out.index[out["station_id"].eq("S07")]
        if len(idx):
            i = idx[0]
            severity = min(step, 10)
            if pd.notna(out.loc[i, "temperature"]):
                out.loc[i, "temperature"] = min(58 + severity * 2.2, 92)
            if pd.notna(out.loc[i, "vibration"]):
                out.loc[i, "vibration"] = min(2.4 + severity * 0.36, 7.0)
            if pd.notna(out.loc[i, "torque"]):
                out.loc[i, "torque"] = min(45 + severity * 2.1, 72)
            out.loc[i, "cycle_time"] = min(48 + severity * 1.4, 70)

    elif scenario == "Sensor-poor station":
        idx = out.index[out["station_id"].eq("S20")]
        if len(idx):
            i = idx[0]
            out.loc[i, "sensor_level"] = "manual"
            out.loc[i, ["temperature", "vibration", "torque"]] = np.nan
            out.loc[i, "cycle_time"] = min(52 + step * 1.2, 70)
            out.loc[i, "queue_length"] = min(4 + step, 18)

    return out

def simulate_vehicle_batch(n: int = 120, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "vehicle_id": [f"V{i:04d}" for i in range(1, n + 1)],
        "model": rng.choice(["Model-A", "Model-B", "Model-C"], size=n, p=[0.45, 0.35, 0.20]),
        "temperature": rng.normal(60, 8, n),
        "vibration": rng.normal(2.6, 0.8, n),
        "torque": rng.normal(46, 6, n),
        "cycle_time": rng.normal(50, 7, n),
    })
