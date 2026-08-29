from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "cycle_time",
    "temperature",
    "vibration",
    "torque",
    "queue_length",
    "throughput",
    "downtime_min",
]

def _synthetic_training_data(n: int = 3500, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({
        "cycle_time": rng.normal(49, 9, n),
        "temperature": rng.normal(59, 11, n),
        "vibration": rng.normal(2.7, 1.2, n),
        "torque": rng.normal(46, 8, n),
        "queue_length": rng.integers(0, 18, n),
        "throughput": rng.normal(69, 10, n),
        "downtime_min": rng.exponential(1.5, n),
    })

    # Inject missing values to make the model tolerant of uneven sensor coverage.
    for col in ["temperature", "vibration", "torque"]:
        miss = rng.random(n) < 0.08
        x.loc[miss, col] = np.nan

    def z(col, center, scale):
        return (col - center) / scale

    bottleneck_score = (
        1.5 * z(x["cycle_time"], 49, 8)
        + 0.14 * x["queue_length"]
        - 0.08 * (x["throughput"] - 69)
        + 0.22 * x["downtime_min"]
        + rng.normal(0, 0.8, n)
    )
    y_bottleneck = (bottleneck_score > 2.2).astype(int)

    temp = x["temperature"].fillna(59)
    vib = x["vibration"].fillna(2.7)
    tq = x["torque"].fillna(46)
    defect_score = (
        0.09 * (temp - 60)
        + 0.9 * (vib - 2.5)
        + 0.08 * (tq - 48)
        + 0.025 * (x["cycle_time"] - 50)
        + rng.normal(0, 0.7, n)
    )
    y_defect = (defect_score > 1.7).astype(int)
    return x, y_bottleneck, y_defect

def train_models(seed: int = 7):
    x, y_b, y_d = _synthetic_training_data(seed=seed)

    def make_model():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=180,
                max_depth=8,
                class_weight="balanced",
                random_state=seed,
            )),
        ])

    bottleneck_model = make_model()
    defect_model = make_model()
    bottleneck_model.fit(x[FEATURES], y_b)
    defect_model.fit(x[FEATURES], y_d)

    anomaly_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=160,
            contamination=0.08,
            random_state=seed,
        )),
    ])
    anomaly_pipe.fit(x[FEATURES])

    return bottleneck_model, defect_model, anomaly_pipe

def _confidence(row: pd.Series) -> float:
    sensor_cols = ["temperature", "vibration", "torque"]
    observed = sum(pd.notna(row[c]) for c in sensor_cols)
    sensor_factor = observed / len(sensor_cols)

    # Process data is still available even at manual stations.
    base = 0.58 + 0.38 * sensor_factor
    return float(np.clip(base, 0.55, 0.96))

def score_line(
    df: pd.DataFrame,
    bottleneck_model,
    defect_model,
    anomaly_model,
) -> pd.DataFrame:
    out = df.copy()
    x = out[FEATURES]

    out["bottleneck_risk"] = bottleneck_model.predict_proba(x)[:, 1]
    out["defect_risk"] = defect_model.predict_proba(x)[:, 1]

    # IsolationForest: higher decision_function means more normal.
    raw = anomaly_model.decision_function(x)
    out["anomaly_score"] = np.clip(0.5 - raw * 2.2, 0, 1)

    out["confidence"] = out.apply(_confidence, axis=1)
    out["bottleneck_risk"] *= out["confidence"]
    out["defect_risk"] *= out["confidence"]

    out["status"] = "Normal"
    out.loc[
        (out["bottleneck_risk"] >= 0.55)
        | (out["defect_risk"] >= 0.55)
        | (out["anomaly_score"] >= 0.62),
        "status"
    ] = "Warning"
    out.loc[
        (out["bottleneck_risk"] >= 0.75)
        | (out["defect_risk"] >= 0.75),
        "status"
    ] = "Critical"

    return out

def explain_station(row: pd.Series) -> tuple[list[str], str]:
    reasons = []

    if row["cycle_time"] >= 62:
        reasons.append("cycle time is well above baseline")
    if row["queue_length"] >= 9:
        reasons.append("queue is building upstream")
    if row["throughput"] <= 58:
        reasons.append("throughput has dropped")
    if row["downtime_min"] >= 4:
        reasons.append("recent downtime is elevated")
    if pd.notna(row["vibration"]) and row["vibration"] >= 4.2:
        reasons.append("vibration is abnormal")
    if pd.notna(row["temperature"]) and row["temperature"] >= 75:
        reasons.append("temperature is elevated")
    if pd.notna(row["torque"]) and row["torque"] >= 60:
        reasons.append("torque is outside the normal band")
    if row["confidence"] < 0.75:
        reasons.append("sensor coverage is incomplete, so confidence is reduced")

    if not reasons:
        reasons.append("process values are close to the normal operating range")

    if row["bottleneck_risk"] >= row["defect_risk"] and row["bottleneck_risk"] >= 0.55:
        action = "Inspect station capacity/tooling and rebalance work before the queue grows."
    elif row["defect_risk"] >= 0.55:
        action = "Hold or inspect affected units and check process parameters before final quality inspection."
    elif row["anomaly_score"] >= 0.62:
        action = "Review the station because its current behavior differs from the historical pattern."
    else:
        action = "Continue monitoring."

    return reasons, action
