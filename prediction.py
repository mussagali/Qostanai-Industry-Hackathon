from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from simulator import load_config


FEATURES = [
    "cycle_time",
    "temperature",
    "vibration",
    "torque",
    "queue_length",
    "throughput",
    "downtime_min",
]

# Typical/baseline values a healthy station operates around. Used to turn
# alert explanations into quantified deviations ("29% above baseline")
# instead of plain threshold crossings.
BASELINES = {
    "cycle_time": 48.0,
    "temperature": 57.0,
    "vibration": 2.3,
    "torque": 45.0,
    "queue_length": 2.0,
    "throughput": 72.0,
    "downtime_min": 1.0,
}

_CFG = load_config()
_THRESH = _CFG.get("thresholds", {
    "warning": {"bottleneck_risk": 0.55, "defect_risk": 0.55, "anomaly_score": 0.62},
    "critical": {"bottleneck_risk": 0.75, "defect_risk": 0.75},
})

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

    w = _THRESH.get("warning", {})
    c = _THRESH.get("critical", {})

    out["status"] = "Normal"
    out.loc[
        (out["bottleneck_risk"] >= w.get("bottleneck_risk", 0.55))
        | (out["defect_risk"] >= w.get("defect_risk", 0.55))
        | (out["anomaly_score"] >= w.get("anomaly_score", 0.62)),
        "status"
    ] = "Warning"
    out.loc[
        (out["bottleneck_risk"] >= c.get("bottleneck_risk", 0.75))
        | (out["defect_risk"] >= c.get("defect_risk", 0.75)),
        "status"
    ] = "Critical"

    return out

def _pct_dev(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value - baseline) / baseline * 100.0


def explain_station(row: pd.Series) -> tuple[list[str], str]:
    """Returns quantified, ranked explanations rather than plain threshold
    checks — e.g. 'cycle time is 29% above baseline (62.0s vs 48s typical)'.
    This makes the top contributing factor(s) explicit for a multi-causal
    alert instead of a flat list of everything that crossed a line."""
    reasons: list[tuple[float, str]] = []  # (abs magnitude for ranking, text)

    checks = [
        ("cycle_time", "s", "cycle time"),
        ("queue_length", " units", "queue length"),
        ("throughput", " units/hr", "throughput"),
        ("downtime_min", " min", "downtime"),
        ("vibration", " mm/s", "vibration"),
        ("temperature", " °C", "temperature"),
        ("torque", " Nm", "torque"),
    ]
    for col, unit, label in checks:
        val = row.get(col)
        if pd.isna(val):
            continue
        baseline = BASELINES[col]
        dev = _pct_dev(val, baseline)
        if abs(dev) >= 20:  # meaningful deviation threshold
            direction = "above" if dev > 0 else "below"
            reasons.append((
                abs(dev),
                f"{label} is {abs(dev):.0f}% {direction} baseline ({val:.1f}{unit} vs {baseline:.0f}{unit} typical)",
            ))

    if row["confidence"] < 0.75:
        reasons.append((15, "sensor coverage is incomplete at this station, so confidence is reduced"))

    reasons.sort(key=lambda t: t[0], reverse=True)
    reasons = [text for _, text in reasons]

    if not reasons:
        reasons = ["process values are close to the normal operating range"]

    if row["bottleneck_risk"] >= row["defect_risk"] and row["bottleneck_risk"] >= 0.55:
        action = "Inspect station capacity/tooling and rebalance work before the queue grows."
    elif row["defect_risk"] >= 0.55:
        action = "Hold or inspect affected units and check process parameters before final quality inspection."
    elif row["anomaly_score"] >= 0.62:
        action = "Review the station because its current behavior differs from the historical pattern."
    else:
        action = "Continue monitoring."

    return reasons, action