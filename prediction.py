from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

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
_THRESH = _CFG.get(
    "thresholds",
    {
        "warning": {
            "bottleneck_risk": 0.55,
            "defect_risk": 0.55,
            "anomaly_score": 0.62,
        },
        "critical": {
            "bottleneck_risk": 0.75,
            "defect_risk": 0.75,
        },
    },
)


# ---------------------------------------------------------------------------
# IMPORTANT — what the models actually learn, and why the demo scenarios
# aren't "cheating":
#
# The bottleneck/defect labels below are generated from a hand-specified
# risk formula over a WIDE synthetic distribution of station conditions
# (cycle_time, temperature, vibration, torque, queue, throughput, downtime).
# This is a stand-in for historical plant data a real deployment would use.
#
# The Streamlit "Developing bottleneck" / "High defect risk" / "Sensor-poor
# station" scenarios in simulator.py are a SEPARATE, narrower process that
# nudges specific stations (S14, S07, S20) along a realistic trajectory over
# time. The model never sees these trajectories during training — it only
# sees the general feature -> risk relationship. When a demo scenario makes
# S14's cycle time and queue climb, the model flags it because those FEATURE
# VALUES fall into the region it learned to associate with risk, not because
# it memorized "S14 = bad". That's the intended test: can a model trained on
# general risk patterns generalize to a specific, unseen degradation curve.
# ---------------------------------------------------------------------------

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


def _make_model(seed: int) -> Pipeline:
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        ("scale", StandardScaler()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=180,
                max_depth=8,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
    ])


def train_models(seed: int = 7):
    """Train the bottleneck/defect classifiers and the anomaly detector.

    Also computes held-out validation metrics so the app can show
    precision/recall/AUC numbers instead of only the per-station
    confidence heuristic. Metrics are stashed as a module-level dict
    (`LAST_METRICS`).
    """
    x, y_b, y_d = _synthetic_training_data(seed=seed)

    (
        x_train,
        x_test,
        yb_train,
        yb_test,
        yd_train,
        yd_test,
    ) = train_test_split(
        x,
        y_b,
        y_d,
        test_size=0.25,
        random_state=seed,
        stratify=y_b,
    )

    bottleneck_model = _make_model(seed)
    defect_model = _make_model(seed)

    bottleneck_model.fit(x_train[FEATURES], yb_train)
    defect_model.fit(x_train[FEATURES], yd_train)

    anomaly_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        ("scale", StandardScaler()),
        (
            "model",
            IsolationForest(
                n_estimators=160,
                contamination=0.08,
                random_state=seed,
            ),
        ),
    ])
    anomaly_pipe.fit(x_train[FEATURES])

    # --- Held-out evaluation -----------------------------------------
    def _eval(model, x_te, y_te, name):
        proba = model.predict_proba(x_te[FEATURES])[:, 1]
        pred = (proba >= 0.5).astype(int)

        return {
            "name": name,
            "auc": (
                float(roc_auc_score(y_te, proba))
                if y_te.nunique() > 1
                else float("nan")
            ),
            "precision": float(
                precision_score(y_te, pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_te, pred, zero_division=0)
            ),
            "f1": float(
                f1_score(y_te, pred, zero_division=0)
            ),
            "positive_rate": float(y_te.mean()),
            "n_test": int(len(y_te)),
        }

    global LAST_METRICS
    LAST_METRICS = {
        "bottleneck": _eval(
            bottleneck_model,
            x_test,
            yb_test,
            "Bottleneck risk",
        ),
        "defect": _eval(
            defect_model,
            x_test,
            yd_test,
            "Defect risk",
        ),
    }

    # Calibrate anomaly-score scaling using the observed training distribution.
    train_raw = anomaly_pipe.decision_function(x_train[FEATURES])

    global ANOMALY_LO, ANOMALY_HI
    ANOMALY_LO = float(np.percentile(train_raw, 1))
    ANOMALY_HI = float(np.percentile(train_raw, 99))

    return bottleneck_model, defect_model, anomaly_pipe


# Populated by train_models(); read by score_line() and the app.
LAST_METRICS: dict = {}
ANOMALY_LO: float = -0.05
ANOMALY_HI: float = 0.15


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

    # Model probabilities are kept as risk values.
    # IMPORTANT: confidence is reported separately below; it must not
    # suppress a high-risk prediction simply because sensor coverage is poor.
    out["bottleneck_risk"] = bottleneck_model.predict_proba(x)[:, 1]
    out["defect_risk"] = defect_model.predict_proba(x)[:, 1]

    # IsolationForest: higher decision_function means more normal.
    # Scale the observed training range into an intuitive 0..1 anomaly score.
    raw = anomaly_model.decision_function(x)
    span = max(ANOMALY_HI - ANOMALY_LO, 1e-6)
    out["anomaly_score"] = np.clip(
        (ANOMALY_HI - raw) / span,
        0,
        1,
    )

    # Confidence describes data completeness/observability separately.
    out["confidence"] = out.apply(_confidence, axis=1)
    out["low_confidence"] = out["confidence"] < 0.75

    w = _THRESH.get("warning", {})
    c = _THRESH.get("critical", {})

    out["status"] = "Normal"

    out.loc[
        (
            out["bottleneck_risk"]
            >= w.get("bottleneck_risk", 0.55)
        )
        | (
            out["defect_risk"]
            >= w.get("defect_risk", 0.55)
        )
        | (
            out["anomaly_score"]
            >= w.get("anomaly_score", 0.62)
        ),
        "status",
    ] = "Warning"

    out.loc[
        (
            out["bottleneck_risk"]
            >= c.get("bottleneck_risk", 0.75)
        )
        | (
            out["defect_risk"]
            >= c.get("defect_risk", 0.75)
        ),
        "status",
    ] = "Critical"

    return out


def get_model_metrics() -> dict:
    """Expose the held-out validation metrics computed in train_models()
    so the UI can display precision/recall/AUC instead of only the
    per-station confidence heuristic."""
    return LAST_METRICS


def _pct_dev(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value - baseline) / baseline * 100.0


def explain_station(row: pd.Series) -> tuple[list[str], str]:
    """Return quantified, ranked explanations for station risk."""
    reasons: list[tuple[float, str]] = []

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

        if abs(dev) >= 20:
            direction = "above" if dev > 0 else "below"
            reasons.append((
                abs(dev),
                (
                    f"{label} is {abs(dev):.0f}% {direction} baseline "
                    f"({val:.1f}{unit} vs {baseline:.0f}{unit} typical)"
                ),
            ))

    if row["confidence"] < 0.75:
        reasons.append((
            15,
            (
                "sensor coverage is incomplete at this station, "
                "so confidence is reduced"
            ),
        ))

    reasons.sort(key=lambda t: t[0], reverse=True)
    reasons = [text for _, text in reasons]

    if not reasons:
        reasons = [
            "process values are close to the normal operating range"
        ]

    if (
        row["bottleneck_risk"] >= row["defect_risk"]
        and row["bottleneck_risk"] >= 0.55
    ):
        action = (
            "Inspect station capacity/tooling and rebalance work "
            "before the queue grows."
        )
    elif row["defect_risk"] >= 0.55:
        action = (
            "Hold or inspect affected units and check process parameters "
            "before final quality inspection."
        )
    elif row["anomaly_score"] >= 0.62:
        action = (
            "Review the station because its current behavior differs "
            "from the historical pattern."
        )
    else:
        action = "Continue monitoring."

    return reasons, action
