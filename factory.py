"""ALLUR case 2: observation contract, plant KPIs and incident lifecycle."""
from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import numpy as np
import pandas as pd

from prediction import FEATURES, score_line
from simulator import initial_line, simulate_step

ZONES = {"Body": "Сварка кузова", "Paint": "Окраска", "Trim": "Сборка салона",
         "Chassis": "Шасси и силовой агрегат", "Final": "Финальная сборка и ОТК"}
SCENARIOS = {"Штатная работа": "Normal", "Узкое место S14": "Developing bottleneck",
             "Риск брака S07": "High defect risk", "Потеря датчиков S20": "Sensor-poor station",
             "Остановка S14": "Stopped", "Восстановление": "Recovery"}
STATUS = {"Normal": "Норма", "Warning": "Внимание", "Critical": "Критично"}
REQUIRED = ["station_id", "station_type", "sensor_level", *FEATURES,
            "window_min", "total_count", "good_count"]


def demo_observation(line: pd.DataFrame) -> pd.DataFrame:
    """One rolling hour. Demo quality counters derive from process conditions, not ML scores."""
    out = line.copy()
    out["window_min"] = 60.0
    # Serial stations cannot produce faster than the upstream bottleneck.
    out["total_count"] = np.floor(out["throughput"].cummin()).astype(int)
    severity = ((out["temperature"].fillna(57) - 68).clip(lower=0) / 100
                + (out["vibration"].fillna(2.3) - 3.2).clip(lower=0) / 25)
    out["good_count"] = (out["total_count"] * (1 - severity.clip(0, .35)).cumprod()).round().astype(int)
    return out


def validate_observation(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError("Нет колонок: " + ", ".join(sorted(missing)))
    out = df.copy().reset_index(drop=True)
    expected = set(initial_line(config=config)["station_id"])
    if out.empty or out.station_id.duplicated().any() or set(out.station_id) != expected:
        raise ValueError("Нужен один полный снимок: по одной строке на каждую станцию конфигурации.")
    if not out.station_type.isin(ZONES).all() or not out.sensor_level.isin(["rich", "partial", "manual"]).all():
        raise ValueError("Неизвестный участок или уровень датчиков.")
    configured = initial_line(config=config).set_index("station_id")
    if not out.station_type.equals(out.station_id.map(configured.station_type)):
        raise ValueError("Участки станций не совпадают с конфигурацией.")
    for col in FEATURES + ["window_min", "total_count", "good_count"]:
        out[col] = pd.to_numeric(out[col], errors="raise")
        nullable = col in ["temperature", "vibration", "torque"]
        values = out[col].dropna() if nullable else out[col]
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{col}: нужны конечные неотрицательные числа.")
    if (out.cycle_time <= 0).any() or (out.window_min <= 0).any() or out.window_min.nunique() != 1:
        raise ValueError("Цикл и единое окно наблюдения должны быть положительными.")
    if (out.downtime_min > out.window_min).any() or (out.good_count > out.total_count).any():
        raise ValueError("Простой превышает окно или годных изделий больше общего выпуска.")
    for col in ["total_count", "good_count", "queue_length"]:
        if (out[col] % 1 != 0).any():
            raise ValueError(f"{col}: нужны целые значения.")
    manual = out.sensor_level.eq("manual")
    if out.loc[manual, ["temperature", "vibration", "torque"]].notna().any().any():
        raise ValueError("Для manual температурные, вибрационные и моментные датчики должны быть пустыми.")
    out["process_name"] = out.station_id.map(configured.process_name)
    return out.sort_values("station_id").reset_index(drop=True)


def read_observation(data: bytes, config: dict) -> pd.DataFrame:
    return validate_observation(pd.read_csv(StringIO(data.decode("utf-8-sig"))), config)


def station_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["availability"] = 1 - out.downtime_min / out.window_min
    # Occupied cycle time per available second; capped for inconsistent counters.
    available_seconds = (out.window_min - out.downtime_min) * 60
    out["utilization"] = (out.total_count * out.cycle_time / available_seconds.replace(0, np.nan)).fillna(0).clip(0, 1)
    out["quality"] = out.good_count / out.total_count.replace(0, np.nan)
    out["output_rate"] = out.total_count * 60 / out.window_min
    return out


def plant_kpis(df: pd.DataFrame, config: dict) -> dict:
    """Serial-line output is measured at last station; never sum station throughput."""
    data = station_metrics(df).sort_values("station_id")
    final = data.iloc[-1]
    capacity = float(config["plant"]["rated_units_per_hour"])
    throughput = float(final.output_rate)
    bottleneck = data.loc[data.output_rate.idxmin()]
    availability = float(final.availability)
    quality = float(final.quality) if pd.notna(final.quality) else None
    performance = min(throughput / (capacity * availability), 1) if availability > 0 else 0
    return {"throughput": throughput, "utilization": float(data.utilization.mean()),
            "equipment_downtime": float(data.downtime_min.sum()),
            "quality": quality, "availability": availability, "performance": performance,
            "oee": availability * performance * quality if quality is not None else None,
            "bottleneck_station": str(bottleneck.station_id), "bottleneck_capacity": float(bottleneck.output_rate),
            "shift_forecast": throughput * float(config["plant"]["shift_hours"]),
            "plan": int(config["plant"]["shift_plan"])}


def forecast(history: list[dict], station_id: str, horizon: int = 5, critical: float = .75) -> dict:
    values = [h["risks"][station_id] for h in history[-8:] if station_id in h["risks"]]
    if not values:
        return {"projected": None, "eta_steps": None, "slope": 0.0}
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) >= 3 else 0.0
    eta = 0 if values[-1] >= critical else int(np.ceil((critical - values[-1]) / slope)) if slope > .001 else None
    return {"projected": float(np.clip(values[-1] + slope * horizon, 0, 1)), "eta_steps": eta, "slope": slope}


def incident_signals(row: pd.Series) -> dict[str, tuple[str, str]]:
    signals = {}
    if row.status != "Normal":
        signals["risk"] = (STATUS[row.status], "Отклонение производственного процесса")
    if row.throughput == 0 or row.downtime_min >= row.window_min:
        signals["stop"] = ("Критично", "Оборудование остановлено")
    if row.confidence < .75:
        signals["data"] = ("Внимание", "Неполная телеметрия")
    return signals


class Twin:
    def __init__(self, config: dict, models: tuple):
        self.config, self.models = config, models
        self.step = 0
        self.source = "Симуляция"
        self.line = demo_observation(initial_line(config=config))
        self.history: list[dict] = []
        self.incidents: list[dict] = []
        self.refresh()

    def refresh(self):
        self.scored = station_metrics(score_line(self.line, *self.models, config=self.config))
        self.kpis = plant_kpis(self.scored, self.config)
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        item = {"step": self.step, "time": self.updated_at, **self.kpis,
                "risks": self.scored.set_index("station_id").bottleneck_risk.to_dict()}
        if self.history and self.history[-1]["step"] == self.step:
            self.history[-1] = item
        else:
            self.history.append(item)
        self.history = self.history[-240:]
        active = {(r.station_id, kind): (severity, title) for _, r in self.scored.iterrows()
                  for kind, (severity, title) in incident_signals(r).items()}
        for incident in self.incidents:
            key = (incident["station_id"], incident["kind"])
            if incident["state"] != "Устранён":
                if key not in active:
                    incident.update(state="Устранён", resolved_step=self.step)
                else:
                    severity, _ = active.pop(key)
                    incident.update(severity=severity, last_step=self.step)
        for (station, kind), (severity, title) in active.items():
            self.incidents.append(dict(id=len(self.incidents)+1, station_id=station, kind=kind,
                                       severity=severity, title=title, state="Открыт", first_step=self.step,
                                       last_step=self.step, resolved_step=None))

    def advance(self, scenario: str):
        if self.source != "Симуляция":
            raise ValueError("Для CSV автоматическая симуляция отключена.")
        self.step += 1
        if scenario == "Recovery":
            self.line = demo_observation(initial_line(config=self.config))
        else:
            next_line = simulate_step(self.line, self.step, "Normal" if scenario == "Stopped" else scenario)
            if scenario == "Stopped":
                next_line.loc[next_line.station_id.eq("S14"), ["throughput", "downtime_min", "queue_length"]] = [0, 60, 25]
            self.line = demo_observation(next_line)
        self.refresh()

    def import_csv(self, data: bytes):
        # Validate fully before changing the current state.
        line = read_observation(data, self.config)
        self.line, self.source = line, "CSV"
        self.step += 1
        self.refresh()

    def acknowledge(self, incident_id: int):
        for incident in self.incidents:
            if incident["id"] == incident_id and incident["state"] == "Открыт":
                incident["state"] = "Принят в работу"


def business_effect(downtime_hours: float, reduction_pct: float, cost_per_hour: float,
                    defective_units: int, defect_reduction_pct: float, rework_cost: float) -> dict:
    inputs = [downtime_hours, reduction_pct, cost_per_hour, defective_units, defect_reduction_pct, rework_cost]
    if not all(np.isfinite(v) and v >= 0 for v in inputs) or max(reduction_pct, defect_reduction_pct) > 100:
        raise ValueError("Неверные предположения для расчёта эффекта.")
    hours = downtime_hours * reduction_pct / 100
    units = defective_units * defect_reduction_pct / 100
    return {"hours": hours, "units": units, "kzt": hours * cost_per_hour + units * rework_cost}
