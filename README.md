# DigitalTwin.ai — Vehicle Assembly Line MVP

A one-day hackathon prototype of a **vehicle assembly-line digital twin** that:

- simulates 30 production stations,
- supports uneven sensor coverage,
- predicts developing bottlenecks,
- predicts defect risk,
- detects abnormal process behavior,
- lowers confidence when sensor data is missing,
- explains alerts and recommends operator actions,
- exposes separate views for a floor supervisor, plant manager, and leadership.

## Why this project

The prototype is designed for the DigitalTwin.ai Round 2 challenge: a vehicle assembly line with mixed legacy/modern equipment, uneven sensing, bottleneck prediction, and defect prediction.

## Quick start

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Recommended 2-minute demo

### Scenario 1 — Bottleneck prediction
1. Open **Live Twin**.
2. Choose **Developing bottleneck** in the sidebar.
3. Press **Next step** 6–10 times.
4. Observe Station **S14**:
   - cycle time rises,
   - queue grows,
   - throughput falls,
   - vibration/temperature may worsen,
   - bottleneck risk increases.
5. Open **Alerts** to show the explanation and recommended action.

### Scenario 2 — Defect risk
1. Reset.
2. Choose **High defect risk**.
3. Press **Next step** 6–10 times.
4. Observe Station **S07** as process parameters drift.
5. Show the defect-risk prediction and recommendation.

### Scenario 3 — Uneven sensor coverage
1. Reset.
2. Choose **Sensor-poor station**.
3. Advance several steps.
4. Inspect **S20**.
5. Temperature/vibration/torque are missing, but the twin still uses process data such as cycle time and queue.
6. Confidence is intentionally lower.

## Architecture

```text
Synthetic shop-floor data
        |
        v
Assembly-line simulator
        |
        v
Digital twin state (30 stations)
        |
        +--------------------+
        |                    |
        v                    v
Bottleneck model        Defect-risk model
        |                    |
        +----------+---------+
                   |
                   v
          Anomaly detection
                   |
                   v
        Confidence + explanation
                   |
                   v
         Streamlit dashboard
        /        |          \
 Supervisor   Manager    Leadership
```

## ML approach

This MVP intentionally uses lightweight, explainable building blocks:

- **Random Forest** for bottleneck probability.
- **Random Forest** for defect-risk probability.
- **Isolation Forest** for abnormal station behavior.
- **Median imputation + missing indicators** for incomplete sensor data.
- A **confidence penalty** when temperature/vibration/torque are unavailable.

The models are trained on synthetic data at startup. This is suitable for a prototype, but not a production claim.

## Project structure

```text
digital_twin_mvp/
├── app.py
├── simulator.py
├── prediction.py
├── requirements.txt
├── README.md
├── DEMO_SCRIPT.md
├── BUSINESS_PROPOSAL.md
└── data/
    └── sample_factory.csv
```

## Important prototype assumptions

- 30 stations are simulated.
- Sensor coverage is intentionally mixed: rich, partial, and manual.
- Predictions are illustrative and trained on synthetic examples.
- No PLC or production-control system is modified.
- The twin is read-only from an operational-control perspective.
- Any real deployment would require plant validation, change control, cybersecurity review, and measured false-positive/false-negative performance.

## Suggested GitHub description

> Predictive vehicle assembly-line digital twin that simulates uneven sensor coverage and predicts bottlenecks, defects, and anomalous station behavior with confidence-aware alerts.

## Next upgrades

If you have more time:

1. Add a 2D/3D line visualization.
2. Add vehicle-level genealogy and root-cause tracing.
3. Add time-series persistence with PostgreSQL.
4. Add WebSocket/MQTT ingestion.
5. Validate predictions against labeled production outcomes.
6. Add configurable thresholds per station/plant.
7. Add ROI estimation based on real downtime and scrap cost.
