# DigitalTwin.ai — Vehicle Assembly Line MVP

A prototype of a **vehicle assembly-line digital twin** that:

- simulates a configurable production line (30 stations by default),
- supports uneven sensor coverage (rich / partial / manual),
- predicts developing bottlenecks,
- predicts defect risk,
- detects abnormal process behavior,
- lowers confidence when sensor data is missing,
- explains alerts with **quantified deviations from baseline** and recommends operator actions,
- **traces defect-risk exposure back to individual simulated vehicles**, even after they've left the affected station,
- captures **human feedback on each alert** (confirmed / false alarm) and reports an acceptance rate,
- is **parameterized by `config.yaml`** so a different line/plant can be modeled without code changes,
- exposes separate views for a floor supervisor, plant manager, leadership, and QA traceability.

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

### Scenario 4 — Traceability & feedback loop
1. Reset, choose **High defect risk**, advance 6–10 steps.
2. Open **Traceability** — see which simulated vehicles passed through S07 while defect risk was elevated, even ones that have since moved further down the line.
3. Open **Alerts** and click **✅ Confirm issue** or **❌ False alarm** on a couple of alerts.
4. Note the **alert acceptance rate** metric update — this is the feedback loop that would calibrate thresholds against real plant outcomes.

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
├── config.yaml
├── requirements.txt
├── README.md
├── DEMO_SCRIPT.md
├── BUSINESS_PROPOSAL.md
└── data/
    └── sample_factory.csv
```

## Multi-site configuration

`config.yaml` defines the station count/type mix, sensor coverage split, vehicle-batch settings, and alert thresholds. To model a different plant or line, copy it (e.g. `config_plant_b.yaml`), edit the values, and point the app at it:

```bash
CONFIG_PATH=config_plant_b.yaml streamlit run app.py
```

No code changes are required to represent a line with a different layout or sensor maturity.

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
2. Persist the feedback log and vehicle genealogy to a database (e.g. PostgreSQL) instead of session state.
3. Add WebSocket/MQTT ingestion from real OT/PLC data sources.
4. Validate predictions against labeled production outcomes using the captured feedback log.
5. Add per-station, per-plant threshold overrides on top of the global `config.yaml`.
6. Add ROI estimation based on real downtime and scrap cost.
7. Propose specific low-cost retrofit sensors (e.g. clamp-on current/vibration sensors) for manual-coverage stations.