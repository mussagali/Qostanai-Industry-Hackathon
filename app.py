from __future__ import annotations

import time
import pandas as pd
import plotly.express as px
import streamlit as st

from simulator import initial_line, simulate_step
from prediction import train_models, score_line, explain_station


st.set_page_config(
    page_title="DigitalTwin.ai — Vehicle Assembly Line",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 DigitalTwin.ai")
st.caption("Vehicle assembly-line digital twin • bottleneck prediction • defect-risk prediction • uneven sensor coverage")

@st.cache_resource
def get_models():
    return train_models()

bottleneck_model, defect_model, anomaly_model = get_models()

if "step" not in st.session_state:
    st.session_state.step = 0
if "line" not in st.session_state:
    st.session_state.line = initial_line(30)
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Simulation controls")
    scenario = st.selectbox(
        "Scenario",
        ["Normal", "Developing bottleneck", "High defect risk", "Sensor-poor station"],
    )
    col1, col2 = st.columns(2)
    with col1:
        advance = st.button("▶ Next step", use_container_width=True)
    with col2:
        reset = st.button("↺ Reset", use_container_width=True)

    if reset:
        st.session_state.step = 0
        st.session_state.line = initial_line(30)
        st.session_state.history = []
        st.rerun()

    if advance:
        st.session_state.step += 1
        st.session_state.line = simulate_step(
            st.session_state.line,
            st.session_state.step,
            scenario=scenario,
        )

    st.markdown("---")
    st.markdown("**Suggested demo:**")
    st.markdown("1. Choose **Developing bottleneck**")
    st.markdown("2. Press **Next step** 6–10 times")
    st.markdown("3. Watch Station S14 turn warning/critical")
    st.markdown("4. Switch to **High defect risk** and reset")
    st.markdown("5. Advance 6–10 steps to see S07")

scored = score_line(
    st.session_state.line,
    bottleneck_model,
    defect_model,
    anomaly_model,
)

# History for trend chart
top_b = scored.sort_values("bottleneck_risk", ascending=False).iloc[0]
top_d = scored.sort_values("defect_risk", ascending=False).iloc[0]
st.session_state.history.append({
    "step": st.session_state.step,
    "scenario": scenario,
    "max_bottleneck_risk": float(top_b["bottleneck_risk"]),
    "max_defect_risk": float(top_d["defect_risk"]),
    "avg_cycle_time": float(scored["cycle_time"].mean()),
    "total_queue": int(scored["queue_length"].sum()),
})
# Keep history bounded and avoid repeated entries if user only changes tabs
hist_df = pd.DataFrame(st.session_state.history).drop_duplicates(
    subset=["step", "scenario"], keep="last"
).tail(100)

# Top KPIs
critical_count = int((scored["status"] == "Critical").sum())
warning_count = int((scored["status"] == "Warning").sum())
avg_cycle = scored["cycle_time"].mean()
line_throughput = scored["throughput"].median()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Line throughput", f"{line_throughput:.1f} units/hr")
c2.metric("Avg cycle time", f"{avg_cycle:.1f} sec")
c3.metric("Warnings", warning_count)
c4.metric("Critical stations", critical_count)
c5.metric("Simulation step", st.session_state.step)

tab1, tab2, tab3, tab4 = st.tabs([
    "🟢 Live Twin",
    "⚠️ Alerts",
    "📈 Plant Manager",
    "💼 Leadership",
])

with tab1:
    st.subheader("Live station twin")

    # Station cards
    cols = st.columns(5)
    for i, (_, row) in enumerate(scored.iterrows()):
        with cols[i % 5]:
            icon = {"Normal": "🟢", "Warning": "🟡", "Critical": "🔴"}[row["status"]]
            st.markdown(
                f"""
                **{icon} {row['station_id']} — {row['station_type']}**  
                Status: **{row['status']}**  
                Cycle: {row['cycle_time']:.1f}s  
                Queue: {int(row['queue_length'])}  
                Bottleneck: {row['bottleneck_risk']*100:.0f}%  
                Defect: {row['defect_risk']*100:.0f}%  
                Confidence: {row['confidence']*100:.0f}%
                """
            )

    st.markdown("---")
    station = st.selectbox("Inspect a station", scored["station_id"].tolist())
    row = scored.loc[scored["station_id"].eq(station)].iloc[0]
    reasons, action = explain_station(row)

    a, b = st.columns([1.1, 1])
    with a:
        st.write("### Station details")
        display = pd.DataFrame({
            "Metric": [
                "Sensor coverage", "Cycle time", "Temperature", "Vibration",
                "Torque", "Queue length", "Throughput", "Downtime",
                "Bottleneck risk", "Defect risk", "Anomaly score", "Confidence",
            ],
            "Value": [
                row["sensor_level"],
                f"{row['cycle_time']:.1f} sec",
                "N/A" if pd.isna(row["temperature"]) else f"{row['temperature']:.1f} °C",
                "N/A" if pd.isna(row["vibration"]) else f"{row['vibration']:.2f} mm/s",
                "N/A" if pd.isna(row["torque"]) else f"{row['torque']:.1f} Nm",
                int(row["queue_length"]),
                f"{row['throughput']:.1f} units/hr",
                f"{row['downtime_min']:.1f} min",
                f"{row['bottleneck_risk']*100:.1f}%",
                f"{row['defect_risk']*100:.1f}%",
                f"{row['anomaly_score']*100:.1f}%",
                f"{row['confidence']*100:.1f}%",
            ]
        })
        st.dataframe(display, hide_index=True, use_container_width=True)

    with b:
        st.write("### Explanation")
        for r in reasons:
            st.write(f"• {r.capitalize()}.")
        st.info(f"Recommended action: {action}")

with tab2:
    st.subheader("Priority alerts")
    alerts = scored[
        (scored["status"] != "Normal")
        | (scored["anomaly_score"] >= 0.62)
    ].copy()

    if alerts.empty:
        st.success("No active high-priority alerts.")
    else:
        alerts["priority"] = (
            alerts[["bottleneck_risk", "defect_risk", "anomaly_score"]].max(axis=1)
        )
        alerts = alerts.sort_values("priority", ascending=False)

        for _, row in alerts.head(8).iterrows():
            reasons, action = explain_station(row)
            if row["status"] == "Critical":
                st.error(
                    f"{row['station_id']} • bottleneck {row['bottleneck_risk']*100:.0f}% • "
                    f"defect {row['defect_risk']*100:.0f}% • confidence {row['confidence']*100:.0f}%"
                )
            else:
                st.warning(
                    f"{row['station_id']} • bottleneck {row['bottleneck_risk']*100:.0f}% • "
                    f"defect {row['defect_risk']*100:.0f}% • confidence {row['confidence']*100:.0f}%"
                )
            st.caption("Why: " + "; ".join(reasons[:3]) + ".")
            st.write(f"**Action:** {action}")

with tab3:
    st.subheader("Plant manager view")

    plot_df = scored.sort_values("station_id")
    fig = px.bar(
        plot_df,
        x="station_id",
        y="bottleneck_risk",
        hover_data=["cycle_time", "queue_length", "throughput", "sensor_level"],
        title="Bottleneck risk by station",
        labels={"bottleneck_risk": "Bottleneck risk", "station_id": "Station"},
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    if not hist_df.empty:
        long = hist_df.melt(
            id_vars=["step"],
            value_vars=["max_bottleneck_risk", "max_defect_risk"],
            var_name="metric",
            value_name="risk",
        )
        fig2 = px.line(
            long,
            x="step",
            y="risk",
            color="metric",
            markers=True,
            title="Prediction trend during simulation",
        )
        fig2.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)

    st.write("### Lowest sensor coverage")
    sensor_poor = scored.loc[
        scored["sensor_level"].isin(["partial", "manual"]),
        ["station_id", "station_type", "sensor_level", "confidence", "status"]
    ].copy()
    sensor_poor["confidence"] = (sensor_poor["confidence"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(sensor_poor, hide_index=True, use_container_width=True)

with tab4:
    st.subheader("Leadership / business-case view")
    high_risk = int(((scored["bottleneck_risk"] >= 0.65) | (scored["defect_risk"] >= 0.65)).sum())

    # Illustrative business-case values — intentionally labeled.
    estimated_downtime_avoided = high_risk * 18
    estimated_units_protected = high_risk * 7
    illustrative_value = estimated_downtime_avoided * 150 + estimated_units_protected * 400

    x1, x2, x3 = st.columns(3)
    x1.metric("High-risk stations detected", high_risk)
    x2.metric("Illustrative downtime avoided", f"{estimated_downtime_avoided} min")
    x3.metric("Illustrative protected value", f"${illustrative_value:,.0f}")

    st.caption(
        "Business-case numbers above are illustrative prototype assumptions, not measured plant savings."
    )

    st.write("### Rollout logic")
    st.markdown(
        """
        - **Phase 1:** connect the digital twin read-only to one pilot line.
        - **Phase 2:** validate alerts against actual downtime and quality outcomes.
        - **Phase 3:** tune thresholds and confidence by station type.
        - **Phase 4:** expand to other lines/plants while preserving site-specific configuration.
        """
    )

st.markdown("---")
st.caption(
    "Prototype note: predictions are trained on synthetic data for demonstration. "
    "A production deployment would require validation against real plant outcomes before operational use."
)
