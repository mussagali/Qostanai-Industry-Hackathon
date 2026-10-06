# 2-Minute Demo Script

## Opening (15 sec)
"This prototype is a digital twin of a vehicle assembly line. It simulates 30 stations with mixed sensor maturity and predicts two operational risks: bottlenecks and defects."

## Live line (20 sec)
"Each station has cycle time, queue, throughput, and—where available—temperature, vibration, and torque. Green is normal, yellow is warning, and red is critical."

## Bottleneck scenario (40 sec)
"I'll trigger a developing bottleneck. Station S14 gradually slows, its queue grows, and throughput falls. The model increases the bottleneck probability before the station fully fails. The alert also explains why it is concerned and recommends an inspection/rebalancing action. Below that, a predictive outlook panel shows the risk trend — current risk, where it's projected a few steps ahead, and roughly how many steps until it crosses the critical threshold. This is labeled explicitly as a prototype trend estimate, not a production forecasting model."

## Defect scenario (25 sec)
"In the defect scenario, process parameters at S07 drift. The twin estimates rising defect risk so an affected unit can be inspected before final quality inspection."

## Traceability (15 sec)
"Because a defect at S07 might not surface until a later inspection point, every vehicle is tagged with the stations it passed and the risk at the time. So even after a unit has moved downstream, we can still pull up its journey and flag it for inspection — this is the audit trail a quality team would use for root-cause tracing."

## Missing-sensor scenario (10 sec)
"Not every station is richly instrumented. At S20, direct machine sensors are unavailable. The twin still uses process behavior such as cycle time and queue length, but it visibly lowers confidence."

## Feedback loop (10 sec)
"On the Alerts tab, a supervisor can confirm an issue or mark it a false alarm. That feedback rolls up into an acceptance rate — the mechanism that would calibrate thresholds against real outcomes over time."

## Stakeholder views (10 sec)
"The same underlying twin supports a supervisor's real-time alerts, a quality engineer's traceability view, a manager's trend analytics, and a leadership business-case view."

## Close (15 sec)
"The key idea is not a perfect virtual factory. It is a predictive, confidence-aware twin that remains useful in a realistic mixed legacy/modern production environment — and there's also an unscripted Random stress-test scenario if you want to see it react to conditions it wasn't tuned against."