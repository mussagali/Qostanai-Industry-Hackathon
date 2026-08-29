# 2-Minute Demo Script

## Opening (15 sec)
"This prototype is a digital twin of a vehicle assembly line. It simulates 30 stations with mixed sensor maturity and predicts two operational risks: bottlenecks and defects."

## Live line (20 sec)
"Each station has cycle time, queue, throughput, and—where available—temperature, vibration, and torque. Green is normal, yellow is warning, and red is critical."

## Bottleneck scenario (35 sec)
"I'll trigger a developing bottleneck. Station S14 gradually slows, its queue grows, and throughput falls. The model increases the bottleneck probability before the station fully fails. The alert also explains why it is concerned and recommends an inspection/rebalancing action."

## Defect scenario (25 sec)
"In the defect scenario, process parameters at S07 drift. The twin estimates rising defect risk so an affected unit can be inspected before final quality inspection."

## Missing-sensor scenario (15 sec)
"Not every factory station is richly instrumented. At S20, the direct machine sensors are unavailable. The twin still uses process behavior such as cycle time and queue length, but it visibly lowers confidence."

## Stakeholder views (10 sec)
"The same underlying twin supports a supervisor's real-time alerts, a manager's trend view, and a leadership business-case view."

## Close (10 sec)
"The key idea is not a perfect virtual factory. It is a predictive, confidence-aware twin that remains useful in a realistic mixed legacy/modern production environment."
