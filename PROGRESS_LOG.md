2026-11-07 | Tool 1 release v0.1.0 archived on Zenodo. DOI: 10.5281/zenodo.8475

2026-11-07 | Tool 2 v1: recall by service type. Gap best-worst 0.036 before, 0.096 after per-group thresholds. Fix hurt File and mail (0.962 to 0.904, only 52 attacks, tuning set too small). DNS and Remote login too small to compare. Chart: docs/fairness_chart.png

2026-11-07 | Tool 2 v2: service-type fairness check with 95% ranges. File and mail recall 0.962 (0.870 to 0.989, 52 attacks) vs Web 0.996 and Other 0.997. Gap 0.036 before, 0.038 after; judged real by range test. Own thresholds only for Web and Other (overall recall 0.996 to 0.999, false alarms 0.15% to 0.22%). DNS and Remote login too small. Chart: docs/fairness_chart.png