2026-11-07 | Tool 1 release v0.1.0 archived on Zenodo. DOI: 10.5281/zenodo.8475

2026-11-07 | Tool 2 v1: recall by service type. Gap best-worst 0.036 before, 0.096 after per-group thresholds. Fix hurt File and mail (0.962 to 0.904, only 52 attacks, tuning set too small). DNS and Remote login too small to compare. Chart: docs/fairness_chart.png

2026-11-07 | Tool 2 v2: service-type fairness check with 95% ranges. File and mail recall 0.962 (0.870 to 0.989, 52 attacks) vs Web 0.996 and Other 0.997. Gap 0.036 before, 0.038 after; judged real by range test. Own thresholds only for Web and Other (overall recall 0.996 to 0.999, false alarms 0.15% to 0.22%). DNS and Remote login too small. Chart: docs/fairness_chart.png

2026-11-08 | Tool 2 error analysis by attack type. File and mail group holds only port scans (2 of 52 missed, 3.8% vs 0.15% elsewhere, 2 misses only). Bot attacks are the real weak spot: 25 of 76 missed (33%), 28% of all misses. Web Attack Brute Force/XSS 9 of 134 missed. Output: docs/attack_types_by_group.csv

2026-11-08 | Smaller labeled slices. Pipeline recall at 20% / 5% / 2%: Wed 99.7 / 98.4 / 96.0, Thu 94.1 / 83.4 / 74.3, Fri 99.6 / 99.1 / 99.1. Thursday's labeled slice had only 11 and 5 attacks. Limit is attack examples, not the percentage. Drift check fired on Thursday this time (0.244, 0.218) but health check fired every day. Files: docs/pipeline_results_slice5.csv, slice2.csv, charts.

2026-11-08 | Rare-attack weighting (inverse sqrt frequency, cap 10). Missed attacks 88 to 66 of 23,388; false alarms 99 to 102. Bot 67.1% to 72.4% (51 to 55 of 76, ranges overlap, still weakest). Brute Force 94.6% to 100%. Some gains also in common types, so partly ordinary variation. File: docs/rare_attack_results.csv

2026-11-08 | Paired check on rare-attack weighting: 35 attacks caught only by weighted, 13 only by plain (p=0.002, probably optimistic since flows are correlated). Bot only: 7 vs 3 (p=0.34, not clear). Weighting helps overall; Bot still weakest. Progress report v2 written (Tool 1 and Tool 2).

2026-11-08 | Second dataset CSE-CIC-IDS2018, settings frozen from 2017. Pipeline vs never updated: Thu 15-02 100% vs 0%, Fri 16-02 99.8% vs 23% (no switch, shadow test blocked it), Wed 21-02 100% vs 0%, Thu 22-02 41% vs 0% (web attacks, 61 test attacks, 21 labeled), Fri 02-03 100% vs 0% (Bot, 16,419 labeled examples). Traffic-shape alarm fired every day; health check fired only when needed. Weak spot: rare web attacks.