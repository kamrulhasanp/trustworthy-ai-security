# Trustworthy AI Security Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.8475)

Free, open tools that keep AI-based cloud security accurate, fair, and hard to trick.
This project is built in 12 weeks on a laptop with no GPU.

## Status
- Tool 1: Self-updating threat detector (released, v0.1.0)
- Tool 2: Fairness check (first version done)
- Tool 3: Early warning system (next)
- Tool 4: Harmful content detector (planned)
- Tool 5: Attack test kit (planned)

## Tool 1: the problem
A threat detector trained on old attacks quietly fails when new attack types appear. In my test it caught 100% of the attacks it was trained on, then 0% to 1% of the new ones, while false alarms stayed low. Nothing looked wrong on a dashboard.

## What Tool 1 does
1. Checks if incoming traffic has changed, without needing labels.
2. Checks how the current model does on a small labeled slice.
3. If either check raises an alert, trains a new model.
4. Tests the old and new models side by side.
5. Switches only if the new one is clearly better, and logs every decision.

## Results (CIC-IDS2017)
| Day | Never updated | Self-updating pipeline |
|---|---|---|
| Wednesday | 0% | 99.7% |
| Thursday | 0% | 94.1% |
| Friday | 0.9% | 99.6% |

Share of attacks caught (recall). Charts are in `docs/`.

## Second dataset check (CSE-CIC-IDS2018)
Same method and same settings as the first dataset, with no tuning afterwards. First model trained on Wed 14-02 (FTP and SSH brute force), then five later days. Share of attacks caught by the self-updating pipeline (a never-updated model caught 0% on four days and 23% on one):

| Day | 20% labeled | 5% labeled | 2% labeled | Attacks in labeled slice (20% / 5% / 2%) |
|---|---|---|---|---|
| Thu 15-02 (DoS) | about 100% | 99.9% | 99.8% | 3,010 / 752 / 301 |
| Fri 16-02 (DoS) | 99.8% | 99.8% | 99.8% | 34,410 / 8,602 / 3,441 |
| Wed 21-02 (DDoS) | about 100% | about 100% | 99.9% | 39,343 / 9,836 / 3,934 |
| Thu 22-02 (web attacks) | 41% | 18% | 14% | 21 / 5 / 2 |
| Fri 02-03 (Bot) | about 100% | about 100% | 99.9% | 16,419 / 4,105 / 1,642 |

The pipeline worked on four of five days, even with 2% labeled. It failed on the web attack day, which had only 2 to 21 labeled attacks (and only 61 to 76 in the test). Rare new attack types with few examples are the weak spot on both datasets.
Data: CSE-CIC-IDS2018 (registry.opendata.aws/cse-cic-ids2018), same citation as above.

## Limits
- One dataset only.
- Test data comes from the same day as the update data, so scores are best-case.
- The traffic-shape check missed Thursday. The labeled health check caught it, but I added that check after seeing the miss, so it needs testing on another dataset.
- Thursday's test has only 136 attacks.
- The labeled slice is 20% of each day, more than a real team would label.

## How to run
1. Python 3.14 on macOS. Create a virtual environment and run `pip install -r requirements.txt`. On a Mac, XGBoost also needs `brew install libomp`.
2. Download CIC-IDS2017 (MachineLearningCSV.zip) from the University of New Brunswick and put the CSV files in `data/`. The data is not included here.
3. Run, from the project folder: `train.py`, `test_drift.py`, `retrain_experiment.py`, `drift_check.py`, then `pipeline.py` (all in `tool1_drift/`).

## Data and citation
CIC-IDS2017, Canadian Institute for Cybersecurity, University of New Brunswick.
Sharafaldin, Lashkari, and Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP 2018.

## Author
Md Kamrul Hasan, M.S. Cybersecurity, California State University, Dominguez Hills
