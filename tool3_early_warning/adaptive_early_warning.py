import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Adaptive early warning: Tool 3 plus the idea from Tool 1.
# A base model is trained on the other four days. The adapted model also learns from the first part of the
# new day, labeled by an analyst: from the start of the day until LABEL_MINUTES after the first attack begins.
# Both models are then tested only on what comes AFTER that labeled part, so nothing leaks backwards in time.
# Usage: python tool3_early_warning/adaptive_early_warning.py [label_minutes] [weight]
W = 10
LABEL_MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 15
NEW_WEIGHT = float(sys.argv[2]) if len(sys.argv) > 2 else 20   # chosen in advance, not tuned
TAG = "" if (LABEL_MINUTES == 15 and NEW_WEIGHT == 20) else f"_{int(LABEL_MINUTES)}min_w{int(NEW_WEIGHT)}"
CACHE = f"data2018/windows_{W}s.csv"
MIN_ATTACK_FLOWS = 5
MIN_EPISODE = 3
MERGE_GAP = max(1, 60 // W)
PRE = max(1, 60 // W)
THRESH = 0.5
CLOCK_FIX_BEFORE = 8
DAYS = ["Wed 14-02", "Thu 15-02", "Fri 16-02", "Wed 21-02", "Fri 02-03"]


def fix_day(d, label):
    """Same clock fix and features as early_warning.py."""
    d = d[d["n_flows"] > 0].copy()
    d["t_start"] = np.where(d["t_start"] < CLOCK_FIX_BEFORE * 3600, d["t_start"] + 12 * 3600, d["t_start"])
    d.index = (d["t_start"] // W).astype(int)
    d = d[~d.index.duplicated(keep="first")].sort_index()
    d = d.reindex(pd.RangeIndex(int(d.index.min()), int(d.index.max()) + 1))
    d["day"] = label
    for c in ["n_flows", "n_ports", "attack_flows"]:
        d[c] = d[c].fillna(0)
    d["active"] = (d["n_flows"] > 0).astype(int)
    for c in ["n_flows", "n_ports"]:
        base = d[c].shift(1).rolling(max(5, 300 // W), min_periods=3).median()
        d[c + "_ratio"] = d[c] / (base + 1)
    d["t_start"] = d.index * W
    return d.reset_index(drop=True)


def find_episodes(y):
    idx = np.flatnonzero(y)
    if len(idx) == 0:
        return []
    eps, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > MERGE_GAP:
            eps.append((s, prev))
            s = i
        prev = i
    eps.append((s, prev))
    return [(a, b) for a, b in eps if b - a + 1 >= MIN_EPISODE]


def new_model():
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=-1)


def alarms(p, act):
    p = np.where(act, p, 0.0)
    flag = p >= THRESH
    alarm = flag & np.r_[False, flag[:-1]]
    return flag, alarm


raw = pd.read_csv(CACHE)
raw = raw.drop(columns=[c for c in raw.columns if c.endswith("_ratio") or c == "active"])
allw = pd.concat([fix_day(raw[raw["day"] == lb], lb) for lb in DAYS], ignore_index=True)
allw["y_now"] = (allw["attack_flows"] >= MIN_ATTACK_FLOWS).astype(int)
FEATS = [c for c in allw.columns if c not in ("attack_flows", "day", "t_start", "y_now", "active")]
print(f"Analyst labels the day from the start until {LABEL_MINUTES:.0f} minutes after the first attack begins. "
      f"New-day windows get weight {NEW_WEIGHT:.0f}.\n")

rows, ep_lines = [], []
for label in DAYS:
    others = allw[(allw["day"] != label) & (allw["active"] == 1)]
    te = allw[allw["day"] == label].reset_index(drop=True)
    act = te["active"].values == 1
    y = te["y_now"].values
    eps = find_episodes(y)
    split = min(len(te), eps[0][0] + int(LABEL_MINUTES * 60 / W))
    lab = te.iloc[:split]
    lab = lab[lab["active"] == 1]

    m0 = new_model().fit(others[FEATS], others["y_now"])
    X1 = pd.concat([others[FEATS], lab[FEATS]])
    y1 = np.r_[others["y_now"].values, lab["y_now"].values]
    w1 = np.r_[np.ones(len(others)), np.full(len(lab), NEW_WEIGHT)]
    m1 = new_model().fit(X1, y1, sample_weight=w1)

    test_part = np.arange(len(te)) >= split
    atk = (y == 1) & test_part
    ext = np.zeros(len(te), dtype=bool)
    for s, e in eps:
        ext[max(0, s - PRE): min(len(te), e + PRE + 1)] = True
    calm = ~ext & act & test_part
    calm_hours = max(calm.sum() * W / 3600, 1e-9)
    later = [(s, e) for s, e in eps if s >= split]

    out = {}
    for name, model in (("base", m0), ("adapted", m1)):
        flag, alarm = alarms(model.predict_proba(te[FEATS])[:, 1], act)
        starts = alarm & ~np.r_[False, alarm[:-1]]
        delays = []
        for s, e in later:
            cand = [w for w in range(max(0, s - PRE), e + 1) if alarm[w] and w >= split]
            delays.append((cand[0] + 1 - s) * W if cand else None)
        out[name] = dict(recall=float((flag & atk).sum() / max(1, atk.sum())),
                         false_alarms=int((starts & calm).sum()), delays=delays)

    rows.append(dict(
        day=label, labeled_until_hour=round(te.loc[min(split, len(te) - 1), "t_start"] / 3600, 2),
        attack_windows_labeled=int(lab["y_now"].sum()), attack_windows_tested=int(atk.sum()),
        recall_base=round(out["base"]["recall"], 3), recall_adapted=round(out["adapted"]["recall"], 3),
        false_alarms_base=out["base"]["false_alarms"], false_alarms_adapted=out["adapted"]["false_alarms"],
        calm_hours_tested=round(calm_hours, 1)))
    for k, (s, e) in enumerate(later):
        b, a = out["base"]["delays"][k], out["adapted"]["delays"][k]
        fmt = lambda v: "no alarm" if v is None else f"{v} s"
        ep_lines.append(dict(day=label, starts_at_hour=round(te.loc[s, "t_start"] / 3600, 2),
                             length_s=(e - s + 1) * W, base=fmt(b), adapted=fmt(a)))

res = pd.DataFrame(rows)
res.to_csv(f"docs/adaptive_results{TAG}.csv", index=False)
print("Per day (tested only on what comes after the labeled part):")
print(res.to_string(index=False))
if ep_lines:
    eps_df = pd.DataFrame(ep_lines)
    eps_df.to_csv(f"docs/adaptive_episodes{TAG}.csv", index=False)
    print("\nAttack periods that start after the labeled part (seconds from start to alarm):")
    print(eps_df.to_string(index=False))

x = np.arange(len(res))
w = 0.35
fig, ax = plt.subplots(figsize=(7.5, 4.6))
b1 = ax.bar(x - w / 2, res["recall_base"], w, label="Base (trained on other days)", color="#e63946")
b2 = ax.bar(x + w / 2, res["recall_adapted"], w, label="Adapted (plus labeled start of the day)", color="#2a9d8f")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.2f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(res["day"])
ax.set_ylim(0, 1.3)
ax.set_ylabel("Attack windows flagged (after the labeled part)")
ax.set_title(f"Adaptive early warning ({LABEL_MINUTES:.0f} minutes labeled after the first attack starts)", fontsize=10)
ax.legend(loc="upper center", ncol=1, fontsize=8)
plt.tight_layout()
plt.savefig(f"docs/adaptive_chart{TAG}.png", dpi=150)
print(f"\nSaved docs/adaptive_results{TAG}.csv, docs/adaptive_episodes{TAG}.csv, docs/adaptive_chart{TAG}.png")