import datetime as dt
import glob
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Tool 3 (version 2): early warning on CSE-CIC-IDS2018.
# Traffic is grouped into windows of W seconds. A model scores each window, and an alarm is raised
# when two windows in a row score high. Optional: python tool3_early_warning/early_warning.py 30
W = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DATA = "data2018"
CACHE = f"{DATA}/windows_{W}s.csv"
MIN_ATTACK_FLOWS = 5          # a window counts as "under attack" if it holds at least this many attack flows
MIN_EPISODE = 3               # ignore attack episodes shorter than this many windows
MERGE_GAP = max(1, 60 // W)   # attack windows closer than 60 seconds belong to one episode
PRE = max(1, 60 // W)         # look back 60 seconds before an episode starts
THRESH = 0.5                  # score level that counts as "high"
CLOCK_FIX_BEFORE = 8          # the timestamps look like a 12-hour clock with no AM/PM: hours before 8 are really afternoon

DAYS = [
    ("Wednesday-14-02", dt.date(2018, 2, 14), "Wed 14-02"),
    ("Thursday-15-02", dt.date(2018, 2, 15), "Thu 15-02"),
    ("Friday-16-02", dt.date(2018, 2, 16), "Fri 16-02"),
    ("Wednesday-21-02", dt.date(2018, 2, 21), "Wed 21-02"),
    ("Friday-02-03", dt.date(2018, 3, 2), "Fri 02-03"),
]
DROP = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]


def build_windows(prefix, expected, label):
    """Read one day in pieces and turn it into one row per W-second window."""
    files = sorted(glob.glob(f"{DATA}/{prefix}*.csv"))
    if not files:
        sys.exit(f"No file starting with '{prefix}' found in {DATA}/")
    midnight = pd.Timestamp(expected)
    dayfirst = None
    S = C = N = P = A = None
    for chunk in pd.read_csv(files[0], low_memory=False, chunksize=200000):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk[chunk["Label"] != "Label"]
        if dayfirst is None:   # decide the date format once, on the first piece
            best = -1
            for df_ in (True, False):
                tt = pd.to_datetime(chunk["Timestamp"], dayfirst=df_, errors="coerce")
                sc = (tt.dt.date == expected).mean()
                if sc > best:
                    best, dayfirst = sc, df_
            if best < 0.9:
                print(f"  Warning: only {best:.0%} of timestamps in {prefix} fall on {expected}")
        t = pd.to_datetime(chunk["Timestamp"], dayfirst=dayfirst, errors="coerce")
        keep = (t.dt.date == expected).values
        chunk, t = chunk[keep], t[keep]
        if chunk.empty:
            continue
        y = (chunk["Label"] != "Benign").astype(int)
        port = pd.to_numeric(chunk["Dst Port"], errors="coerce")
        X = chunk.drop(columns=[c for c in DROP + ["Label", "Dst Port"] if c in chunk.columns])
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        win = ((t - midnight).dt.total_seconds() // W).astype(int)
        g = X.groupby(win)
        s, c = g.sum(), g.count()
        n = win.groupby(win).size()
        a = y.groupby(win).sum()
        p = port.groupby(win).nunique()
        S = s if S is None else S.add(s, fill_value=0)
        C = c if C is None else C.add(c, fill_value=0)
        N = n if N is None else N.add(n, fill_value=0)
        A = a if A is None else A.add(a, fill_value=0)
        P = p if P is None else P.add(p, fill_value=0)
    feat = S / C.replace(0, np.nan)
    feat["n_flows"], feat["n_ports"], feat["attack_flows"] = N, P, A
    feat["t_start"] = feat.index * W
    feat["day"] = label
    return feat.reset_index(drop=True)


def fix_day(d, label):
    """Fix the clock, line the windows up in time, and add 'how unusual is this window' features."""
    d = d[d["n_flows"] > 0].copy()
    before = sorted(set((d["t_start"] // 3600).astype(int)))
    d["t_start"] = np.where(d["t_start"] < CLOCK_FIX_BEFORE * 3600, d["t_start"] + 12 * 3600, d["t_start"])
    after = sorted(set((d["t_start"] // 3600).astype(int)))
    print(f"  {label}: hours with traffic before the clock fix {before}; after {after}")
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


# 1. Build (or load) the window table, then fix the clock
try:
    raw = pd.read_csv(CACHE)
    print(f"Loaded {CACHE}")
except FileNotFoundError:
    print("Building windows (this reads five large files and takes a few minutes)...")
    raw = pd.concat([build_windows(pf, d, lb) for pf, d, lb in DAYS], ignore_index=True)
    raw.to_csv(CACHE, index=False)
    print(f"Saved {CACHE}")
raw = raw.drop(columns=[c for c in raw.columns if c.endswith("_ratio") or c == "active"])
allw = pd.concat([fix_day(raw[raw["day"] == lb], lb) for _, _, lb in DAYS], ignore_index=True)
allw["y_now"] = (allw["attack_flows"] >= MIN_ATTACK_FLOWS).astype(int)
FEATS = [c for c in allw.columns if c not in ("attack_flows", "day", "t_start", "y_now", "active")]

# 2. One day at a time: train on the other days, then watch the new day
ep_rows, day_rows, curves = [], [], {}
for _, _, label in DAYS:
    tr = allw[(allw["day"] != label) & (allw["active"] == 1)]
    te = allw[allw["day"] == label].reset_index(drop=True)
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=-1)
    model.fit(tr[FEATS], tr["y_now"])
    act = te["active"].values == 1
    p = np.where(act, model.predict_proba(te[FEATS])[:, 1], 0.0)   # windows with no traffic cannot alarm
    y = te["y_now"].values
    flag = p >= THRESH
    alarm = flag & np.r_[False, flag[:-1]]            # two high windows in a row
    starts = alarm & ~np.r_[False, alarm[:-1]]        # the first window of each alarm
    eps = find_episodes(y)

    ext = np.zeros(len(te), dtype=bool)
    pre = np.zeros(len(te), dtype=bool)
    for s, e in eps:
        ext[max(0, s - PRE): min(len(te), e + PRE + 1)] = True
        pre[max(0, s - PRE): s] = True
    pre &= (y == 0) & act
    calm = ~ext & act                                  # calm = quiet windows that do have traffic
    calm_hours = calm.sum() * W / 3600
    false_alarms = int((starts & calm).sum())

    for k, (s, e) in enumerate(eps, 1):
        cand = [w for w in range(max(0, s - PRE), e + 1) if alarm[w]]
        duration = (e - s + 1) * W
        base = dict(day=label, episode=k, start_hour=round(te.loc[s, "t_start"] / 3600, 2), duration_s=duration)
        if cand:
            delay = (cand[0] + 1 - s) * W
            ep_rows.append({**base, "detected": True, "seconds_to_alarm": delay,
                            "share_still_ahead": round(max(0.0, 1 - delay / duration), 3)})
        else:
            ep_rows.append({**base, "detected": False, "seconds_to_alarm": np.nan, "share_still_ahead": 0.0})

    auc = np.nan
    if pre.sum() >= 3 and calm.sum() >= 3:
        lab = np.r_[np.ones(pre.sum()), np.zeros(calm.sum())]
        auc = roc_auc_score(lab, np.r_[p[pre], p[calm]])
    day_rows.append(dict(
        day=label, episodes=len(eps), active_hours=round(act.sum() * W / 3600, 1),
        attack_window_recall=round(float((flag & (y == 1)).sum() / max(1, y.sum())), 3),
        false_alarms=false_alarms, false_alarms_per_calm_hour=round(false_alarms / max(calm_hours, 1e-9), 2),
        pre_onset_windows=int(pre.sum()), pre_onset_auc=round(float(auc), 3) if not np.isnan(auc) else np.nan))
    curves[label] = (te["t_start"].values / 3600, p, y, alarm)

eps_df, day_df = pd.DataFrame(ep_rows), pd.DataFrame(day_rows)
eps_df.to_csv("docs/tool3_episodes.csv", index=False)
day_df.to_csv("docs/tool3_summary.csv", index=False)
print("\nPer attack episode (alarm = two high windows in a row):")
print(eps_df.to_string(index=False))
print("\nPer day:")
print(day_df.to_string(index=False))
det = eps_df[eps_df["detected"]]
early = eps_df[eps_df["share_still_ahead"] >= 0.9]
print(f"\nEpisodes with an alarm: {len(det)} of {len(eps_df)}")
print(f"Episodes alarmed with at least 90% of the attack still ahead: {len(early)} of {len(eps_df)}")
if len(early):
    print(f"Median seconds to alarm in those episodes: {early['seconds_to_alarm'].median():.0f}")

# 3. Chart: score over time, true attack periods shaded, alarms marked
fig, axes = plt.subplots(len(DAYS), 1, figsize=(10, 2.0 * len(DAYS) + 1), sharey=True)
for ax, (_, _, label) in zip(np.atleast_1d(axes), DAYS):
    hrs, p, y, alarm = curves[label]
    ax.fill_between(hrs, 0, 1, where=y == 1, color="#e63946", alpha=0.25, step="mid", label="Attack (true)")
    ax.plot(hrs, p, color="#1f6f66", lw=0.9, label="Model score")
    ax.plot(hrs[alarm], np.full(alarm.sum(), 1.05), "|", color="black", ms=6, label="Alarm")
    ax.axhline(THRESH, color="gray", ls="--", lw=0.7)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel(label, fontsize=8)
np.atleast_1d(axes)[0].legend(loc="upper right", ncol=3, fontsize=7)
np.atleast_1d(axes)[-1].set_xlabel("Hour of day (after the clock fix)")
np.atleast_1d(axes)[0].set_title(f"Early warning on CSE-CIC-IDS2018 ({W}-second windows, trained on the other days)", fontsize=10)
plt.tight_layout()
plt.savefig("docs/tool3_chart.png", dpi=150)
print("Saved docs/tool3_episodes.csv, docs/tool3_summary.csv, docs/tool3_chart.png")