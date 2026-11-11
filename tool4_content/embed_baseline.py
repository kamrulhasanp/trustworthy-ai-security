import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Tool 4, step 3: multilingual sentence embeddings + a simple classifier, on the same MLMA data and the same splits as baseline_v2.py.
# The embedding model turns a tweet in any of its languages into numbers that carry meaning, so a classifier trained on
# one language can work on another. It runs on the CPU. The embeddings are saved, so the slow step happens only once.
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA = "data_text"
LANGS = ["en", "fr", "ar"]
NAMES = {"en": "English", "fr": "French", "ar": "Arabic"}
TARGETS = ["any_toxic", "severe"]
MIN_GROUP = 20
SEED = 42
_model = None


def load(lang):
    df = pd.read_csv(f"{DATA}/{lang}_dataset.csv")
    df = df.dropna(subset=["tweet"]).drop_duplicates(subset=["tweet"]).reset_index(drop=True)
    s = df["sentiment"].astype(str).str.strip().str.lower()
    df["any_toxic"] = (s != "normal").astype(int)
    df["severe"] = s.str.split("_").apply(lambda p: int("hateful" in p or "abusive" in p))
    df["group"] = df["group"].fillna("none").astype(str).str.strip().str.lower()
    return df


def embeddings(lang, df):
    global _model
    path = f"{DATA}/emb_{lang}.npy"
    if os.path.exists(path):
        E = np.load(path)
        if len(E) == len(df):
            return E
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading {MODEL} (the first time it downloads the model)...")
        _model = SentenceTransformer(MODEL)
    print(f"Embedding {len(df)} {NAMES[lang]} tweets...")
    E = _model.encode(df["tweet"].astype(str).tolist(), batch_size=64, show_progress_bar=True,
                      convert_to_numpy=True, normalize_embeddings=True)
    np.save(path, E)
    return E


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    d = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def boot_auc(y, score, n=300, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(set(y[s])) > 1:
            vals.append(roc_auc_score(y[s], score[s]))
    return np.percentile(vals, [2.5, 97.5])


def trivial_macro_f1(y):
    """Best score from a model that always gives the same answer (always yes, or always no)."""
    p = y.mean()
    return (2 * p / (1 + p)) / 2 if p >= 0.5 else (2 * (1 - p) / (2 - p)) / 2


data = {l: load(l) for l in LANGS}
emb = {l: embeddings(l, data[l]) for l in LANGS}
split = {}
for l in LANGS:
    tr, te = train_test_split(np.arange(len(data[l])), test_size=0.2, random_state=SEED, stratify=data[l]["any_toxic"])
    split[l] = (tr, te)

trainsets = {NAMES[l]: [l] for l in LANGS}
trainsets["All three"] = LANGS
rows, grp_rows, auc_mats = [], [], {}
for target in TARGETS:
    mat = pd.DataFrame(index=list(trainsets), columns=[NAMES[l] for l in LANGS], dtype=float)
    for tname, tl in trainsets.items():
        X = np.vstack([emb[l][split[l][0]] for l in tl])
        y = np.concatenate([data[l][target].values[split[l][0]] for l in tl])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, y)
        for l in LANGS:
            idx = split[l][1]
            yt = data[l][target].values[idx]
            sc = clf.predict_proba(emb[l][idx])[:, 1]
            pred = (sc >= 0.5).astype(int)
            auc = roc_auc_score(yt, sc)
            lo, hi = boot_auc(yt, sc)
            mat.loc[tname, NAMES[l]] = auc
            rows.append(dict(target=target, trained_on=tname, tested_on=NAMES[l], n_test=len(yt),
                             positive_share=round(yt.mean(), 3),
                             macro_f1=round(f1_score(yt, pred, average="macro"), 3),
                             best_trivial_macro_f1=round(trivial_macro_f1(yt), 3),
                             auc=round(auc, 3), auc_low=round(lo, 3), auc_high=round(hi, 3),
                             positives_caught=round(float(pred[yt == 1].mean()), 3),
                             negatives_left_alone=round(float(1 - pred[yt == 0].mean()), 3)))
    auc_mats[target] = mat
    for l in LANGS:   # which target groups are missed most? (same-language model)
        tr, te = split[l]
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(emb[l][tr], data[l][target].values[tr])
        t = data[l].iloc[te].assign(pred=(clf.predict_proba(emb[l][te])[:, 1] >= 0.5).astype(int))
        for g, sub in t[t[target] == 1].groupby("group"):
            if len(sub) >= MIN_GROUP and g != "none":
                k = int(sub["pred"].sum())
                lo, hi = wilson(k, len(sub))
                grp_rows.append(dict(target=target, language=NAMES[l], group=g, positive_tweets=len(sub),
                                     caught=round(k / len(sub), 3), low=round(lo, 3), high=round(hi, 3)))

res, grp = pd.DataFrame(rows), pd.DataFrame(grp_rows)
res.to_csv("docs/tool4_embed_results.csv", index=False)
grp.to_csv("docs/tool4_embed_group_recall.csv", index=False)

old = None
if os.path.exists("docs/tool4_v2_results.csv"):
    old = pd.read_csv("docs/tool4_v2_results.csv")[["target", "trained_on", "tested_on", "auc"]].rename(columns={"auc": "auc_wordcounts"})

for target in TARGETS:
    print(f"\n=== Task: {target} ===")
    same = res[(res["target"] == target) & (res["trained_on"] == res["tested_on"])]
    print("Same language (compare macro F1 with the best trivial model; AUC 0.5 means no skill):")
    print(same.drop(columns=["target", "trained_on"]).to_string(index=False))
    print("\nAUC with embeddings (rows: trained on, columns: tested on):")
    print(auc_mats[target].round(3).to_string())
    if old is not None:
        cmp = res[res["target"] == target][["target", "trained_on", "tested_on", "auc"]].merge(old, on=["target", "trained_on", "tested_on"])
        cmp = cmp.rename(columns={"auc": "auc_embeddings"}).drop(columns=["target"])
        print("\nAUC, word-count model (version 2) against embeddings:")
        print(cmp.to_string(index=False))
    g = grp[grp["target"] == target]
    print("\nShare of positive tweets caught, by target group (embeddings, same-language model):")
    print(g.drop(columns=["target"]).to_string(index=False) if len(g) else "No group had enough tweets.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, target in zip(axes, TARGETS):
    m = auc_mats[target]
    ax.imshow(m.values.astype(float), vmin=0.4, vmax=1.0, cmap="Greens")
    ax.set_xticks(range(3)); ax.set_xticklabels(m.columns)
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center")
    ax.set_title(f"AUC with multilingual embeddings, task: {target}", fontsize=9)
    ax.set_xlabel("Tested on")
axes[0].set_ylabel("Trained on")
plt.tight_layout()
plt.savefig("docs/tool4_embed_chart.png", dpi=150)
print("\nSaved docs/tool4_embed_results.csv, docs/tool4_embed_group_recall.csv, docs/tool4_embed_chart.png")