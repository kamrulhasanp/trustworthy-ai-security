import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Tool 5, part 2: spelling tricks against the Tool 4 text detectors (MLMA, English, French, Arabic).
# For each language: take the test tweets that are toxic AND that the clean model catches, disguise the words,
# and measure the share that are STILL caught. Nothing here prints tweets.
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA = "data_text"
LANGS = ["en", "fr", "ar"]
NAMES = {"en": "English", "fr": "French", "ar": "Arabic"}
SEED = 42
SHARE = 0.3                     # share of words changed by swap, delete, space, repeat and tatweel
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}
KINDS = {"en": ["swap", "delete", "space", "repeat", "leet"],
         "fr": ["swap", "delete", "space", "repeat", "leet"],
         "ar": ["swap", "delete", "space", "repeat", "tatweel"]}
_model = None


def perturb(text, kind, rng):
    out = []
    for w in str(text).split():
        if kind == "leet":
            out.append("".join(LEET.get(ch.lower(), ch) for ch in w))
        elif len(w) >= 4 and rng.random() < SHARE:
            i = int(rng.integers(1, len(w) - 1))
            if kind == "swap":
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif kind == "delete":
                w = w[:i] + w[i + 1:]
            elif kind == "space":
                w = w[:i] + " " + w[i:]
            elif kind == "repeat":
                w = w[:i] + w[i] * 3 + w[i + 1:]
            elif kind == "tatweel":
                w = w[:i] + "\u0640" + w[i:]
            out.append(w)
        else:
            out.append(w)
    return " ".join(out)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    d = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def embed(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL)
    return _model.encode(list(texts), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)


rows = []
for lang in LANGS:
    df = pd.read_csv(f"{DATA}/{lang}_dataset.csv")
    df = df.dropna(subset=["tweet"]).drop_duplicates(subset=["tweet"]).reset_index(drop=True)
    df["toxic"] = (df["sentiment"].astype(str).str.strip().str.lower() != "normal").astype(int)
    E = np.load(f"{DATA}/emb_{lang}.npy")
    assert len(E) == len(df), "Saved embeddings do not match. Run embed_baseline.py first."
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=df["toxic"])
    texts = df["tweet"].astype(str)

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    Xtr = hstack([word.fit_transform(texts.iloc[tr]), char.fit_transform(texts.iloc[tr])]).tocsr()
    wc = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, df["toxic"].iloc[tr])
    em = LogisticRegression(max_iter=2000, class_weight="balanced").fit(E[tr], df["toxic"].iloc[tr])

    def flag_wc(t):
        return wc.predict(hstack([word.transform(t), char.transform(t)]).tocsr()) == 1

    tox = te[df["toxic"].values[te] == 1]
    t_clean = texts.iloc[tox].tolist()
    caught = {"word counts": flag_wc(t_clean), "embeddings": em.predict(E[tox]) == 1}
    for kind in KINDS[lang]:
        rng = np.random.default_rng(1)
        t_pert = [perturb(t, kind, rng) for t in t_clean]
        after = {"word counts": flag_wc(t_pert), "embeddings": em.predict(embed(t_pert)) == 1}
        for mname in ["word counts", "embeddings"]:
            c = caught[mname]
            k, n = int((after[mname] & c).sum()), int(c.sum())
            lo, hi = wilson(k, n)
            rows.append(dict(language=NAMES[lang], model=mname, trick=kind, tweets_caught_clean=n,
                             still_caught=round(k / n, 3), low=round(lo, 3), high=round(hi, 3)))

res = pd.DataFrame(rows)
res.to_csv("docs/tool5_text_results.csv", index=False)
print("Share of toxic test tweets that were caught when clean AND are still caught after the spelling trick")
for lang in [NAMES[l] for l in LANGS]:
    t = res[res["language"] == lang].pivot(index="model", columns="trick", values="still_caught")
    t["worst"] = t.min(axis=1)
    print(f"\n{lang} (tweets caught when clean: " + ", ".join(
        f"{m} {int(res[(res['language'] == lang) & (res['model'] == m)]['tweets_caught_clean'].iloc[0])}" for m in ["word counts", "embeddings"]) + ")")
    print(t.round(3).to_string())
print("\nRobustness score by model (lowest share still caught over all languages and tricks):")
print(res.groupby("model")["still_caught"].min().round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, l in zip(axes, LANGS):
    sub = res[res["language"] == NAMES[l]]
    kinds = KINDS[l]
    x = np.arange(len(kinds))
    for j, (m, c) in enumerate([("word counts", "#e63946"), ("embeddings", "#2a9d8f")]):
        v = [sub[(sub["model"] == m) & (sub["trick"] == k)]["still_caught"].iloc[0] for k in kinds]
        ax.bar(x + (j - 0.5) * 0.38, v, 0.38, label=m, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, fontsize=8)
    ax.set_title(NAMES[l], fontsize=10)
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Still caught after the trick")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("docs/tool5_text_chart.png", dpi=150)
print("\nSaved docs/tool5_text_results.csv and docs/tool5_text_chart.png")