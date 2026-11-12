import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# Tool 4, step 4: a bias check with HateCheck (functional tests for hate speech models).
# Models are trained on MLMA (as before) and tested on HateCheck, in English, French and Arabic.
# The key question: does the model wrongly flag NON-hateful statements that merely mention a group
# (for example neutral or positive statements, labeled ident_* in HateCheck)?
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA = "data_text"
LANGS = ["en", "fr", "ar"]
NAMES = {"en": "English", "fr": "French", "ar": "Arabic"}
TARGETS = ["any_toxic", "severe"]
MIN_IDENT = 10
_model = None


def load_mlma(lang):
    df = pd.read_csv(f"{DATA}/{lang}_dataset.csv")
    df = df.dropna(subset=["tweet"]).drop_duplicates(subset=["tweet"]).reset_index(drop=True)
    s = df["sentiment"].astype(str).str.strip().str.lower()
    df["any_toxic"] = (s != "normal").astype(int)
    df["severe"] = s.str.split("_").apply(lambda p: int("hateful" in p or "abusive" in p))
    return df


def load_hc(lang):
    path = f"{DATA}/hatecheck_{lang}.csv"
    if not os.path.exists(path):
        sys.exit(f"Missing {path}. Download it first (see the instructions).")
    df = pd.read_csv(path)
    missing = [c for c in ["functionality", "test_case", "label_gold"] if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing columns {missing}. Columns found: {list(df.columns)}")
    if "target_ident" not in df.columns:
        df["target_ident"] = np.nan
    if "disagreement_in_case" in df.columns:      # drop cases where the annotators' majority disagrees with the gold label
        df = df[~(df["disagreement_in_case"].astype(str).str.lower() == "true")]
    df = df.dropna(subset=["test_case"]).reset_index(drop=True)
    df["hateful"] = (df["label_gold"].astype(str).str.lower() == "hateful").astype(int)
    df["ident_stmt"] = (df["functionality"].astype(str).str.startswith("ident_")) & (df["hateful"] == 0)
    df["target_ident"] = df["target_ident"].fillna("none").astype(str).str.strip().str.lower()
    return df


def embed(texts, cache):
    global _model
    if os.path.exists(cache):
        E = np.load(cache)
        if len(E) == len(texts):
            return E
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading {MODEL}...")
        _model = SentenceTransformer(MODEL)
    print(f"Embedding {len(texts)} texts...")
    E = _model.encode(list(texts), batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    np.save(cache, E)
    return E


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    d = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def fit_wordcount(texts, y):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    X = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
    return lambda t: clf.predict_proba(hstack([word.transform(t), char.transform(t)]).tocsr())[:, 1]


mlma = {l: load_mlma(l) for l in LANGS}
hc = {l: load_hc(l) for l in LANGS}
for l in LANGS:
    print(f"{NAMES[l]}: {len(hc[l])} HateCheck cases | {int(hc[l]['hateful'].sum())} hateful | "
          f"{int(hc[l]['ident_stmt'].sum())} non-hateful statements that mention a group | "
          f"target groups: {sorted(set(hc[l]['target_ident']) - {'none'})}")

m_emb = {l: np.load(f"{DATA}/emb_{l}.npy") for l in LANGS}
for l in LANGS:
    if len(m_emb[l]) != len(mlma[l]):
        sys.exit(f"Saved embeddings for {l} do not match the data. Run embed_baseline.py first.")
h_emb = {l: embed(hc[l]["test_case"].astype(str).tolist(), f"{DATA}/hc_emb_{l}.npy") for l in LANGS}

rows, grp_rows = [], []
for target in TARGETS:
    all_X = np.vstack([m_emb[l] for l in LANGS])
    all_y = np.concatenate([mlma[l][target].values for l in LANGS])
    emb_all = LogisticRegression(max_iter=2000, class_weight="balanced").fit(all_X, all_y)
    for l in LANGS:
        emb_same = LogisticRegression(max_iter=2000, class_weight="balanced").fit(m_emb[l], mlma[l][target].values)
        wc = fit_wordcount(mlma[l]["tweet"].astype(str), mlma[l][target].values)
        texts = hc[l]["test_case"].astype(str)
        scores = {
            "wordcount (same language)": wc(texts),
            "embeddings (same language)": emb_same.predict_proba(h_emb[l])[:, 1],
            "embeddings (all three)": emb_all.predict_proba(h_emb[l])[:, 1],
        }
        d = hc[l]
        for mname, sc in scores.items():
            flag = sc >= 0.5
            hate, non, ident = d["hateful"] == 1, d["hateful"] == 0, d["ident_stmt"]
            k_i, n_i = int(flag[ident].sum()), int(ident.sum())
            lo, hi = wilson(k_i, n_i)
            rows.append(dict(task=target, model=mname, language=NAMES[l],
                             hateful_cases=int(hate.sum()), hateful_flagged=round(float(flag[hate].mean()), 3),
                             nonhateful_cases=int(non.sum()), nonhateful_flagged=round(float(flag[non].mean()), 3),
                             identity_statements=n_i, identity_flagged=round(k_i / n_i, 3) if n_i else np.nan,
                             identity_low=round(lo, 3), identity_high=round(hi, 3)))
            for g, sub in d[ident].assign(flag=flag[ident.values]).groupby("target_ident"):
                if len(sub) >= MIN_IDENT and g != "none":
                    k = int(sub["flag"].sum())
                    lo, hi = wilson(k, len(sub))
                    grp_rows.append(dict(task=target, model=mname, language=NAMES[l], group=g,
                                         statements=len(sub), flagged=round(k / len(sub), 3), low=round(lo, 3), high=round(hi, 3)))

res, grp = pd.DataFrame(rows), pd.DataFrame(grp_rows)
res.to_csv("docs/tool4_hatecheck_results.csv", index=False)
grp.to_csv("docs/tool4_hatecheck_groups.csv", index=False)
pd.set_option("display.width", 220)
print("\nOverall (share flagged as toxic or severe). Ideal: high on hateful cases, low on non-hateful and identity statements.")
print(res.to_string(index=False))
for target in TARGETS:
    g = grp[(grp["task"] == target) & (grp["model"] == "embeddings (all three)")]
    print(f"\nNon-hateful statements that mention a group, flagged by group (task {target}, embeddings all three):")
    print(g.drop(columns=["task", "model"]).to_string(index=False) if len(g) else "No group had enough statements.")
print("\nSaved docs/tool4_hatecheck_results.csv and docs/tool4_hatecheck_groups.csv")