import warnings

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Why does the word-count model miss so many toxic Arabic tweets about women?
# This prints counts and shares only. It never prints tweets.
DATA = "data_text"
SEED = 42
df = pd.read_csv(f"{DATA}/ar_dataset.csv")
df = df.dropna(subset=["tweet"]).drop_duplicates(subset=["tweet"]).reset_index(drop=True)
s = df["sentiment"].astype(str).str.strip().str.lower()
df["sentiment"] = s
df["toxic"] = (s != "normal").astype(int)
df["group"] = df["group"].fillna("none").astype(str).str.strip().str.lower()
tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["toxic"])

word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
X = hstack([word.fit_transform(tr["tweet"].astype(str)), char.fit_transform(tr["tweet"].astype(str))]).tocsr()
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, tr["toxic"])
te = te.assign(pred=clf.predict(hstack([word.transform(te["tweet"].astype(str)), char.transform(te["tweet"].astype(str))]).tocsr()))

print("All Arabic tweets, by group: how many, share toxic, share direct, share with target 'gender'")
rows = []
for g in ["women", "individual", "other"]:
    sub = df[df["group"] == g]
    rows.append(dict(group=g, tweets=len(sub), toxic_share=round(sub["toxic"].mean(), 3),
                     direct_share=round((sub["directness"].astype(str).str.lower() == "direct").mean(), 3),
                     target_gender_share=round((sub["target"].astype(str).str.lower() == "gender").mean(), 3)))
print(pd.DataFrame(rows).to_string(index=False))

tox = te[te["toxic"] == 1]
for g in ["women", "individual", "other"]:
    sub = tox[tox["group"] == g]
    print(f"\n--- Toxic test tweets about '{g}': {len(sub)}, caught {sub['pred'].mean():.1%} ---")
    by_dir = sub.groupby(sub["directness"].astype(str).str.lower())["pred"].agg(tweets="size", caught="mean").round(3)
    print("By directness:")
    print(by_dir.to_string())
    by_lab = sub.groupby("sentiment")["pred"].agg(tweets="size", caught="mean").round(3).sort_values("tweets", ascending=False).head(6)
    print("By hostility label (six most common):")
    print(by_lab.to_string())