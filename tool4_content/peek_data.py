import pandas as pd

# Shows how the three MLMA files are organized. It prints counts only, not the tweets themselves.
DATA = "data_text"
for lang in ["en", "fr", "ar"]:
    path = f"{DATA}/{lang}_dataset.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"\n{lang}: file not found at {path}")
        continue
    print(f"\n=== {lang} ===")
    print("rows:", len(df), "| columns:", list(df.columns))
    if "tweet" in df.columns:
        print("empty tweets:", int(df["tweet"].isna().sum()), "| exact duplicate tweets:", int(df["tweet"].duplicated().sum()),
              "| median length (characters):", int(df["tweet"].astype(str).str.len().median()))
    for col in ["sentiment", "directness", "target", "group"]:
        if col in df.columns:
            vc = df[col].fillna("(empty)").value_counts()
            print(f"{col}: {len(vc)} different values. Most common:", vc.head(8).to_dict())
    if "sentiment" in df.columns:
        normal_only = (df["sentiment"].astype(str).str.strip().str.lower() == "normal").mean()
        print(f"share of rows whose sentiment is exactly 'normal': {normal_only:.1%}")