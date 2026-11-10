import glob
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# Looks at how the Timestamp column of two daily files is written, and how many rows land on the expected date.
CHECK = [("Friday-16-02", "2018-02-16"), ("Wednesday-21-02", "2018-02-21")]

for prefix, expected in CHECK:
    files = glob.glob(f"data2018/{prefix}*.csv")
    if not files:
        print(prefix, "not found")
        continue
    df = pd.read_csv(files[0], usecols=lambda c: c.strip() in ("Timestamp", "Label"), low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df["Label"] != "Label"]
    print(f"\n{prefix}: {len(df):,} rows")
    print("Sample timestamps (one every 100,000 rows):", df["Timestamp"].iloc[::100000].tolist())
    for dayfirst in (True, False):
        t = pd.to_datetime(df["Timestamp"], dayfirst=dayfirst, errors="coerce")
        on_day = (t.dt.date.astype(str) == expected).mean()
        print(f"  dayfirst={dayfirst}: cannot read {t.isna().mean():.1%}, on {expected}: {on_day:.1%}")
        print("   most common dates:", t.dt.date.astype(str).value_counts().head(4).to_dict())
    t = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    print("  rows per hour of the day (dayfirst=True):", t.dt.hour.value_counts().sort_index().to_dict())
    print("  label counts:", df["Label"].value_counts().head(4).to_dict())