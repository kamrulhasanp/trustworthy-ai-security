import glob
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load(pattern, sample=None):
    files = glob.glob(pattern)
    df = pd.concat([pd.read_csv(f) for f in files])
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)
    y = (df["Label"] != "BENIGN").astype(int)
    X = df.drop(columns=["Label"])
    return X, y


if __name__ == "__main__":
    # Train on Monday (normal) + Tuesday (first attacks)
    X1, y1 = load("data/*Monday*.csv", sample=100000)
    X2, y2 = load("data/*Tuesday*.csv", sample=100000)
    X = pd.concat([X1, X2])
    y = pd.concat([y1, y2])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(n_estimators=200, max_depth=6, n_jobs=-1)
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))
    joblib.dump(model, "models/model_v1.joblib")