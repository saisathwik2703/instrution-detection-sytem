"""
preprocessing.py
-----------------
Loads the raw NSL-KDD text files and turns them into a clean, fully numeric
feature matrix that scikit-learn models can train on.

Steps performed:
1. Read the headerless CSV/text files and attach the 41 feature names + label.
2. Drop the optional 43rd "difficulty" column if present.
3. Collapse the ~22 raw attack labels into a binary target:
       0 = normal traffic
       1 = attack (any of the DoS / Probe / R2L / U2R types)
4. Label-encode the three categorical columns (protocol_type, service, flag).
5. Scale the numeric columns with StandardScaler.
6. Save the fitted encoders + scaler with joblib so the Streamlit app can
   apply the EXACT same transformation to a single live input row.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

from constants import COLUMN_NAMES, CATEGORICAL_COLS, RANDOM_STATE

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "..", "data")
MODEL_DIR = os.path.join(THIS_DIR, "..", "models")


def load_raw(path):
    """Read a headerless NSL-KDD file and attach column names."""
    df = pd.read_csv(path, header=None)
    # Some distributions have 43 columns (41 features + label + difficulty).
    if df.shape[1] == len(COLUMN_NAMES) + 1:
        df = df.iloc[:, :-1]
    df.columns = COLUMN_NAMES
    return df


def binarize_label(df):
    """Convert the raw multi-class label into binary: 0 = normal, 1 = attack."""
    df = df.copy()
    df["binary_label"] = (df["label"].str.strip().str.lower() != "normal").astype(int)
    return df


def fit_transform_train(df):
    """
    Fit label encoders + scaler on the TRAINING data and transform it.
    Returns (X, y, encoders, scaler, feature_columns).
    """
    df = binarize_label(df)
    y = df["binary_label"].values

    X = df.drop(columns=["label", "binary_label"]).copy()

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    feature_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, encoders, scaler, feature_columns


def transform_with_fitted(df, encoders, scaler, feature_columns):
    """
    Apply ALREADY-FITTED encoders/scaler to new data (e.g. the held-out test
    set, or a single row submitted through the Streamlit app). Any category
    value not seen during training is safely mapped to a fallback code
    instead of raising an error, since real-world / test traffic can contain
    services or flags the training set never saw.
    """
    df = df.copy()
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    if "binary_label" in df.columns:
        df = df.drop(columns=["binary_label"])

    X = df[feature_columns].copy()

    for col in CATEGORICAL_COLS:
        le = encoders[col]
        known = set(le.classes_)
        X[col] = X[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
        X[col] = le.transform(X[col])

    X_scaled = scaler.transform(X)
    return X_scaled


def run_preprocessing(train_path, test_path):
    """Full pipeline: load both files, fit on train, transform both, save artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df = load_raw(train_path)
    test_df = load_raw(test_path)

    X_train, y_train, encoders, scaler, feature_columns = fit_transform_train(train_df)

    test_df_bin = binarize_label(test_df)
    y_test = test_df_bin["binary_label"].values
    X_test = transform_with_fitted(test_df_bin, encoders, scaler, feature_columns)

    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.joblib"))

    print(f"Train shape: {X_train.shape}  |  Attack ratio: {y_train.mean():.3f}")
    print(f"Test shape : {X_test.shape}  |  Attack ratio: {y_test.mean():.3f}")

    return X_train, y_train, X_test, y_test, encoders, scaler, feature_columns


if __name__ == "__main__":
    train_path = os.path.join(DATA_DIR, "KDDTrain.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest.txt")
    run_preprocessing(train_path, test_path)
