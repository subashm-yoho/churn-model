import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

from features_demo import add_features

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RS = 42

ORDINAL_COLS = ["Contract"]
NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
            "signup_year", "signup_month", "charge_per_tenure", "log_total_charges"]


def load_data():
    df = pd.read_csv(CSV_PATH).drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    return X, y


def split_data(X, y):
    """60/20/20, stratified, random_state=42 (same as Task 1)."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RS)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RS)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocess(X):
    nominal_cols = [c for c in X.columns
                    if X[c].dtype == "object" and c not in ORDINAL_COLS]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUM_COLS),
        ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(
                              categories=[["Month-to-month", "One year", "Two year"]]))]),
         ORDINAL_COLS),
        ("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(handle_unknown="ignore"))]),
         nominal_cols),
    ])


def make_pipeline(model, X):
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", build_preprocess(X)),
        ("model", model),
    ])
