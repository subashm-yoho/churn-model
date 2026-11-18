import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
OrdinalEncoder, StandardScaler)

from features_demo import add_features

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "churn_pipeline.joblib"
RS = 42
 
df = pd.read_csv(CSV_PATH).drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])

X_temp, X_test, y_temp, y_test = train_test_split(
X, y, test_size=0.20, stratify=y, random_state=RS)
X_train, X_val, y_train, y_val = train_test_split(
X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RS)


ordinal_cols = ["Contract"]
nominal_cols = [c for c in X.columns if X[c].dtype == "object" and c not in ordinal_cols]
num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
"signup_year", "signup_month", "charge_per_tenure", "log_total_charges"]


def build_pipeline(num_imputer):
 preprocess = ColumnTransformer([
("num", Pipeline([("imp", num_imputer),
("scale", StandardScaler())]), num_cols),
("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
("enc", OrdinalEncoder(
categories=[["Month-to-month", "One year", "Two year"]]))]),
ordinal_cols),
("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
("enc", OneHotEncoder(handle_unknown="ignore"))]),
nominal_cols),
])
 return Pipeline([
("features", FunctionTransformer(add_features)), # 3+ new features
("preprocess", preprocess),
("model", LogisticRegression(max_iter=1000)),
])

pipe_median = build_pipeline(SimpleImputer(strategy="median"))
pipe_knn = build_pipeline(KNNImputer(n_neighbors=5))

cv_median = cross_val_score(pipe_median, X_train, y_train, cv=5, scoring="accuracy")
cv_knn = cross_val_score(pipe_knn, X_train, y_train, cv=5, scoring="accuracy")
print(f"Pipeline CV accuracy (median imputer): {cv_median.mean():.4f} +/- {cv_median.std():.4f}")
print(f"Pipeline CV accuracy (KNN imputer) : {cv_knn.mean():.4f} +/- {cv_knn.std():.4f}")

pipe = pipe_median.fit(X_train, y_train)
print(f"Validation accuracy: {pipe.score(X_val, y_val):.4f} (test set untouched)")


joblib.dump(pipe, MODEL_PATH)
loaded = joblib.load(MODEL_PATH)
print(f"\nSaved and reloaded: {MODEL_PATH}")

new_rows = pd.DataFrame([
["Female", 0, "Yes", "No", 2, "Yes", "No", "Fiber optic", "No", "No", "No", "No",
"No", "No", "Month-to-month", "Yes", "Electronic check", 70.7, 151.65],
["Male", 0, "Yes", "Yes", 60, "Yes", "Yes", "DSL", "Yes", "Yes", "Yes", "Yes",
"No", "No", "Two year", "No", "Bank transfer (automatic)", 65.5, 3900.0],
["Female", 1, "No", "No", 0, "No", "No phone service", "DSL", "No", "No", "No", "No",
"No", "No", "Month-to-month", "Yes", "Mailed check", 29.85, np.nan],
], columns=X.columns)

pred = loaded.predict(new_rows)
proba = loaded.predict_proba(new_rows)[:, 1]
print("Predictions for 3 new rows:", pred)
print("Churn probabilities :", np.round(proba, 3))
print("(1 = will churn, 0 = will stay; row 3 had missing TotalCharges, pipeline handled it)")
