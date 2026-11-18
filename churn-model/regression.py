import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from common import ORDINAL_COLS, load_data, split_data

X, y_churn = load_data()
X_train, X_val, X_test, _, _, _ = split_data(X, y_churn) # test set stays aside

# TotalCharges is left out so the target does not leak
Xtr = X_train.drop(columns=["MonthlyCharges", "TotalCharges"])
Xva = X_val.drop(columns=["MonthlyCharges", "TotalCharges"])
y_train = X_train["MonthlyCharges"]
y_val = X_val["MonthlyCharges"]

num_cols = ["SeniorCitizen", "tenure"]
nominal_cols = [c for c in Xtr.columns if Xtr[c].dtype.kind not in "if" and c not in ORDINAL_COLS]
print("Numeric:", num_cols, "| Ordinal:", ORDINAL_COLS, "| Nominal:", len(nominal_cols), "columns")

pre = ColumnTransformer([
("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
("enc", OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]]))]), ORDINAL_COLS),
("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), nominal_cols),
])

models = {"LinearRegression": LinearRegression(), "Ridge (alpha=1)": Ridge(alpha=1.0), "Lasso (alpha=0.5)": Lasso(alpha=0.5, max_iter=10000)}

pipes = {n: Pipeline([("preprocess", clone(pre)), ("model", m)]).fit(Xtr, y_train) for n, m in models.items()}
preds = {n: p.predict(Xva) for n, p in pipes.items()}

results = pd.DataFrame([{"Model": n, "Val R2": round(r2_score(y_val, p), 4), "Val MAE": round(mean_absolute_error(y_val, p), 3), "Val RMSE": round(float(np.sqrt(mean_squared_error(y_val, p))), 3)} for n, p in preds.items()])
print("\nRegression results (target = MonthlyCharges, validation set)")
print(results.to_string(index=False))

feat_names = [x.split("__", 1)[-1] for x in pipes["LinearRegression"].named_steps["preprocess"].get_feature_names_out()]
coef_df = pd.DataFrame({n: p.named_steps["model"].coef_ for n, p in pipes.items()}, index=feat_names)

top = coef_df.reindex(coef_df["LinearRegression"].abs().sort_values(ascending=False).index).head(10)
print("\nTop 10 coefficients (by size in LinearRegression):")
print(top.round(2))

print("\nTotal features:", len(coef_df))
print("\n".join([f"{n:<20} zero coefficients: {int((coef_df[n].abs() < 1e-8).sum())}" for n in coef_df.columns]))
print("\nRidge shrinks coefficients but keeps all of them; Lasso can push some exactly to zero")
print("(feature selection). Higher alpha = more shrinkage = more zeros.")

results.to_csv("task3_regression_results.csv", index=False)
