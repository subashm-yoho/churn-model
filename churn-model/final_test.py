import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
mean_squared_error, precision_recall_curve, precision_score,
r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from common import ORDINAL_COLS, load_data, make_pipeline, split_data

X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
print("Train", X_train.shape, "| Val", X_val.shape, "| Test", X_test.shape)

Xtr = X_train.drop(columns=["MonthlyCharges", "TotalCharges"])
Xva = X_val.drop(columns=["MonthlyCharges", "TotalCharges"])
r_train = X_train["MonthlyCharges"]
r_val = X_val["MonthlyCharges"]
nominal_cols = [c for c in Xtr.columns if Xtr[c].dtype.kind not in "if" and c not in ORDINAL_COLS]

pre = ColumnTransformer([
("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["SeniorCitizen", "tenure"]),
("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
("enc", OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]]))]), ORDINAL_COLS),
("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), nominal_cols),
])
reg_models = {"Mean baseline": DummyRegressor(strategy="mean"), "LinearRegression": LinearRegression(), "Ridge (alpha=1)": Ridge(alpha=1.0), "Lasso (alpha=0.5)": Lasso(alpha=0.5, max_iter=10000)}
reg_pipes = {n: Pipeline([("preprocess", clone(pre)), ("model", m)]).fit(Xtr, r_train) for n, m in reg_models.items()}
reg_preds = {n: p.predict(Xva) for n, p in reg_pipes.items()}
reg_table = pd.DataFrame([{"Model": n, "MAE": round(mean_absolute_error(r_val, p), 3), "RMSE": round(float(np.sqrt(mean_squared_error(r_val, p))), 3), "R2": round(r2_score(r_val, p), 4)} for n, p in reg_preds.items()])
print("\nRegression (target = MonthlyCharges) on validation set")
print(reg_table.to_string(index=False))
base_rmse = reg_table.loc[0, "RMSE"]
best_rmse = reg_table.loc[1:, "RMSE"].min()
print(f"Best regression model cuts RMSE from {base_rmse} (mean baseline) to {best_rmse}.")
print("R2 = 0 means no better than predicting the mean; the regression models are close to 1.")

log_pipe = make_pipeline(LogisticRegression(max_iter=1000), X).fit(X_train, y_train)
candidates = {"LogisticRegression": log_pipe, "RandomForest (tuned)": joblib.load("best_rf.joblib"), "XGBoost (tuned)": joblib.load("best_xgb.joblib")}
val_auc = {n: roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]) for n, m in candidates.items()}
print("\nValidation ROC-AUC of candidates:")
print("\n".join([f" {n:<22} {a:.4f}" for n, a in val_auc.items()]))
best_name = max(val_auc, key=val_auc.get)
best_model = candidates[best_name]
print(f"Best model chosen on validation: {best_name}")

val_proba = best_model.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, val_proba)
chosen = float(thr[np.where(rec[:-1] >= 0.75)[0][-1]])
print(f"Threshold for >= 75% recall (from validation): {chosen:.3f}")

test_proba = best_model.predict_proba(X_test)[:, 1]
pred_05 = (test_proba >= 0.5).astype(int)
pred_thr = (test_proba >= chosen).astype(int)
print("\n==================== FINAL TEST SET RESULT (used once) ====================")
print(f"Best model: {best_name} | ROC-AUC {roc_auc_score(y_test, test_proba):.4f} | F1 {f1_score(y_test, pred_05):.4f}")
final = pd.DataFrame([{"Threshold": t, "Accuracy": accuracy_score(y_test, p), "Precision": precision_score(y_test, p), "Recall": recall_score(y_test, p), "F1": f1_score(y_test, p)} for t, p in [("0.5", pred_05), (f"{chosen:.3f}", pred_thr)]]).round(4)
print(final.to_string(index=False))
print("Confusion matrix at threshold 0.5 [[TN FP] [FN TP]]:")
print(confusion_matrix(y_test, pred_05))
print(f"Confusion matrix at threshold {chosen:.3f}:")
print(confusion_matrix(y_test, pred_thr))
final.to_csv("task4_final_test.csv", index=False)
reg_table.to_csv("task4_regression_vs_baseline.csv", index=False)
