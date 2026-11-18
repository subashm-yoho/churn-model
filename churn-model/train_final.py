import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform, loguniform

from sklearn.model_selection import (train_test_split, StratifiedKFold,
RandomizedSearchCV, cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
ConfusionMatrixDisplay, precision_score,
recall_score, classification_report)
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = "outputs"
TARGET_RECALL = 0.75
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop(columns=["customerID"], errors="ignore")
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

prep = ColumnTransformer([
("num", Pipeline([("imp", SimpleImputer(strategy="median")),
("sc", StandardScaler())]), num_cols),
("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

spw = (y_train == 0).sum() / (y_train == 1).sum()
pipe = Pipeline([
("prep", prep),
("model", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
scale_pos_weight=spw, eval_metric="auc",
random_state=RANDOM_STATE, n_jobs=-1)),
])

space = {
"model__n_estimators": randint(200, 700),
"model__learning_rate": loguniform(0.01, 0.2),
"model__max_depth": randint(2, 7),
"model__min_child_weight": randint(1, 10),
"model__subsample": uniform(0.6, 0.4),
"model__colsample_bytree": uniform(0.6, 0.4),
"model__gamma": uniform(0, 5),
"model__reg_lambda": loguniform(0.1, 10),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(pipe, space, n_iter=30, scoring="roc_auc", cv=cv,
random_state=RANDOM_STATE, n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_
print("Best params:", search.best_params_)
print(f"Best CV ROC-AUC: {search.best_score_:.4f}\n")

oof = cross_val_predict(best, X_train, y_train, cv=cv,
method="predict_proba", n_jobs=-1)[:, 1]
grid = np.linspace(0.05, 0.95, 181)
ok = [t for t in grid if recall_score(y_train, (oof >= t).astype(int)) >= TARGET_RECALL]
threshold = float(max(ok)) if ok else 0.5
print(f"Chosen threshold: {threshold:.2f}\n")

p_test = best.predict_proba(X_test)[:, 1]
pred = (p_test >= threshold).astype(int)

auc = roc_auc_score(y_test, p_test)
f1 = f1_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()

report = f"""FINAL TEST SET EVALUATION (XGBoost tuned, threshold={threshold:.2f})
ROC-AUC : {auc:.4f}
F1 : {f1:.4f}
Precision : {prec:.4f}
Recall : {rec:.4f}

Confusion matrix (rows = actual, cols = predicted)
Pred No Pred Yes
Actual No {tn:6d} {fp:6d}
Actual Yes {fn:6d} {tp:6d}

TN={tn} FP={fp} FN={fn} TP={tp}
Churners caught: {tp} of {tp + fn} ({rec:.1%})
"""
print(report)
print(classification_report(y_test, pred, target_names=["No churn", "Churn"]))
with open(f"{OUT}/final_metrics.txt", "w") as f:
 f.write(report)

disp = ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix (test set, thr={threshold:.2f})")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.close()

joblib.dump({"pipeline": best,
"threshold": threshold,
"feature_columns": list(X.columns)},
f"{OUT}/churn_model.joblib")
print(f"Saved: {OUT}/churn_model.joblib, confusion_matrix.png, final_metrics.txt")
