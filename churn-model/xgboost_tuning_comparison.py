import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform, loguniform

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_val_score,
                                     cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve,
                             precision_score, recall_score)
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"  
OUT = "outputs"
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

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

spw = (y_train == 0).sum() / (y_train == 1).sum()

pipes = {
    "Logistic Regression": Pipeline([
        ("prep", make_preprocessor()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=RANDOM_STATE))]),
    "Random Forest": Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                         random_state=RANDOM_STATE, n_jobs=-1))]),
    "XGBoost": Pipeline([
        ("prep", make_preprocessor()),
        ("model", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                scale_pos_weight=spw, eval_metric="auc",
                                random_state=RANDOM_STATE, n_jobs=-1))]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

base_cv = {}
for name, pipe in pipes.items():
    base_cv[name] = cross_val_score(pipe, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=-1).mean()
print("Baseline CV ROC-AUC:")
print(pd.Series(base_cv).sort_values(ascending=False).round(4), "\n")

top2 = sorted(base_cv, key=base_cv.get, reverse=True)[:2]
print("Tuning these 2 models:", top2, "\n")

param_space = {
    "Logistic Regression": {
        "model__C": loguniform(1e-3, 1e2),
        "model__penalty": ["l2"],
    },
    "Random Forest": {
        "model__n_estimators": randint(200, 600),
        "model__max_depth": randint(4, 20),
        "model__min_samples_leaf": randint(1, 15),
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "XGBoost": {
        "model__n_estimators": randint(200, 700),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 7),
        "model__min_child_weight": randint(1, 10),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__gamma": uniform(0, 5),
        "model__reg_lambda": loguniform(0.1, 10),
    },
}

final_models, tuned_cv = {}, {}
for name in pipes:
    if name in top2:
        search = RandomizedSearchCV(pipes[name], param_space[name], n_iter=30,
                                    scoring="roc_auc", cv=cv,
                                    random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_train, y_train)
        final_models[name + " (tuned)"] = search.best_estimator_
        tuned_cv[name + " (tuned)"] = search.best_score_
        print(f"{name} best params: {search.best_params_}")
        print(f"{name} tuned CV AUC: {search.best_score_:.4f}\n")
    else:
        pipes[name].fit(X_train, y_train)
        final_models[name] = pipes[name]
        tuned_cv[name] = base_cv[name]

rows, probas = [], {}
for name, m in final_models.items():
    p = m.predict_proba(X_test)[:, 1]
    probas[name] = p
    rows.append({
        "Model": name,
        "CV ROC-AUC (train)": round(tuned_cv[name], 4),
        "Test ROC-AUC": round(roc_auc_score(y_test, p), 4),
        "Test F1 (thr=0.5)": round(f1_score(y_test, (p >= 0.5).astype(int)), 4),
    })
table = pd.DataFrame(rows).sort_values("Test ROC-AUC", ascending=False)
table.to_csv(f"{OUT}/comparison_table.csv", index=False)
print("=== COMPARISON TABLE ===")
print(table.to_string(index=False), "\n")

plt.figure(figsize=(7, 6))
for name, p in probas.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Model Comparison (Test set)")
plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/roc_comparison.png", dpi=150); plt.close()

best_name = table.iloc[0]["Model"]
best_model = final_models[best_name]
TARGET_RECALL = 0.75

oof = cross_val_predict(best_model, X_train, y_train, cv=cv,
                        method="predict_proba", n_jobs=-1)[:, 1]
grid = np.linspace(0.05, 0.95, 181)
ok = [t for t in grid if recall_score(y_train, (oof >= t).astype(int)) >= TARGET_RECALL]
threshold = max(ok) if ok else 0.5   # highest threshold that still gives 75% recall

p_test = probas[best_name]
pred = (p_test >= threshold).astype(int)
print(f"=== THRESHOLD for {best_name} ===")
print(f"Chosen threshold : {threshold:.2f}  (target recall >= {TARGET_RECALL:.0%})")
print(f"Test recall      : {recall_score(y_test, pred):.3f}")
print(f"Test precision   : {precision_score(y_test, pred):.3f}")
print(f"Test F1          : {f1_score(y_test, pred):.3f}")
print(f"Customers flagged: {pred.mean():.1%} of test base\n")

rec = [recall_score(y_test, (p_test >= t).astype(int)) for t in grid]
prec = [precision_score(y_test, (p_test >= t).astype(int), zero_division=0) for t in grid]
plt.figure(figsize=(7, 5))
plt.plot(grid, rec, label="Recall"); plt.plot(grid, prec, label="Precision")
plt.axvline(threshold, color="r", ls="--", label=f"Chosen thr = {threshold:.2f}")
plt.axhline(TARGET_RECALL, color="gray", ls=":", label="75% recall target")
plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Threshold choice")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/threshold_choice.png", dpi=150); plt.close()

prep = best_model.named_steps["prep"]
est = best_model.named_steps["model"]
names = prep.get_feature_names_out()
names = [n.replace("num__", "").replace("cat__", "") for n in names]

if hasattr(est, "feature_importances_"):
    imp = est.feature_importances_
    xlabel = "Importance"
else:                                  
    imp = np.abs(est.coef_[0])
    xlabel = "|Coefficient| (scaled features)"

imp_s = pd.Series(imp, index=names).sort_values(ascending=False)
top5 = imp_s.head(5)
print("=== TOP 5 FEATURES ===")
print(top5.round(4).to_string(), "\n")

top15 = imp_s.head(15)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top15.index, top15.values, color="#2a7fba")
plt.xlabel(xlabel); plt.title(f"Top 15 Feature Importances - {best_name}")
plt.tight_layout(); plt.savefig(f"{OUT}/feature_importance.png", dpi=150); plt.close()

print(f"Saved files in ./{OUT}/ : comparison_table.csv, roc_comparison.png, "
      "feature_importance.png, threshold_choice.png")