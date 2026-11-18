import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import RS, load_data, make_pipeline, split_data

X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)   # test set stays aside
print("Train", X_train.shape, "| Val", X_val.shape)

results = []


def report(name, pipe):
    pred = pipe.predict(X_val)
    proba = pipe.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, pred)
    auc = roc_auc_score(y_val, proba)
    results.append({"Model": name, "Val accuracy": round(acc, 4), "Val ROC-AUC": round(auc, 4)})
    print(f"{name:<20} val accuracy {acc:.4f} | val ROC-AUC {auc:.4f}")


def top5(names, importances):
    s = pd.Series(importances, index=[n.split("__", 1)[-1] for n in names])
    return s.sort_values(ascending=False).head(5).round(4)


# ---------- 1. Logistic Regression ----------
log_pipe = make_pipeline(LogisticRegression(max_iter=1000), X).fit(X_train, y_train)
report("LogisticRegression", log_pipe)

# ---------- 2. Decision Tree ----------
tree_pipe = make_pipeline(DecisionTreeClassifier(max_depth=5, random_state=RS), X).fit(X_train, y_train)
report("DecisionTree", tree_pipe)

# ---------- 3. Random Forest (OOB score) ----------
rf_pipe = make_pipeline(
    RandomForestClassifier(n_estimators=300, oob_score=True, random_state=RS, n_jobs=-1), X
).fit(X_train, y_train)
report("RandomForest", rf_pipe)
print(f"   RandomForest OOB score: {rf_pipe.named_steps['model'].oob_score_:.4f}")

# ---------- 4. XGBoost with early stopping on the validation set ----------
xgb_pipe = make_pipeline(
    XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=3,
                  early_stopping_rounds=20, eval_metric="logloss", random_state=RS), X)
pre = xgb_pipe[:-1]                      # features + preprocess (fit on train only)
Xt = pre.fit_transform(X_train, y_train)
Xv = pre.transform(X_val)
xgb_pipe.named_steps["model"].fit(Xt, y_train, eval_set=[(Xv, y_val)], verbose=False)
report("XGBoost", xgb_pipe)
print(f"   XGBoost stopped at best iteration: {xgb_pipe.named_steps['model'].best_iteration}")

# ---------- predict_proba vs predict (5 rows, logistic regression) ----------
rows = X_val.head(5)
proba = log_pipe.predict_proba(rows)
pred = log_pipe.predict(rows)
cmp = pd.DataFrame({
    "P(stay=0)": proba[:, 0].round(3),
    "P(churn=1)": proba[:, 1].round(3),
    "predict": pred,
    "actual": y_val.head(5).values,
})
print("\nLogistic regression: predict_proba vs predict (5 validation rows)")
print(cmp)
print("predict = 1 only when P(churn) > 0.5, so predict is just a threshold on predict_proba.")

# ---------- Top 5 important features ----------
names = rf_pipe.named_steps["preprocess"].get_feature_names_out()
print("\nTop 5 features - RandomForest:")
print(top5(names, rf_pipe.named_steps["model"].feature_importances_))
print("\nTop 5 features - XGBoost:")
print(top5(names, xgb_pipe.named_steps["model"].feature_importances_))

# ---------- Results table ----------
table = pd.DataFrame(results)
print("\nResults table (Model | Validation score)")
print(table.to_string(index=False))
table.to_csv("task3_classifier_results.csv", index=False)