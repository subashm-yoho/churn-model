import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import RS, load_data, make_pipeline, split_data

X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y) # test set stays aside
print("Train", X_train.shape, "| Val", X_val.shape)

log_pipe = make_pipeline(LogisticRegression(max_iter=1000), X).fit(X_train, y_train)
tree_pipe = make_pipeline(DecisionTreeClassifier(max_depth=5, random_state=RS), X).fit(X_train, y_train)
rf_pipe = make_pipeline(RandomForestClassifier(n_estimators=300, oob_score=True, random_state=RS, n_jobs=-1), X).fit(X_train, y_train)
xgb_pipe = make_pipeline(XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, early_stopping_rounds=20, eval_metric="logloss", random_state=RS), X)
Xt = xgb_pipe[:-1].fit_transform(X_train, y_train)
Xv = xgb_pipe[:-1].transform(X_val)
xgb_pipe.named_steps["model"].fit(Xt, y_train, eval_set=[(Xv, y_val)], verbose=False)

pipes = {"LogisticRegression": log_pipe, "DecisionTree": tree_pipe, "RandomForest": rf_pipe, "XGBoost": xgb_pipe}
preds = {n: p.predict(X_val) for n, p in pipes.items()}
probas = {n: p.predict_proba(X_val)[:, 1] for n, p in pipes.items()}

table = pd.DataFrame([{"Model": n, "Accuracy": accuracy_score(y_val, p), "Precision": precision_score(y_val, p), "Recall": recall_score(y_val, p), "F1": f1_score(y_val, p), "ROC-AUC": roc_auc_score(y_val, probas[n])} for n, p in preds.items()]).round(4)
print("\nValidation metrics (threshold 0.5)")
print(table.to_string(index=False))

print("\nConfusion matrices [[TN FP] [FN TP]]")
print("\n".join([f"{n}\n{confusion_matrix(y_val, p)}\n" for n, p in preds.items()]))

stay_share = 1 - y_val.mean()
print(f"Class balance in validation: {stay_share:.1%} stay, {1 - stay_share:.1%} churn")
print(f"A dummy model that ALWAYS predicts 'stay' gets accuracy {stay_share:.1%} but recall 0%.")
print("So accuracy looks good even when the model finds no churners; use recall, precision, F1 and ROC-AUC.")

plt.figure(figsize=(6, 5))
roc_models = ["LogisticRegression", "RandomForest", "XGBoost"]
curves = [plt.plot(*roc_curve(y_val, probas[n])[:2], label=f"{n} (AUC = {roc_auc_score(y_val, probas[n]):.3f})") for n in roc_models]
plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves (validation set)"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=120)
print("\nSaved roc_curves.png (close the plot window to continue)")
plt.show()

best_name = max(probas, key=lambda n: roc_auc_score(y_val, probas[n]))
prec, rec, thr = precision_recall_curve(y_val, probas[best_name])
ok = np.where(rec[:-1] >= 0.75)[0]
best_i = ok[-1] # highest threshold that still keeps recall >= 75%
chosen = float(thr[best_i])
final_pred = (probas[best_name] >= chosen).astype(int)
print(f"\nBest model by ROC-AUC: {best_name}")
print(f"Chosen threshold: {chosen:.3f} (default was 0.5)")
print(f"At this threshold: recall {recall_score(y_val, final_pred):.3f} | precision {precision_score(y_val, final_pred):.3f} | F1 {f1_score(y_val, final_pred):.3f}")
print("Confusion matrix at this threshold [[TN FP] [FN TP]]:")
print(confusion_matrix(y_val, final_pred))
print("Lower threshold = catch more churners (higher recall) but more false alarms (lower precision).")

open("threshold.txt", "w").write(f"{best_name},{chosen}")
table.to_csv("task4_metrics.csv", index=False)