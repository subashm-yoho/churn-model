import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

from common import RS, load_data, make_pipeline, split_data

X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y) # test set stays aside
print("Train", X_train.shape, "| Val", X_val.shape)

rf_pipe = make_pipeline(RandomForestClassifier(random_state=RS, n_jobs=1), X)
rf_params = {"model__n_estimators": [100, 300], "model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5]}
rf_search = GridSearchCV(rf_pipe, rf_params, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1).fit(X_train, y_train)
print("\nRandomForest GridSearchCV (12 combinations x 3 folds)")
print("Best parameters:", rf_search.best_params_)
print(f"Best CV ROC-AUC: {rf_search.best_score_:.4f}")

xgb_pipe = make_pipeline(XGBClassifier(eval_metric="logloss", random_state=RS), X)
xgb_params = {"model__n_estimators": [100, 200, 300], "model__max_depth": [2, 3, 4, 5], "model__learning_rate": [0.03, 0.05, 0.1, 0.2], "model__subsample": [0.7, 0.85, 1.0], "model__colsample_bytree": [0.7, 0.85, 1.0]}
xgb_search = RandomizedSearchCV(xgb_pipe, xgb_params, n_iter=15, cv=3, scoring="roc_auc", n_jobs=-1, random_state=RS, verbose=1).fit(X_train, y_train)
print("\nXGBoost RandomizedSearchCV (n_iter=15, 3 folds)")
print("Best parameters:", xgb_search.best_params_)
print(f"Best CV ROC-AUC: {xgb_search.best_score_:.4f}")

tuned = {"RandomForest (tuned)": rf_search.best_estimator_, "XGBoost (tuned)": xgb_search.best_estimator_}
rows = pd.DataFrame([{"Model": n, "CV ROC-AUC": round(s.best_score_, 4), "Val ROC-AUC": round(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]), 4), "Val accuracy": round(accuracy_score(y_val, m.predict(X_val)), 4), "Val recall": round(recall_score(y_val, m.predict(X_val)), 4), "Val F1": round(f1_score(y_val, m.predict(X_val)), 4)} for (n, m), s in zip(tuned.items(), [rf_search, xgb_search])])
print("\nTuned models on validation set")
print(rows.to_string(index=False))
print("(compare with the untuned models in out4.txt)")

rows.to_csv("task4_tuning_results.csv", index=False)
joblib.dump(rf_search.best_estimator_, "best_rf.joblib")
joblib.dump(xgb_search.best_estimator_, "best_xgb.joblib")
print("\nSaved best_rf.joblib and best_xgb.joblib")
