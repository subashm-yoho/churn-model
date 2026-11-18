import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"
RS = 42

df = pd.read_csv(CSV_PATH)
df = df.drop(columns=["customerID"], errors="ignore")
if "TotalCharges" in df.columns:
 df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
y = (df[TARGET] == "Yes").astype(int)
X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True).astype(float)

X_temp, X_test, y_temp, y_test = train_test_split(
X, y, test_size=0.20, stratify=y, random_state=RS)
X_train, X_val, y_train, y_val = train_test_split(
X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RS) # 0.25 of 80% = 20%
print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape} (test set aside, not used)")

depths = [1, 3, 5, 10, None]
train_acc, val_acc = [], []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=RS).fit(X_train, y_train)
    train_acc.append(tree.score(X_train, y_train))
    val_acc.append(tree.score(X_val, y_val))
    print(f"Depth {str(d):>4}: train {train_acc[-1]:.2f} val {val_acc[-1]:.2f}")

labels = [str(d) for d in depths]
plt.figure(figsize=(7, 4))
plt.plot(labels, train_acc, marker="o", label="Train")
plt.plot(labels, val_acc, marker="o", label="Validation")
plt.xlabel("max_depth"); plt.ylabel("Accuracy")
plt.title("Overfitting: train vs validation accuracy")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("overfitting_plot.png", dpi=120)
plt.show()

# Scaler is inside the pipeline so it is fit only on each training fold (no leakage)
log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_val_score(log_reg, X_train, y_train, cv=5, scoring="accuracy")
print(f"\nCV: {scores.mean():.2f} +/- {scores.std():.2f}")

X_scaled_all = StandardScaler().fit_transform(X)
Xtr_w, Xte_w, ytr_w, yte_w = train_test_split(
X_scaled_all, y, test_size=0.20, stratify=y, random_state=RS)
wrong = LogisticRegression(max_iter=1000).fit(Xtr_w, ytr_w)
print("\n[WRONG - scale before split] test acc:", round(wrong.score(Xte_w, yte_w), 4))
print(" test-set mean of scaled data (leaked stats):", np.round(Xte_w.mean(), 4))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RS)
scaler = StandardScaler().fit(Xtr) # fit ONLY on train
right = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
print("[RIGHT - fit on train only] test acc:", round(right.score(scaler.transform(Xte), yte), 4))
print(" test-set mean of scaled data (honest):", np.round(scaler.transform(Xte).mean(), 4))

print("""
LEAKAGE EXPLANATION (4 lines):
1. Scaling before the split lets the mean/std of the test rows influence the training data.
2. So the model indirectly 'sees' test information, and test accuracy becomes optimistic.
3. The correct way: split first, fit the scaler on train only, then only transform val/test.
4. In real deployment, new data is never available at fit time, so only the correct way is honest.
""")

problems = {
"Spam detection": "Supervised (classification - labelled spam/not spam)",
"Customer segmentation": "Unsupervised (clustering - no labels)",
"House price prediction": "Supervised (regression - target is price)",
"Anomaly detection": "Unsupervised (usually - finds outliers without labels)",
}
print("PROBLEM CLASSIFICATION")
for k, v in problems.items():
 print(f" {k:<24} -> {v}")

print("""
ML WORKFLOW (7 steps)
1. Define the problem & success metric (e.g. recall / ROC-AUC for default).
2. Collect & understand data (EDA, missing values, class balance).
3. Split data (train/val/test) BEFORE any fitting, stratified.
4. Preprocess & feature engineering (impute, encode, scale - fit on train only).
5. Train baseline and candidate models; tune with cross-validation.
6. Evaluate on validation, then once on test; check confusion matrix & threshold.
7. Deploy, monitor drift, and retrain when performance drops.
""")
