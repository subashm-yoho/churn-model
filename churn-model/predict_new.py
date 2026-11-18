import sys
import pandas as pd
import joblib

input_csv = sys.argv[1] if len(sys.argv) > 1 else "new_customers.csv"
model_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/churn_model.joblib"
output_csv = sys.argv[3] if len(sys.argv) > 3 else "predictions.csv"

bundle = joblib.load(model_path)
pipeline = bundle["pipeline"]
threshold = bundle["threshold"]
feature_cols = bundle["feature_columns"]

new = pd.read_csv(input_csv)
 
if "TotalCharges" in new.columns:
 new["TotalCharges"] = pd.to_numeric(new["TotalCharges"], errors="coerce")

missing = [c for c in feature_cols if c not in new.columns]
if missing:
  sys.exit(f"ERROR: input CSV is missing columns: {missing}")

X_new = new[feature_cols]
proba = pipeline.predict_proba(X_new)[:, 1]

out = pd.DataFrame()
if "customerID" in new.columns:
 out["customerID"] = new["customerID"]
out["churn_probability"] = proba.round(4)
out["churn_prediction"] = (proba >= threshold).astype(int)
out["risk_level"] = pd.cut(proba, bins=[-0.01, 0.3, threshold, 1.0],
labels=["Low", "Medium", "High"])

out.to_csv(output_csv, index=False)
print(f"Threshold used: {threshold:.2f}")
print(f"Predicted {int(out['churn_prediction'].sum())} churners out of {len(out)} customers")
print(f"Saved: {output_csv}")
print(out.head())
