import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import RS, load_data, split_data

X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y) # test set stays aside

cols = ["tenure", "MonthlyCharges", "TotalCharges"]
data = X_train[cols]

imp = SimpleImputer(strategy="median").fit(data)
filled = pd.DataFrame(imp.transform(data), columns=cols, index=data.index)
scaler = StandardScaler().fit(filled)
Z = scaler.transform(filled)

ks = list(range(2, 9))

models = [
KMeans(n_clusters=k, n_init=10, random_state=RS).fit(Z)
for k in ks
]

inertia = [m.inertia_ for m in models]

sil = [
silhouette_score(Z, m.labels_)
for m in models
]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertia, marker="o")
ax[0].set_title("Elbow curve (inertia)")
ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia"); ax[0].grid(alpha=0.3)
ax[1].plot(ks, sil, marker="o", color="tab:orange")
ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette"); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("kmeans_elbow_silhouette.png", dpi=120)
plt.show()

# Select best K
best_k = ks[sil.index(max(sil))]
print("Best K:", best_k)

final = KMeans(
n_clusters=best_k,
n_init=10,
random_state=RS
).fit(Z)

seg = filled.copy()
seg["segment"] = final.labels_
seg["churn"] = y_train.loc[seg.index].values
overall = filled.mean()
summary = seg.groupby("segment").agg(
customers=("tenure", "size"),
tenure=("tenure", "mean"),
monthly=("MonthlyCharges", "mean"),
total=("TotalCharges", "mean"),
churn_rate=("churn", "mean"),
).round(2)
print(f"\nSegments for k={best_k}")
print(summary)

print("\nSegment description:")
lines = [
f" Segment {s}: {int(r['customers'])} customers, "
f"{'long' if r['tenure'] > overall['tenure'] else 'short'} tenure ({r['tenure']:.0f} months), "
f"{'high' if r['monthly'] > overall['MonthlyCharges'] else 'low'} monthly charges ({r['monthly']:.0f}), "
f"churn rate {r['churn_rate'] * 100:.0f}%"
for s, r in summary.iterrows()
]
print("\n".join(lines))

summary.to_csv("task3_kmeans_segments.csv")