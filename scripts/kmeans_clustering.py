"""
=============================================================
  K-Means Clustering — Patient Risk Segmentation
  Biomedical ML Toolkit | C. Sathish Kumar
=============================================================
HOW TO RUN:  python scripts/kmeans_clustering.py
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

print("=" * 55)
print("  K-MEANS CLUSTERING — PATIENT RISK SEGMENTATION")
print("=" * 55)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, '..', 'datasets', 'diabetes.csv')

df = pd.read_csv(data_path)

# Use key clinical features for clustering
features = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin']
X = df[features].copy()

# Replace zeros with mean
for col in features:
    X[col] = X[col].replace(0, X[col].mean())

# Scale features
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n📊 Dataset: {len(X)} patients")
print(f"  Features used for clustering: {features}")

# ── Elbow Method to find best K ──
print("\n📈 ELBOW METHOD — Finding Best K")
print("  K    WCSS (Inertia)    Silhouette Score")
print("  " + "-" * 42)

best_k, best_sil = 3, -1
wcss_values = []

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss = km.inertia_
    sil  = silhouette_score(X_scaled, km.labels_)
    wcss_values.append(wcss)
    marker = " ← BEST" if sil > best_sil else ""
    print(f"  K={k}  {wcss:10.1f}         {sil:.4f}{marker}")
    if sil > best_sil:
        best_sil = sil
        best_k   = k

# ── Final clustering with K=3 (clinically meaningful) ──
K = 3
print(f"\n🎯 Running final K-Means with K={K} (3 risk groups)")

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster'] += 1   # Make clusters 1, 2, 3 instead of 0, 1, 2

print("\n" + "=" * 55)
print("  RESULTS")
print("=" * 55)

# Describe each cluster
cluster_stats = df.groupby('Cluster')[features + ['Outcome']].mean()
cluster_counts = df['Cluster'].value_counts().sort_index()

print(f"\n📊 CLUSTER SUMMARY (K={K}):")
print(f"  {'Cluster':8s} {'Patients':10s} {'Glucose':10s} {'BMI':8s} {'Age':6s} {'Diabetic%':12s} {'LABEL'}")
print("  " + "-" * 75)

labels = []
for cluster_id in sorted(df['Cluster'].unique()):
    grp     = df[df['Cluster'] == cluster_id]
    count   = len(grp)
    glucose = grp['Glucose'].mean()
    bmi     = grp['BMI'].mean()
    age     = grp['Age'].mean()
    diab_pct= grp['Outcome'].mean() * 100

    if glucose < 110 and bmi < 28:
        label = "🟢 LOW RISK"
    elif glucose < 135 and bmi < 33:
        label = "🟡 MEDIUM RISK"
    else:
        label = "🔴 HIGH RISK"

    labels.append(label)
    print(f"  {cluster_id:8d} {count:10d} {glucose:10.1f} {bmi:8.1f} {age:6.1f} {diab_pct:10.1f}%   {label}")

print(f"\n  Silhouette Score: {silhouette_score(X_scaled, kmeans.labels_):.4f}")
print(f"  Inertia (WCSS)  : {kmeans.inertia_:.2f}")

# ── Assign a new patient ──
print("\n" + "=" * 55)
print("  ASSIGN NEW PATIENT TO A RISK GROUP")
print("=" * 55)

new_patient = pd.DataFrame([[148, 33.6, 50, 72, 94]], columns=features)
new_scaled  = scaler.transform(new_patient)
cluster_id  = kmeans.predict(new_scaled)[0] + 1

print(f"\n  Patient: Glucose=148, BMI=33.6, Age=50, BP=72, Insulin=94")
print(f"\n  Assigned to Cluster {cluster_id}: {labels[cluster_id-1]}")

print("\n" + "=" * 55)
print("  CENTROID VALUES (Cluster Centers)")
print("=" * 55)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
center_df = pd.DataFrame(centers, columns=features)
center_df.index = [f"Cluster {i+1}" for i in range(K)]
print(center_df.round(1).to_string())

print("\n✅ Done! Each patient is now assigned to a risk group.")
print("   Use this for: targeted treatment, early screening, hospital resource planning.")
