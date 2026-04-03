"""
=============================================================
  Decision Tree — Heart Disease Diagnosis
  Biomedical ML Toolkit | C. Sathish Kumar
=============================================================
HOW TO RUN:  python scripts/decision_tree.py
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

print("=" * 55)
print("  DECISION TREE — HEART DISEASE DIAGNOSIS")
print("=" * 55)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, '..', 'datasets', 'heart_disease.csv')

df = pd.read_csv(data_path)

print(f"\n📊 Dataset: {len(df)} patients, {df.shape[1]-1} features")
print(f"  Heart Disease (1): {df['target'].sum()} patients")
print(f"  Healthy       (0): {(df['target']==0).sum()} patients")

# Features and target
feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']
X = df[feature_names]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📂 Train: {len(X_train)} | Test: {len(X_test)}")

# Train Decision Tree
print("\n🌳 TRAINING DECISION TREE (max_depth=4)...")
dt = DecisionTreeClassifier(max_depth=4, min_samples_split=10,
                             criterion='gini', random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
acc    = accuracy_score(y_test, y_pred) * 100

print("\n" + "=" * 55)
print("  RESULTS")
print("=" * 55)
print(f"\n📈 ACCURACY: {acc:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("\n📊 CONFUSION MATRIX:")
print("                  Pred: Healthy  Pred: Disease")
print(f"  Actual: Healthy      {cm[0][0]:5d}        {cm[0][1]:5d}")
print(f"  Actual: Disease      {cm[1][0]:5d}        {cm[1][1]:5d}")

print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Healthy','Heart Disease']))

# Feature importance
print("=" * 55)
print("  FEATURE IMPORTANCE (Top 5)")
print("=" * 55)
importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False).head(5)

for _, row in importances.iterrows():
    bar = "█" * int(row['Importance'] * 40)
    print(f"  {row['Feature']:15s}  {row['Importance']:.4f}  {bar}")

# Print tree rules
print("\n" + "=" * 55)
print("  DECISION TREE RULES (first 3 levels)")
print("=" * 55)
tree_rules = export_text(dt, feature_names=feature_names, max_depth=2)
print(tree_rules)

# Predict new patient
print("=" * 55)
print("  PREDICT NEW PATIENT")
print("=" * 55)
new_p = pd.DataFrame([[62, 1, 0, 120, 267, 0, 1, 99, 1, 1.8, 1, 2, 3]],
                     columns=feature_names)
pred  = dt.predict(new_p)[0]
prob  = dt.predict_proba(new_p)[0]

print(f"\n  Age=62, Sex=Male, Chest Pain Type=0, Cholesterol=267")
print(f"\n  Prediction: {'❤️ HEART DISEASE' if pred==1 else '✅ HEALTHY'}")
print(f"  Probability: Healthy={prob[0]*100:.1f}%  Disease={prob[1]*100:.1f}%")
print("\n✅ Done!")
