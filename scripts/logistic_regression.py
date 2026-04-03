"""
=============================================================
  Logistic Regression — Diabetes Prediction
  Biomedical ML Toolkit | C. Sathish Kumar
  Assistant Professor, Biomedical Engineering
=============================================================

WHAT THIS SCRIPT DOES:
  Predicts whether a patient is diabetic (1) or not (0)
  using Logistic Regression on the PIMA Diabetes dataset.

HOW TO RUN:
  python scripts/logistic_regression.py

EXPECTED OUTPUT:
  Model accuracy, classification report, and confusion matrix
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import os

# ──────────────────────────────────────────────
#  STEP 1: Load the Dataset
# ──────────────────────────────────────────────
print("=" * 55)
print("  LOGISTIC REGRESSION — DIABETES PREDICTION")
print("=" * 55)

# Build path to dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'datasets', 'diabetes.csv')

df = pd.read_csv(data_path)

print("\n📊 DATASET OVERVIEW")
print(f"  Total patients  : {len(df)}")
print(f"  Features        : {df.shape[1] - 1}")
print(f"  Diabetic (1)    : {df['Outcome'].sum()}")
print(f"  Non-diabetic (0): {(df['Outcome'] == 0).sum()}")
print(f"\n  Columns: {list(df.columns)}")

# ──────────────────────────────────────────────
#  STEP 2: Prepare Features and Target
# ──────────────────────────────────────────────
print("\n🔧 PREPARING DATA...")

# Features (X) and Target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Replace zero values in medical columns with column mean
# (0 is medically impossible for Glucose, BMI, Blood Pressure)
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    X[col] = X[col].replace(0, X[col].mean())

print(f"  Zero values replaced with column mean in: {cols_to_fix}")

# ──────────────────────────────────────────────
#  STEP 3: Split into Training and Testing sets
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📂 DATA SPLIT")
print(f"  Training samples : {len(X_train)} (80%)")
print(f"  Testing  samples : {len(X_test)}  (20%)")

# ──────────────────────────────────────────────
#  STEP 4: Feature Scaling (Normalization)
# ──────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n✅ Features scaled using StandardScaler")

# ──────────────────────────────────────────────
#  STEP 5: Train the Logistic Regression Model
# ──────────────────────────────────────────────
print("\n🤖 TRAINING LOGISTIC REGRESSION MODEL...")

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("  Model trained successfully!")

# ──────────────────────────────────────────────
#  STEP 6: Evaluate the Model
# ──────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred) * 100
cm        = confusion_matrix(y_test, y_pred)
report    = classification_report(y_test, y_pred,
                                   target_names=['Non-Diabetic', 'Diabetic'])

print("\n" + "=" * 55)
print("  RESULTS")
print("=" * 55)

print(f"\n📈 ACCURACY: {accuracy:.2f}%")

print("\n📊 CONFUSION MATRIX:")
print("                  Predicted: NO   Predicted: YES")
print(f"  Actual: NO           {cm[0][0]:4d}           {cm[0][1]:4d}")
print(f"  Actual: YES          {cm[1][0]:4d}           {cm[1][1]:4d}")

print("\n📋 CLASSIFICATION REPORT:")
print(report)

# ──────────────────────────────────────────────
#  STEP 7: Predict on a New Patient
# ──────────────────────────────────────────────
print("=" * 55)
print("  PREDICT A NEW PATIENT")
print("=" * 55)

# New patient data: [Pregnancies, Glucose, BP, SkinThick, Insulin, BMI, Pedigree, Age]
new_patient = pd.DataFrame([[2, 148, 72, 35, 94, 33.6, 0.627, 45]],
                            columns=X.columns)
new_patient_scaled = scaler.transform(new_patient)

prediction   = model.predict(new_patient_scaled)[0]
probability  = model.predict_proba(new_patient_scaled)[0][1]

print(f"\n  Patient Data:")
print(f"    Glucose   : 148 mg/dL")
print(f"    BMI       : 33.6")
print(f"    Age       : 45 years")
print(f"    BP        : 72 mmHg")

print(f"\n  Prediction  : {'DIABETIC ⚠️' if prediction == 1 else 'NOT DIABETIC ✅'}")
print(f"  Probability : {probability * 100:.1f}% chance of diabetes")

print("\n" + "=" * 55)
print("  FEATURE IMPORTANCE (Coefficients)")
print("=" * 55)
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

for _, row in coef_df.iterrows():
    bar = "█" * int(abs(row['Coefficient']) * 5)
    direction = "+" if row['Coefficient'] > 0 else "-"
    print(f"  {row['Feature']:30s} {direction}{abs(row['Coefficient']):.3f}  {bar}")

print("\n✅ Done! Higher positive coefficient = stronger predictor of diabetes.")
