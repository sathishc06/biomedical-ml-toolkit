"""
=============================================================
  Linear Regression — Blood Pressure Prediction
  Biomedical ML Toolkit | C. Sathish Kumar
=============================================================
HOW TO RUN:  python scripts/linear_regression.py
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

print("=" * 55)
print("  LINEAR REGRESSION — BLOOD PRESSURE PREDICTION")
print("=" * 55)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, '..', 'datasets', 'diabetes.csv')

df = pd.read_csv(data_path)

# Use linear regression to predict Blood Pressure from other features
target_col = 'BloodPressure'
feature_cols = ['Glucose', 'BMI', 'Age', 'Insulin', 'Pregnancies']

# Clean zeros
for col in [target_col] + feature_cols:
    df[col] = df[col].replace(0, df[col].mean())

X = df[feature_cols]
y = df[target_col]

print(f"\n📊 Predicting: {target_col} from {feature_cols}")
print(f"  Samples: {len(df)}")
print(f"  Mean BP: {y.mean():.1f} mmHg  |  Range: {y.min():.0f}-{y.max():.0f} mmHg")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 55)
print("  RESULTS")
print("=" * 55)
print(f"\n  MSE  (Mean Squared Error)       : {mse:.2f}")
print(f"  RMSE (Root Mean Squared Error)  : {rmse:.2f} mmHg")
print(f"  MAE  (Mean Absolute Error)      : {mae:.2f} mmHg")
print(f"  R²   (Coefficient of Determination): {r2:.4f}")
print(f"\n  Interpretation: Model explains {r2*100:.1f}% of variance in Blood Pressure")

print("\n📈 SAMPLE PREDICTIONS (first 10 test patients):")
print(f"  {'Actual BP':12s} {'Predicted BP':14s} {'Error':8s}")
print("  " + "-" * 38)
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    err = abs(actual - predicted)
    print(f"  {actual:10.1f}   {predicted:12.1f}   {err:6.1f}")

print("\n📊 REGRESSION COEFFICIENTS (feature importance):")
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
for _, r in coef_df.iterrows():
    bar = "█" * int(abs(r['Coefficient']) * 4)
    print(f"  {r['Feature']:20s}  {r['Coefficient']:+.4f}  {bar}")
print(f"\n  Intercept (b): {model.intercept_:.4f}")

new_p = pd.DataFrame([[140, 35.0, 55, 100, 3]], columns=feature_cols)
pred_bp = model.predict(scaler.transform(new_p))[0]
print(f"\n🔮 NEW PATIENT PREDICTION:")
print(f"  Glucose=140, BMI=35, Age=55 → Predicted BP: {pred_bp:.1f} mmHg")
print("\n✅ Done!")
