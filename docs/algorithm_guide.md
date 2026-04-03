# ML Algorithm Guide — Biomedical ML Toolkit

## Overview

This guide explains the 4 machine learning algorithms implemented in this toolkit, with focus on medical applications.

---

## 1. Linear Regression

**Use when:** You want to predict a continuous number (blood pressure, glucose level, recovery days)

**Equation:** `y = mx + b`

- `y` = predicted output (e.g., Blood Pressure in mmHg)
- `m` = slope (how much y changes per unit of x)
- `x` = input feature (e.g., BMI)
- `b` = intercept (baseline value)

**Medical Example:** Predict Blood Pressure from BMI
```
BP = 1.8 × BMI + 65.2
If BMI = 30 → BP = 1.8×30 + 65.2 = 119.2 mmHg
```

**Evaluation Metrics:**
- MSE (Mean Squared Error) — lower is better
- RMSE (Root MSE) — same unit as output
- R² (R-squared) — range 0-1, higher is better

---

## 2. Logistic Regression

**Use when:** You want to predict YES/NO (diabetic/not, disease/healthy)

**Sigmoid Function:** `P = 1 / (1 + e^-z)`  where `z = b0 + b1*x1 + b2*x2`

- Output is always between 0 and 1 (probability)
- If P > 0.5 → Class 1 (YES / Disease)
- If P ≤ 0.5 → Class 0 (NO / Healthy)

**Medical Example:** Predict Diabetes from Glucose, BMI, Age
```
Input: Glucose=148, BMI=33.6, Age=50
Output: P = 0.78 → Diabetic (78% probability)
```

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (TP, FP, TN, FN)

---

## 3. Decision Tree

**Use when:** You need an interpretable, rule-based model that doctors can understand

**How it works:** Asks YES/NO questions about features, splitting data at each node.

**Key Terms:**
- Root Node: First question asked
- Internal Node: Tests a feature condition
- Leaf Node: Final prediction
- Depth: Number of levels in the tree

**Split Criteria:**
- Gini Impurity: `Gini = 1 - Σ(pᵢ)²` (lower = purer)
- Information Gain: Entropy reduction after split

**Medical Example:** Heart Disease Diagnosis Tree
```
Blood Sugar > 126?
├── YES → BMI > 30? → YES → Type 2 Diabetes
│                  → NO  → Pre-Diabetic
└── NO  → Age > 45? → YES → Risk Patient
                    → NO  → Healthy
```

---

## 4. K-Means Clustering

**Use when:** You want to group patients into risk categories without predefined labels

**Algorithm Steps:**
1. Choose K (number of groups)
2. Place K centroids randomly
3. Assign each patient to nearest centroid
4. Move centroid to mean of assigned patients
5. Repeat steps 3-4 until convergence

**Distance Formula:** `d = √[(x₂-x₁)² + (y₂-y₁)²]`

**Medical Example:** Patient Risk Segmentation (K=3)
```
Cluster 1: Low Risk    — Glucose<110, BMI<25
Cluster 2: Medium Risk — Glucose 110-130, BMI 25-30
Cluster 3: High Risk   — Glucose>130, BMI>30
```

**Choose K:** Use Elbow Method — plot WCSS vs K, pick the "elbow" point.

---

## Quick Reference Table

| Algorithm | Output Type | Medical Use | Key Metric |
|-----------|-------------|-------------|------------|
| Linear Regression | Continuous number | Predict BP, dosage | R², RMSE |
| Logistic Regression | Probability / Class | Diagnose disease | Accuracy, F1 |
| Decision Tree | Class + Rules | Explainable diagnosis | Accuracy, Gini |
| K-Means | Cluster Group | Risk segmentation | Silhouette Score |

---

## How to Run

```bash
# Install requirements
pip install -r requirements.txt

# Run any script
python scripts/logistic_regression.py
python scripts/decision_tree.py
python scripts/kmeans_clustering.py
python scripts/linear_regression.py
```

---

*Author: C. Sathish Kumar, Assistant Professor, Biomedical Engineering*  
*IEEE Researcher | QIP Certified (NIT Puducherry)*
