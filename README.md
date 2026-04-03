# 🏥 biomedical-ml-toolkit

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)](notebooks/)
[![Datasets](https://img.shields.io/badge/Datasets-5%20Medical-red)](datasets/)

> **Ready-to-use Machine Learning toolkit for Biomedical Engineering students and researchers in India.**
> Includes curated medical datasets, beginner-friendly Python scripts, and Jupyter notebooks for 4 core ML algorithms — with real healthcare examples.

---

## 📌 Why This Project?

Most ML tutorials use generic datasets (iris, titanic). This toolkit is built **specifically for Biomedical Engineering students** with:

- ✅ Real medical datasets (Diabetes, Heart Disease, Breast Cancer, ECG)
- ✅ Complete Python scripts — just run and see results
- ✅ Notebooks with step-by-step explanations + output screenshots
- ✅ No prior coding experience needed
- ✅ Designed for Indian university curriculum (Anna University, VTU, GTU etc.)

Used by **Assistant Professors and students** for lab practicals, mini-projects, and research demonstrations.

---

## 📁 Project Structure

```
biomedical-ml-toolkit/
│
├── datasets/                    ← CSV medical datasets (ready to use)
│   ├── diabetes.csv             ← 768 patients, 8 features
│   ├── heart_disease.csv        ← 303 patients, 13 features
│   ├── breast_cancer.csv        ← 569 samples, 30 features
│   └── ecg_features.csv         ← 452 ECG signal features
│
├── notebooks/                   ← Jupyter notebooks (run in Google Colab)
│   ├── 01_linear_regression.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_decision_trees.ipynb
│   └── 04_kmeans_clustering.ipynb
│
├── scripts/                     ← Pure Python scripts (run in terminal)
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   └── kmeans_clustering.py
│
├── docs/                        ← Explanation and diagrams
│   └── algorithm_guide.md
│
├── requirements.txt             ← Python packages needed
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start (No Installation — Use Google Colab)

**Option 1: Run in browser (recommended for beginners)**

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open Notebook → GitHub**
3. Paste: `https://github.com/YOUR_USERNAME/biomedical-ml-toolkit`
4. Open any notebook and click **Run All**

**Option 2: Run locally**

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/biomedical-ml-toolkit.git
cd biomedical-ml-toolkit

# 2. Install required packages
pip install -r requirements.txt

# 3. Run any script
python scripts/logistic_regression.py
```

---

## 📊 Datasets Included

| Dataset | Samples | Features | Task | Source |
|---------|---------|----------|------|--------|
| `diabetes.csv` | 768 | 8 | Binary Classification (Diabetic/Not) | PIMA Indian Diabetes |
| `heart_disease.csv` | 303 | 13 | Binary Classification (Disease/None) | UCI Heart Disease |
| `breast_cancer.csv` | 569 | 30 | Binary Classification (Malignant/Benign) | Wisconsin BC Dataset |
| `ecg_features.csv` | 452 | 12 | Multi-class (Normal/Arrhythmia/Other) | MIT-BIH derived |

All datasets are **pre-cleaned**, **normalized**, and **ready to use** — no preprocessing needed.

---

## 🧠 Algorithms Covered

| Algorithm | Script | Notebook | Medical Application |
|-----------|--------|----------|---------------------|
| Linear Regression | `scripts/linear_regression.py` | `01_linear_regression.ipynb` | Predict blood pressure from BMI |
| Logistic Regression | `scripts/logistic_regression.py` | `02_logistic_regression.ipynb` | Diabetes YES/NO prediction |
| Decision Tree | `scripts/decision_tree.py` | `03_decision_trees.ipynb` | Heart disease diagnosis |
| K-Means Clustering | `scripts/kmeans_clustering.py` | `04_kmeans_clustering.ipynb` | Patient risk group segmentation |

---

## 📈 Sample Results

### Logistic Regression — Diabetes Prediction
```
Accuracy:  78.5%
Precision: 72.3%
Recall:    68.9%
F1-Score:  70.6%

Confusion Matrix:
            Predicted NO  Predicted YES
Actual NO       118           15
Actual YES       24           35
```

### K-Means — Patient Segmentation (K=3)
```
Cluster 0 (Low Risk):    198 patients  — Low glucose, Low BMI
Cluster 1 (Medium Risk): 154 patients  — Medium glucose, Medium BMI
Cluster 2 (High Risk):   109 patients  — High glucose, High BMI
Inertia: 4521.3
```

---

## 🎓 Who Is This For?

- 🏫 **B.E./B.Tech Biomedical Engineering students** doing lab practicals
- 👨‍🏫 **Assistant Professors** preparing AI/ML lab sessions
- 🔬 **M.E./M.Tech students** doing literature review projects
- 🏥 **Healthcare researchers** learning ML from scratch

---

## 📚 How to Use in Your College Lab

1. Ask students to open Google Colab (free, no installation)
2. Share the GitHub link
3. Each student opens a notebook and runs it cell by cell
4. Discuss the output — algorithm, results, accuracy

**Perfect for AI/ML lab practicals under Anna University, VTU, GTU syllabi.**

---

## 🤝 Contributing

Pull requests welcome! If you have:
- A new medical dataset
- A better algorithm explanation
- Translation to Tamil/Hindi/Kannada

Please open an issue or submit a PR.

---

## 👤 Author

**C. Sathish Kumar**
Assistant Professor, Biomedical Engineering
IEEE Published Researcher | QIP Certified (AI & Robotics)


---

## 📄 License

MIT License — free to use for education and research.

---

*Built for Indian Biomedical Engineering students. If this helped you, please ⭐ star the repo!*
