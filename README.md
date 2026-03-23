# 🧠 Explainable Federated Learning with Differential Privacy for Type-2 Diabetes Readmission Prediction

> A research-grade, reproducible machine learning framework integrating **Federated Learning (FL)**, **Differential Privacy (DP)**, **Class Imbalance Handling**, and **Explainable AI (XAI)** for predicting 30-day hospital readmission in Type-2 Diabetes patients.

---

## 📄 Associated Publication

This repository implements the methodology described in the ACM-style research paper:

👉 :contentReference[oaicite:0]{index=0}

---

## 📌 Abstract

Hospital readmission within 30 days remains a critical challenge in healthcare systems, particularly for Type-2 Diabetes Mellitus (T2DM) patients. Traditional centralized machine learning approaches are constrained by **privacy risks, regulatory barriers, and data silos**.

This repository presents a **privacy-preserving federated learning framework** that:
- Enables collaborative training across multiple institutions **without data sharing**
- Incorporates **Differential Privacy (ε = 1.0, δ = 10⁻⁵)** to provide formal privacy guarantees
- Handles **class imbalance (~9:1)** using SMOTE and SMOTE-ENN
- Applies **Explainable AI (SHAP)** for interpretability

The framework achieves strong predictive performance while maintaining privacy and clinical interpretability.

---

## 🎯 Key Contributions

- 🔐 **Federated Learning Framework** simulating multi-hospital collaboration (5 clients)
- 🛡️ **Differential Privacy Integration** (gradient clipping + Gaussian noise)
- ⚖️ **Imbalance Handling** via SMOTE and SMOTE-ENN
- 🌳 **Ensemble Models**: Random Forest, XGBoost, LightGBM
- 📊 **Explainability via SHAP** for clinical insights
- 🔁 **Reproducible Pipeline** with config-driven experiments and fixed seeds

---

## 🧬 Methodology Overview

### 1. Data Pipeline
- Dataset: **UCI Diabetes 130-US Hospitals (101,766 records)**
- Preprocessing:
  - Remove identifiers and high-missing features
  - Median (numeric) & mode (categorical) imputation
  - Label encoding
  - Binary target: `<30` → 1, else 0 :contentReference[oaicite:1]{index=1}

---

### 2. Federated Learning Setup

- **Clients:** 5 simulated hospitals
- **Algorithm:** FedProx (extension of FedAvg)
- **Communication Rounds:** 5–10
- **Local Training:** 3 epochs per round
- **Aggregation:** Weighted averaging
- **Prediction:** Ensemble voting across models :contentReference[oaicite:2]{index=2}

---

### 3. Differential Privacy Mechanism

- **Gradient Clipping:** L2 norm (C = 1.0)
- **Noise Injection:** Gaussian noise (σ ≈ 7.44)
- **Privacy Budget:** ε = 1.0, δ = 10⁻⁵ :contentReference[oaicite:3]{index=3}

👉 Ensures formal privacy guarantees while maintaining utility

---

### 4. Class Imbalance Strategy

- SMOTE (synthetic minority oversampling)
- SMOTE-ENN (SMOTE + noise cleaning)

📌 According to results (Page 6):
- SMOTE-ENN improves **F1-score & ROC-AUC**
- Slight trade-off in accuracy :contentReference[oaicite:4]{index=4}

---

### 5. Evaluation Protocol

- **Validation:** Stratified 10-Fold Cross Validation
- **Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

---

### 6. Explainability (XAI)

- SHAP used for:
  - Global feature importance
  - Local prediction explanations

📊 Top features (Page 7):
- number_diagnoses
- discharge_disposition_id
- time_in_hospital
- num_inpatient
- age :contentReference[oaicite:5]{index=5}

---

## 📁 Repository Structure

```bash
.
├── configs/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
├── scripts/
├── experiments/
├── tests/
├── docs/
├── README.md
└── requirements.txt
