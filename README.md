# 🫀 Heart Disease Prediction — ML Pipeline

> **AI-Powered Heart Disease Risk Prediction with Explainable AI (SHAP) & Interactive Web App**
>
> Graduation Project 2026

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189FDD)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9ACD32)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blue)

---

## 📌 Overview

A comprehensive **Machine Learning pipeline** for predicting heart disease risk, built on the **CDC BRFSS** dataset with **319,795 patients**. The project includes:

- 🧠 **6 ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network (MLP), Stacking Ensemble
- ⚡ **Bayesian Hyperparameter Tuning** using Optuna (V2 Optimized)
- 🔍 **Explainable AI (XAI)**: SHAP values, Partial Dependence Plots, Feature Importance Comparison
- 🌐 **Interactive Web App**: Streamlit-based real-time predictor with live SHAP explanations
- 📊 **Power BI Integration**: Patient segmentation (K-Means) + data export for dashboards

---

## 🏆 Model Performance

| Version | Model | ROC-AUC | Recall | Description |
|---------|-------|---------|--------|-------------|
| **V2 ★** | **LightGBM V2** | **0.8404** | **80.57%** | Optuna-tuned, best overall |
| V2 | Stacking Ensemble V2 | 0.8403 | 79.18% | 4 tuned base-learners |
| V1 | Stacking Ensemble V1 | 0.8397 | 78.96% | Baseline ensemble |
| V2 | XGBoost V2 | 0.8395 | 80.07% | Optuna-tuned |
| V1 | Logistic Regression | 0.8374 | 78.58% | Linear baseline |

---

## 📂 Project Structure

```
Heart-Disease-Prediction-ML/
├── app.py                        # Streamlit web application
├── ml_pipeline.py                # V1 Baseline ML pipeline (6 models)
├── ml_pipeline_v2.py             # V2 Optimized pipeline (Optuna tuning)
├── heart_disease_project.ipynb   # Complete Jupyter Notebook
├── requirements.txt              # Python dependencies
├── Project_Documentation_AR.md   # Full documentation (Arabic)
└── outputs/                      # Model outputs & visualizations
    ├── *.png                     # ROC curves, confusion matrices, SHAP plots
    ├── *.csv                     # Model comparison tables
    └── v2/                       # V2 optimized outputs
        └── *.png, *.csv
```

> **Note:** Large files (`.xlsx` datasets & `.pkl` models) are excluded from the repository via `.gitignore` due to GitHub size limits.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Abdelrahmandeep/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the ML Pipeline
```bash
# V1 Baseline
python ml_pipeline.py

# V2 Optimized (Optuna)
python ml_pipeline_v2.py
```

### 4. Launch the Web App
```bash
streamlit run app.py
```

---

## 🔬 Key Techniques

| Technique | Purpose |
|-----------|---------|
| **SMOTE** | Handling class imbalance (9% positive rate) |
| **Optuna** | Bayesian hyperparameter optimization (200 trials) |
| **Stacking Ensemble** | Combining LR + RF + XGBoost + LightGBM |
| **SHAP** | Global & local model interpretability |
| **Youden's J** | Optimal classification threshold tuning |
| **K-Means** | Patient risk segmentation for Power BI |
| **Learning Curves** | Overfitting/underfitting diagnosis |

---

## 🖥️ Web Application Features

- **Real-time Prediction**: Enter patient data → instant risk assessment
- **Model Version Toggle**: Switch between V1 (Baseline) and V2 (Optimized)
- **Live SHAP Waterfall**: Explains which factors increased/decreased risk
- **Dynamic Thresholds**: V1 (45%) vs V2 (49.22%) for clinical accuracy
- **Risk Recommendations**: Actionable health advice based on prediction

---

## 📊 Sample Outputs

The `outputs/` directory contains:
- Confusion matrices for all models
- ROC & Precision-Recall curves
- SHAP global importance & beeswarm plots
- Learning curves (overfitting diagnosis)
- V1 vs V2 comparison charts

---

## 📝 License

This project is developed as a graduation project for academic purposes.

---

*Built with ❤️ using Python, scikit-learn, XGBoost, LightGBM, SHAP & Streamlit*
