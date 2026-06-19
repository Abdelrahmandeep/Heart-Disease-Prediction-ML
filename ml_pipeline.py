#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heart Disease ML Pipeline
=========================
Trains 6 ML models on a heart-disease dataset, evaluates them with
comprehensive metrics and plots, tunes the best model's threshold,
runs cross-validation, and persists all artefacts to the outputs/ directory.

Author : auto-generated
Date   : 2026-05-22
"""

# ── Windows encoding fix ────────────────────────────────────────────────────
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Force non-interactive Matplotlib backend BEFORE any other mpl import ────
import matplotlib
matplotlib.use('Agg')

# ── Standard library ────────────────────────────────────────────────────────
import os
import time
import traceback
import warnings

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay,
    average_precision_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# Helper: evaluate a single model
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    output_dir: str = 'outputs',
    threshold: float = 0.5,
):
    """
    Evaluate *model* on the held-out test set and persist plots.

    Returns
    -------
    dict  – metric_name → value
    """
    # ── predictions ──────────────────────────────────────────────────────
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # ── scalar metrics ───────────────────────────────────────────────────
    roc_auc  = roc_auc_score(y_test, y_proba)
    pr_auc   = average_precision_score(y_test, y_proba)
    f1       = f1_score(y_test, y_pred)
    recall   = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    acc      = accuracy_score(y_test, y_pred)

    metrics = {
        'Model': model_name,
        'ROC-AUC': round(roc_auc, 4),
        'PR-AUC': round(pr_auc, 4),
        'F1-Score': round(f1, 4),
        'Recall': round(recall, 4),
        'Precision': round(precision, 4),
        'Accuracy': round(acc, 4),
    }

    safe_name = model_name.replace(' ', '_')

    # ── 1. Confusion Matrix ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No HD', 'HD'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix – {model_name}')
    fig.savefig(
        os.path.join(output_dir, f'{safe_name}_confusion_matrix.png'),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close(fig)

    # ── 2. ROC Curve ────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve – {model_name}')
    ax.legend(loc='lower right')
    fig.savefig(
        os.path.join(output_dir, f'{safe_name}_roc_curve.png'),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close(fig)

    # ── 3. Precision-Recall Curve ───────────────────────────────────────
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_arr, prec_arr, label=f'{model_name} (PR-AUC={pr_auc:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve – {model_name}')
    ax.legend(loc='upper right')
    fig.savefig(
        os.path.join(output_dir, f'{safe_name}_pr_curve.png'),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close(fig)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    t_pipeline = time.time()

    # ── Create output directory ──────────────────────────────────────────
    os.makedirs('outputs', exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # STEP 1 – Data Loading
    # ────────────────────────────────────────────────────────────────────
    print('=' * 70)
    print('STEP 1: Loading data …')
    t0 = time.time()
    df = pd.read_excel('heart_disease_project_full.xlsx', sheet_name='Main_Dataset')
    print(f'  → Loaded {df.shape[0]:,} rows × {df.shape[1]} columns  '
          f'({time.time() - t0:.1f}s)')

    # ────────────────────────────────────────────────────────────────────
    # STEP 2 – Drop columns
    # ────────────────────────────────────────────────────────────────────
    print('\nSTEP 2: Dropping unnecessary columns …')
    t0 = time.time()
    drop_cols = [
        'PatientID', 'State_Code', 'State', 'AgeCategory', 'Diabetic',
        'Survey_Year', 'Survey_Quarter', 'Survey_Month',
        'State_HD_vs_National', 'State_Risk_Label', 'State_Health_Tier',
        'State_Population_M',
    ]
    existing_drops = [c for c in drop_cols if c in df.columns]
    df_ml = df.drop(columns=existing_drops)
    print(f'  → Dropped {len(existing_drops)} columns  '
          f'({time.time() - t0:.2f}s)')

    # ────────────────────────────────────────────────────────────────────
    # STEP 3 – Encoding
    # ────────────────────────────────────────────────────────────────────
    print('\nSTEP 3: Encoding features …')
    t0 = time.time()

    # --- 3a  Binary Yes/No columns ---
    binary_cols = [
        'HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke',
        'DiffWalking', 'PhysicalActivity', 'Diabetic_Binary',
        'Asthma', 'KidneyDisease', 'SkinCancer',
    ]
    yes_no_map = {'Yes': 1, 'No': 0}
    for col in binary_cols:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].map(yes_no_map).fillna(0).astype(int)

    # --- 3b  Sex ---
    if 'Sex' in df_ml.columns:
        df_ml['Sex'] = df_ml['Sex'].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)

    # --- 3c  Ordinal columns ---
    ordinal_maps = {
        'GenHealth': {
            'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4,
        },
        'BMI_Category': {
            'Underweight': 0, 'Normal': 1, 'Overweight': 2,
            'Obese': 3, 'Obese III': 3, 'Obese II': 3, 'Obese I': 3,
        },
        'BMI_Risk': {
            'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 2,
        },
        'Comorbidity_Level': {
            'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3,
        },
        'Risk_Tier': {
            'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3,
        },
        'Lifestyle_Category': {
            'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3,
        },
        'Sleep_Quality': {
            'Short': 0, 'Optimal': 1, 'Long': 2, 'Excessive': 2,
        },
        'Age_Group': {
            'Young (18-34)': 0, 'Middle (35-54)': 1,
            'Senior (55-69)': 2, 'Elderly (70+)': 3,
        },
    }
    for col, mapping in ordinal_maps.items():
        if col not in df_ml.columns:
            continue
        # Detect unmapped values
        unique_vals = set(df_ml[col].dropna().unique())
        unmapped = unique_vals - set(mapping.keys())
        if unmapped:
            print(f'  ⚠ Unmapped values in "{col}": {unmapped}')
        df_ml[col] = df_ml[col].map(mapping)
        median_val = df_ml[col].median()
        if pd.isna(median_val):
            median_val = 0
        df_ml[col] = df_ml[col].fillna(median_val).astype(int)

    # --- 3d  One-Hot encode nominal columns ---
    ohe_cols = [c for c in ['Race', 'Region'] if c in df_ml.columns]
    if ohe_cols:
        df_ml = pd.get_dummies(df_ml, columns=ohe_cols, drop_first=True)

    # Ensure all column names are strings
    df_ml.columns = [str(c) for c in df_ml.columns]
    print(f'  → Encoding complete. Shape: {df_ml.shape}  '
          f'({time.time() - t0:.2f}s)')

    # ────────────────────────────────────────────────────────────────────
    # STEP 4 – Feature Engineering (Interaction Features)
    # ────────────────────────────────────────────────────────────────────
    print('\nSTEP 4: Creating interaction features …')
    t0 = time.time()

    interactions = {
        'Age_BMI_Interaction': ('Age_Numeric', 'BMI'),
        'Smoke_Diabetes': ('Smoking', 'Diabetic_Binary'),
        'Comorbidity_Age': ('Comorbidity_Count', 'Age_Numeric'),
        'Stroke_Kidney': ('Stroke', 'KidneyDisease'),
    }
    for new_col, (a, b) in interactions.items():
        if a in df_ml.columns and b in df_ml.columns:
            df_ml[new_col] = df_ml[a] * df_ml[b]
            print(f'  ✓ Created {new_col}')
        else:
            missing = [c for c in (a, b) if c not in df_ml.columns]
            print(f'  ✗ Skipped {new_col} (missing: {missing})')

    print(f'  → Done ({time.time() - t0:.2f}s)')

    # ────────────────────────────────────────────────────────────────────
    # STEP 5 – Train / Test Split
    # ────────────────────────────────────────────────────────────────────
    print('\nSTEP 5: Splitting data (80/20, stratified) …')
    t0 = time.time()

    X = df_ml.drop('HeartDisease', axis=1)
    y = df_ml['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f'  → Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}  '
          f'({time.time() - t0:.2f}s)')

    # ────────────────────────────────────────────────────────────────────
    # STEP 6 – Scaling (fit on train only)
    # ────────────────────────────────────────────────────────────────────
    print('\nSTEP 6: Scaling numerical features …')
    t0 = time.time()

    num_cols = [
        'BMI', 'Age_Numeric', 'PhysicalHealth', 'MentalHealth',
        'SleepTime', 'Risk_Score', 'Lifestyle_Score', 'Health_Score',
        'Comorbidity_Count', 'State_HD_Prevalence', 'State_Obesity_Rate',
        'State_Smoking_Rate', 'State_Uninsured_Rate',
        'State_Median_Income_K', 'State_Health_Rank',
        'Age_BMI_Interaction', 'Comorbidity_Age',
    ]
    num_cols = [c for c in num_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    print(f'  → Scaled {len(num_cols)} columns  ({time.time() - t0:.2f}s)')

    # ────────────────────────────────────────────────────────────────────
    # STEP 7 – SMOTE on training data only
    # ────────────────────────────────────────────────────────────────────
    print('\nSTEP 7: Applying SMOTE …')
    t0 = time.time()

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f'  → Before SMOTE: {y_train.value_counts().to_dict()}')
    print(f'  → After  SMOTE: {y_train_sm.value_counts().to_dict()}')
    print(f'  → ({time.time() - t0:.1f}s)')

    # ────────────────────────────────────────────────────────────────────
    # STEP 8 – Train 6 Models
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 8: Training models …')
    models = {}

    # ── Model 1: Logistic Regression ─────────────────────────────────────
    print('\n  [1/6] Logistic Regression …')
    t0 = time.time()
    lr = LogisticRegression(
        C=1.0, solver='lbfgs', class_weight='balanced',
        max_iter=1000, random_state=42,
    )
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    print(f'        Done ({time.time() - t0:.1f}s)')

    # ── Model 2: Random Forest ──────────────────────────────────────────
    print('  [2/6] Random Forest …')
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    print(f'        Done ({time.time() - t0:.1f}s)')

    # ── Model 3: XGBoost ────────────────────────────────────────────────
    print('  [3/6] XGBoost …')
    t0 = time.time()
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=10.7, eval_metric='auc',
        random_state=42,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    models['XGBoost'] = xgb_model
    print(f'        Done ({time.time() - t0:.1f}s)')

    # ── Model 4: LightGBM ──────────────────────────────────────────────
    print('  [4/6] LightGBM …')
    t0 = time.time()
    lgbm = LGBMClassifier(
        n_estimators=300, num_leaves=63, learning_rate=0.05,
        is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    models['LightGBM'] = lgbm
    print(f'        Done ({time.time() - t0:.1f}s)')

    # ── Model 5: Neural Network (MLPClassifier) ─────────────────────────
    print('  [5/6] Neural Network (MLP) – trained on SMOTE data …')
    t0 = time.time()
    nn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True,
    )
    nn_model.fit(X_train_sm, y_train_sm)
    models['Neural Network'] = nn_model
    print(f'        Done ({time.time() - t0:.1f}s)')

    # ── Model 6: Stacking Ensemble ──────────────────────────────────────
    print('  [6/6] Stacking Ensemble …')
    t0 = time.time()
    stacking = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(
                C=1.0, solver='lbfgs', class_weight='balanced',
                max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(
                n_estimators=100, max_depth=12,
                class_weight='balanced', random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                scale_pos_weight=10.7, eval_metric='auc',
                random_state=42)),
        ],
        final_estimator=LogisticRegression(
            class_weight='balanced', max_iter=1000,
        ),
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    stacking.fit(X_train, y_train)
    models['Stacking Ensemble'] = stacking
    print(f'        Done ({time.time() - t0:.1f}s)')

    # ────────────────────────────────────────────────────────────────────
    # STEP 9 – Evaluation
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 9: Evaluating all models …')
    t0 = time.time()
    all_metrics = []

    for name, model in models.items():
        print(f'  → Evaluating {name} …')
        m = evaluate_model(model, X_test, y_test, name, output_dir='outputs')
        all_metrics.append(m)

    results_df = pd.DataFrame(all_metrics)
    print('\n  Model Comparison:')
    print(results_df.to_string(index=False))

    # ── Combined ROC curve ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Combined ROC Curves – All Models')
    ax.legend(loc='lower right', fontsize=8)
    fig.savefig('outputs/combined_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Comparison bar chart ────────────────────────────────────────────
    metric_cols = ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Recall', 'Precision', 'Accuracy']
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.12
    for i, metric in enumerate(metric_cols):
        ax.bar(x + i * width, results_df[metric], width, label=metric)
    ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
    ax.set_xticklabels(results_df['Model'], rotation=20, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.legend(loc='upper left', fontsize=7)
    fig.savefig('outputs/model_comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n  → Evaluation complete ({time.time() - t0:.1f}s)')

    # ────────────────────────────────────────────────────────────────────
    # STEP 10 – Threshold Tuning (best model by ROC-AUC)
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 10: Threshold tuning for best model …')
    t0 = time.time()

    best_row = results_df.loc[results_df['ROC-AUC'].idxmax()]
    best_name = best_row['Model']
    best_model = models[best_name]
    print(f'  → Best model by ROC-AUC: {best_name} ({best_row["ROC-AUC"]:.4f})')

    # Youden's J statistic
    y_proba_best = best_model.predict_proba(X_test)[:, 1]
    fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_proba_best)
    j_scores = tpr_best - fpr_best
    opt_idx = np.argmax(j_scores)
    opt_threshold = thresholds_best[opt_idx]
    print(f'  → Optimal threshold (Youden J): {opt_threshold:.4f}')

    # Re-evaluate at optimal threshold
    tuned_metrics = evaluate_model(
        best_model, X_test, y_test,
        f'{best_name}_Tuned',
        output_dir='outputs',
        threshold=opt_threshold,
    )
    print('  → Tuned metrics:')
    for k, v in tuned_metrics.items():
        if k != 'Model':
            print(f'      {k}: {v}')
    print(f'  → ({time.time() - t0:.2f}s)')

    # ────────────────────────────────────────────────────────────────────
    # STEP 11 – Cross-Validation for best model
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 11: Cross-validation (5-fold Stratified) …')
    t0 = time.time()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        best_model, X_train, y_train,
        cv=cv, scoring='roc_auc', n_jobs=-1,
    )
    print(f'  → CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')
    print(f'  → Fold scores: {[round(s, 4) for s in cv_scores]}')
    print(f'  → ({time.time() - t0:.1f}s)')

    # ────────────────────────────────────────────────────────────────────
    # STEP 12 – Save everything
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 12: Saving artefacts …')
    t0 = time.time()

    joblib.dump(best_model, 'outputs/best_model.pkl')
    joblib.dump(scaler,     'outputs/scaler.pkl')
    joblib.dump(X_train.columns.tolist(), 'outputs/feature_names.pkl')
    joblib.dump(num_cols,   'outputs/num_cols.pkl')

    results_df.to_csv('outputs/model_comparison.csv', index=False)

    print('  ✓ outputs/best_model.pkl')
    print('  ✓ outputs/scaler.pkl')
    print('  ✓ outputs/feature_names.pkl')
    print('  ✓ outputs/num_cols.pkl')
    print('  ✓ outputs/model_comparison.csv')
    print(f'  → ({time.time() - t0:.2f}s)')

    # ────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_pipeline
    print('\n' + '=' * 70)
    print('PIPELINE COMPLETE')
    print('=' * 70)
    print(f'  Total elapsed time : {elapsed / 60:.1f} min ({elapsed:.0f}s)')
    print(f'  Best model         : {best_name}')
    print(f'  Best ROC-AUC       : {best_row["ROC-AUC"]:.4f}')
    print(f'  Optimal threshold  : {opt_threshold:.4f}')
    print(f'  CV AUC             : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    print()
    print('  Comparison table:')
    print(results_df.to_string(index=False))
    print()
    print('  Saved artefacts in outputs/:')
    for f in sorted(os.listdir('outputs')):
        fpath = os.path.join('outputs', f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f'    {f:45s}  ({size_kb:,.1f} KB)')
    print('=' * 70)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
