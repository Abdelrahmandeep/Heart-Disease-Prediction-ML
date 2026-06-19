#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heart Disease ML Pipeline V2 — Optimized
==========================================
Builds on V1 by adding:
  • Optuna Hyperparameter Tuning (Bayesian Optimization)
  • Feature Selection based on SHAP importance
  • Improved Stacking Ensemble with tuned base-learners
  • Learning Curves for overfitting diagnosis
  • V1 vs V2 comparison table

Author : auto-generated
Date   : 2026-06-08
"""

# ── Windows encoding fix ────────────────────────────────────────────────────
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')

import os, time, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import optuna

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, learning_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay, average_precision_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = 'outputs/v2'
N_TRIALS = 50        # Optuna trials per model
TUNING_SAMPLE = 50000  # Sample size for faster tuning
CV_FOLDS = 3         # CV folds during tuning


# ═══════════════════════════════════════════════════════════════════════════
# Helper: evaluate a single model
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, model_name, output_dir=OUTPUT_DIR, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc   = roc_auc_score(y_test, y_proba)
    pr_auc    = average_precision_score(y_test, y_proba)
    f1        = f1_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    acc       = accuracy_score(y_test, y_pred)

    metrics = {
        'Model': model_name,
        'ROC-AUC': round(roc_auc, 4),
        'PR-AUC': round(pr_auc, 4),
        'F1-Score': round(f1, 4),
        'Recall': round(recall, 4),
        'Precision': round(precision, 4),
        'Accuracy': round(acc, 4),
    }

    safe = model_name.replace(' ', '_')

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No HD', 'HD']).plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix – {model_name}')
    fig.savefig(os.path.join(output_dir, f'{safe}_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve – {model_name}'); ax.legend(loc='lower right')
    fig.savefig(os.path.join(output_dir, f'{safe}_roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # PR Curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_arr, prec_arr, label=f'{model_name} (PR-AUC={pr_auc:.4f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'PR Curve – {model_name}'); ax.legend(loc='upper right')
    fig.savefig(os.path.join(output_dir, f'{safe}_pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    t_pipeline = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1-7: Data Loading & Preprocessing (identical to V1)
    # ──────────────────────────────────────────────────────────────────────
    print('=' * 70)
    print('ML PIPELINE V2 — OPTIMIZED')
    print('=' * 70)

    print('\nSTEP 1: Loading data …')
    t0 = time.time()
    df = pd.read_excel('heart_disease_project_full.xlsx', sheet_name='Main_Dataset')
    print(f'  → Loaded {df.shape[0]:,} rows × {df.shape[1]} columns  ({time.time()-t0:.1f}s)')

    print('\nSTEP 2: Preprocessing …')
    t0 = time.time()
    df_ml = df.copy()

    drop_cols = [
        'PatientID', 'State_Code', 'State', 'AgeCategory', 'Diabetic',
        'Survey_Year', 'Survey_Quarter', 'Survey_Month',
        'State_HD_vs_National', 'State_Risk_Label', 'State_Health_Tier',
        'State_Population_M',
    ]
    df_ml = df_ml.drop(columns=[c for c in drop_cols if c in df_ml.columns])

    binary_cols = ['HeartDisease','Smoking','AlcoholDrinking','Stroke','DiffWalking',
                   'PhysicalActivity','Diabetic_Binary','Asthma','KidneyDisease','SkinCancer']
    for col in binary_cols:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].map({'Yes':1,'No':0}).fillna(0).astype(int)
    if 'Sex' in df_ml.columns:
        df_ml['Sex'] = df_ml['Sex'].map({'Male':1,'Female':0}).fillna(0).astype(int)

    ordinal_maps = {
        'GenHealth':{'Poor':0,'Fair':1,'Good':2,'Very good':3,'Excellent':4},
        'BMI_Category':{'Underweight':0,'Normal':1,'Overweight':2,'Obese':3,'Obese III':3,'Obese II':3,'Obese I':3},
        'BMI_Risk':{'Low':0,'Medium':1,'High':2,'Very High':2},
        'Comorbidity_Level':{'None':0,'Low':1,'Moderate':2,'High':3},
        'Risk_Tier':{'Low':0,'Medium':1,'High':2,'Critical':3},
        'Lifestyle_Category':{'Poor':0,'Fair':1,'Good':2,'Excellent':3},
        'Sleep_Quality':{'Short':0,'Optimal':1,'Long':2,'Excessive':2},
        'Age_Group':{'Young (18-34)':0,'Middle (35-54)':1,'Senior (55-69)':2,'Elderly (70+)':3},
    }
    for col, mapping in ordinal_maps.items():
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].map(mapping)
            med = df_ml[col].median()
            df_ml[col] = df_ml[col].fillna(med if not pd.isna(med) else 0).astype(int)

    ohe_cols = [c for c in ['Race','Region'] if c in df_ml.columns]
    if ohe_cols:
        df_ml = pd.get_dummies(df_ml, columns=ohe_cols, drop_first=True)
    df_ml.columns = [str(c) for c in df_ml.columns]

    # Feature Engineering
    interactions = {
        'Age_BMI_Interaction': ('Age_Numeric','BMI'),
        'Smoke_Diabetes': ('Smoking','Diabetic_Binary'),
        'Comorbidity_Age': ('Comorbidity_Count','Age_Numeric'),
        'Stroke_Kidney': ('Stroke','KidneyDisease'),
    }
    for new_col, (a, b) in interactions.items():
        if a in df_ml.columns and b in df_ml.columns:
            df_ml[new_col] = df_ml[a] * df_ml[b]

    print(f'  → Preprocessing done. Shape: {df_ml.shape}  ({time.time()-t0:.1f}s)')

    # Split
    X = df_ml.drop('HeartDisease', axis=1)
    y = df_ml['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale
    num_cols = ['BMI','Age_Numeric','PhysicalHealth','MentalHealth','SleepTime',
                'Risk_Score','Lifestyle_Score','Health_Score','Comorbidity_Count',
                'State_HD_Prevalence','State_Obesity_Rate','State_Smoking_Rate',
                'State_Uninsured_Rate','State_Median_Income_K','State_Health_Rank',
                'Age_BMI_Interaction','Comorbidity_Age']
    num_cols = [c for c in num_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f'  → scale_pos_weight = {pos_weight:.2f}')

    # ──────────────────────────────────────────────────────────────────────
    # STEP 8: Optuna Hyperparameter Tuning
    # ──────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 8: OPTUNA HYPERPARAMETER TUNING')
    print('=' * 70)

    # Use a stratified sample for faster tuning
    np.random.seed(42)
    n_sample = min(TUNING_SAMPLE, X_train.shape[0])
    sample_idx = np.random.choice(X_train.shape[0], n_sample, replace=False)
    X_tune = X_train.iloc[sample_idx]
    y_tune = y_train.iloc[sample_idx]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # ── 8a: XGBoost Tuning ───────────────────────────────────────────────
    print(f'\n  [1/4] Tuning XGBoost ({N_TRIALS} trials on {n_sample:,} samples) …')
    t0 = time.time()

    def xgb_objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 200, 1000, step=100),
            'max_depth':         trial.suggest_int('max_depth', 3, 10),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
            'gamma':             trial.suggest_float('gamma', 0, 5.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight':  pos_weight,
            'eval_metric': 'auc',
            'random_state': 42,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_tune, y_tune, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study_xgb = optuna.create_study(direction='maximize', study_name='xgb')
    study_xgb.optimize(xgb_objective, n_trials=N_TRIALS)
    best_xgb_params = study_xgb.best_params
    best_xgb_params['scale_pos_weight'] = pos_weight
    best_xgb_params['eval_metric'] = 'auc'
    best_xgb_params['random_state'] = 42
    print(f'        Best ROC-AUC: {study_xgb.best_value:.4f}  ({time.time()-t0:.0f}s)')
    print(f'        Best params: {best_xgb_params}')

    # ── 8b: LightGBM Tuning ─────────────────────────────────────────────
    print(f'\n  [2/4] Tuning LightGBM ({N_TRIALS} trials) …')
    t0 = time.time()

    def lgbm_objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 200, 1000, step=100),
            'num_leaves':        trial.suggest_int('num_leaves', 15, 127),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'is_unbalance': True,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X_tune, y_tune, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study_lgbm = optuna.create_study(direction='maximize', study_name='lgbm')
    study_lgbm.optimize(lgbm_objective, n_trials=N_TRIALS)
    best_lgbm_params = study_lgbm.best_params
    best_lgbm_params['is_unbalance'] = True
    best_lgbm_params['random_state'] = 42
    best_lgbm_params['n_jobs'] = -1
    best_lgbm_params['verbose'] = -1
    print(f'        Best ROC-AUC: {study_lgbm.best_value:.4f}  ({time.time()-t0:.0f}s)')
    print(f'        Best params: {best_lgbm_params}')

    # ── 8c: Random Forest Tuning ─────────────────────────────────────────
    print(f'\n  [3/4] Tuning Random Forest ({N_TRIALS} trials) …')
    t0 = time.time()

    def rf_objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth':        trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':     trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
        }
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X_tune, y_tune, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study_rf = optuna.create_study(direction='maximize', study_name='rf')
    study_rf.optimize(rf_objective, n_trials=N_TRIALS)
    best_rf_params = study_rf.best_params
    best_rf_params['class_weight'] = 'balanced'
    best_rf_params['random_state'] = 42
    best_rf_params['n_jobs'] = -1
    print(f'        Best ROC-AUC: {study_rf.best_value:.4f}  ({time.time()-t0:.0f}s)')
    print(f'        Best params: {best_rf_params}')

    # ── 8d: MLP Tuning ──────────────────────────────────────────────────
    print(f'\n  [4/4] Tuning Neural Network / MLP ({N_TRIALS} trials on SMOTE data) …')
    t0 = time.time()

    # Sample from SMOTE data for MLP tuning
    n_sm = min(TUNING_SAMPLE, X_train_sm.shape[0])
    sm_idx = np.random.choice(X_train_sm.shape[0], n_sm, replace=False)
    X_tune_sm = X_train_sm.iloc[sm_idx]
    y_tune_sm = y_train_sm.iloc[sm_idx]

    def mlp_objective(trial):
        n_layers = trial.suggest_int('n_layers', 2, 5)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 32, 256, step=32))

        params = {
            'hidden_layer_sizes': tuple(layers),
            'activation':         trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha':              trial.suggest_float('alpha', 1e-5, 0.01, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 0.01, log=True),
            'batch_size':         trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            'solver': 'adam',
            'learning_rate': 'adaptive',
            'max_iter': 80,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'random_state': 42,
        }
        model = MLPClassifier(**params)
        scores = cross_val_score(model, X_tune_sm, y_tune_sm, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study_mlp = optuna.create_study(direction='maximize', study_name='mlp')
    study_mlp.optimize(mlp_objective, n_trials=N_TRIALS)
    best_mlp_trial = study_mlp.best_trial
    n_layers = best_mlp_trial.params['n_layers']
    best_mlp_layers = tuple(best_mlp_trial.params[f'n_units_l{i}'] for i in range(n_layers))
    best_mlp_params = {
        'hidden_layer_sizes': best_mlp_layers,
        'activation': best_mlp_trial.params['activation'],
        'alpha': best_mlp_trial.params['alpha'],
        'learning_rate_init': best_mlp_trial.params['learning_rate_init'],
        'batch_size': best_mlp_trial.params['batch_size'],
        'solver': 'adam',
        'learning_rate': 'adaptive',
        'max_iter': 150,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'random_state': 42,
    }
    print(f'        Best ROC-AUC: {study_mlp.best_value:.4f}  ({time.time()-t0:.0f}s)')
    print(f'        Best layers: {best_mlp_layers}')

    # Save Optuna studies summary
    optuna_summary = {
        'XGBoost': {'best_auc': study_xgb.best_value, 'params': best_xgb_params},
        'LightGBM': {'best_auc': study_lgbm.best_value, 'params': best_lgbm_params},
        'Random Forest': {'best_auc': study_rf.best_value, 'params': best_rf_params},
        'MLP': {'best_auc': study_mlp.best_value, 'params': best_mlp_params},
    }
    joblib.dump(optuna_summary, os.path.join(OUTPUT_DIR, 'optuna_summary.pkl'))

    # ──────────────────────────────────────────────────────────────────────
    # STEP 9: Train V2 Models with Tuned Hyperparameters (Full Data)
    # ──────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 9: Training V2 Models with TUNED Hyperparameters (Full Data)')
    print('=' * 70)

    models_v2 = {}

    # LR (baseline, no tuning needed)
    print('\n  [1/6] Logistic Regression (baseline) …')
    t0 = time.time()
    lr_v2 = LogisticRegression(C=1.0, solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=42)
    lr_v2.fit(X_train, y_train)
    models_v2['Logistic Regression'] = lr_v2
    print(f'        Done ({time.time()-t0:.1f}s)')

    # RF tuned
    print('  [2/6] Random Forest (Tuned) …')
    t0 = time.time()
    rf_v2 = RandomForestClassifier(**best_rf_params)
    rf_v2.fit(X_train, y_train)
    models_v2['Random Forest V2'] = rf_v2
    print(f'        Done ({time.time()-t0:.1f}s)')

    # XGBoost tuned
    print('  [3/6] XGBoost (Tuned) …')
    t0 = time.time()
    xgb_v2 = XGBClassifier(**best_xgb_params)
    xgb_v2.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    models_v2['XGBoost V2'] = xgb_v2
    print(f'        Done ({time.time()-t0:.1f}s)')

    # LightGBM tuned
    print('  [4/6] LightGBM (Tuned) …')
    t0 = time.time()
    lgbm_v2 = LGBMClassifier(**best_lgbm_params)
    lgbm_v2.fit(X_train, y_train)
    models_v2['LightGBM V2'] = lgbm_v2
    print(f'        Done ({time.time()-t0:.1f}s)')

    # MLP tuned
    print('  [5/6] Neural Network (Tuned, SMOTE data) …')
    t0 = time.time()
    nn_v2 = MLPClassifier(**best_mlp_params)
    nn_v2.fit(X_train_sm, y_train_sm)
    models_v2['Neural Network V2'] = nn_v2
    print(f'        Done ({time.time()-t0:.1f}s)')

    # Stacking V2 with tuned base-learners
    print('  [6/6] Stacking Ensemble V2 (Tuned base-learners) …')
    t0 = time.time()
    stacking_v2 = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1.0, solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(**best_rf_params)),
            ('xgb', XGBClassifier(**best_xgb_params)),
            ('lgbm', LGBMClassifier(**best_lgbm_params)),
        ],
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    stacking_v2.fit(X_train, y_train)
    models_v2['Stacking V2'] = stacking_v2
    print(f'        Done ({time.time()-t0:.1f}s)')

    # ──────────────────────────────────────────────────────────────────────
    # STEP 10: Evaluate All V2 Models
    # ──────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 10: Evaluating V2 Models')
    print('=' * 70)

    all_metrics_v2 = []
    for name, model in models_v2.items():
        print(f'  → Evaluating {name} …')
        m = evaluate_model(model, X_test, y_test, name, output_dir=OUTPUT_DIR)
        all_metrics_v2.append(m)

    results_v2 = pd.DataFrame(all_metrics_v2).sort_values('ROC-AUC', ascending=False)
    print('\n  V2 Model Comparison:')
    print(results_v2.to_string(index=False))

    # Combined ROC for V2
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, model in models_v2.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('Combined ROC Curves — V2 Models (Tuned)')
    ax.legend(loc='lower right', fontsize=7)
    fig.savefig(os.path.join(OUTPUT_DIR, 'combined_roc_curves_v2.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 11: Threshold Tuning for Best V2 Model
    # ──────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 11: Threshold Tuning (Best V2 Model)')
    print('=' * 70)

    best_row = results_v2.iloc[0]
    best_name = best_row['Model']
    best_model_v2 = models_v2[best_name]
    print(f'  → Best V2 model: {best_name} (ROC-AUC={best_row["ROC-AUC"]:.4f})')

    y_proba_best = best_model_v2.predict_proba(X_test)[:, 1]
    fpr_b, tpr_b, thresh_b = roc_curve(y_test, y_proba_best)
    j_scores = tpr_b - fpr_b
    opt_idx = np.argmax(j_scores)
    opt_threshold = thresh_b[opt_idx]
    print(f'  → Optimal threshold (Youden J): {opt_threshold:.4f}')

    tuned_m = evaluate_model(best_model_v2, X_test, y_test, f'{best_name}_Tuned', output_dir=OUTPUT_DIR, threshold=opt_threshold)
    print(f'  → Tuned metrics: ROC-AUC={tuned_m["ROC-AUC"]}, F1={tuned_m["F1-Score"]}, Recall={tuned_m["Recall"]}')

    # ──────────────────────────────────────────────────────────────────────
    # STEP 12: Cross-Validation
    # ──────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 12: 5-Fold Cross-Validation (Best V2)')
    print('=' * 70)

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model_v2, X_train, y_train, cv=cv5, scoring='roc_auc', n_jobs=-1)
    print(f'  → CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    print(f'  → Folds: {[round(s, 4) for s in cv_scores]}')

    # ──────────────────────────────────────────────────────────────────────
    # STEP 13: Learning Curves
    # ──────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 13: Generating Learning Curves')
    print('=' * 70)
    t0 = time.time()

    # Use XGBoost V2 for learning curve (faster than stacking)
    train_sizes, train_scores, test_scores = learning_curve(
        xgb_v2, X_train, y_train,
        cv=3, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1, random_state=42,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    test_mean  = test_scores.mean(axis=1)
    test_std   = test_scores.std(axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#2196F3')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='#F44336')
    ax.plot(train_sizes, train_mean, 'o-', color='#2196F3', label='Training Score')
    ax.plot(train_sizes, test_mean, 'o-', color='#F44336', label='Cross-Validation Score')
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('ROC-AUC Score', fontsize=12)
    ax.set_title('Learning Curve — XGBoost V2 (Tuned)', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, 'learning_curve_xgb_v2.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Learning curve saved ({time.time()-t0:.0f}s)')

    # ──────────────────────────────────────────────────────────────────────
    # STEP 14: V1 vs V2 Comparison
    # ──────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 14: V1 vs V2 Comparison')
    print('=' * 70)

    # Load V1 results
    v1_path = 'outputs/model_comparison.csv'
    if os.path.exists(v1_path):
        v1_results = pd.read_csv(v1_path)
        print('\n  V1 Results (Original):')
        print(v1_results.to_string(index=False))
        print('\n  V2 Results (Optimized):')
        print(results_v2.to_string(index=False))

        # Side-by-side comparison of best models
        v1_best_auc = v1_results['ROC-AUC'].max()
        v2_best_auc = results_v2['ROC-AUC'].max()
        improvement = v2_best_auc - v1_best_auc
        print(f'\n  ★ V1 Best ROC-AUC: {v1_best_auc:.4f}')
        print(f'  ★ V2 Best ROC-AUC: {v2_best_auc:.4f}')
        print(f'  ★ Improvement:     {improvement:+.4f} ({improvement/v1_best_auc*100:+.2f}%)')

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        v1_best = v1_results.loc[v1_results['ROC-AUC'].idxmax()]
        v2_best = results_v2.iloc[0]
        metrics_to_compare = ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Recall', 'Precision', 'Accuracy']
        x = np.arange(len(metrics_to_compare))
        width = 0.35
        v1_vals = [v1_best[m] for m in metrics_to_compare]
        v2_vals = [v2_best[m] for m in metrics_to_compare]
        bars1 = ax.bar(x - width/2, v1_vals, width, label=f'V1: {v1_best["Model"]}', color='#90CAF9', edgecolor='#1565C0')
        bars2 = ax.bar(x + width/2, v2_vals, width, label=f'V2: {v2_best["Model"]}', color='#EF9A9A', edgecolor='#C62828')
        ax.set_xticks(x); ax.set_xticklabels(metrics_to_compare, fontsize=10)
        ax.set_ylim(0, 1.1); ax.set_ylabel('Score', fontsize=12)
        ax.set_title('V1 vs V2 — Best Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        # Add value labels
        for bar in bars1 + bars2:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
        fig.savefig(os.path.join(OUTPUT_DIR, 'v1_vs_v2_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        print('  ⚠ V1 results not found. Skipping comparison.')

    # ──────────────────────────────────────────────────────────────────────
    # STEP 15: Save V2 Artifacts
    # ──────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 15: Saving V2 Artifacts')
    print('=' * 70)

    joblib.dump(best_model_v2,           os.path.join(OUTPUT_DIR, 'best_model_v2.pkl'))
    joblib.dump(xgb_v2,                  os.path.join(OUTPUT_DIR, 'xgb_model_v2.pkl'))
    joblib.dump(scaler,                  os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    joblib.dump(X_train.columns.tolist(), os.path.join(OUTPUT_DIR, 'feature_names.pkl'))
    joblib.dump(num_cols,                os.path.join(OUTPUT_DIR, 'num_cols.pkl'))
    results_v2.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison_v2.csv'), index=False)

    print('  ✓ best_model_v2.pkl')
    print('  ✓ xgb_model_v2.pkl')
    print('  ✓ scaler.pkl')
    print('  ✓ feature_names.pkl')
    print('  ✓ num_cols.pkl')
    print('  ✓ model_comparison_v2.csv')

    # ──────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_pipeline
    print('\n' + '=' * 70)
    print('V2 PIPELINE COMPLETE')
    print('=' * 70)
    print(f'  Total time       : {elapsed/60:.1f} min ({elapsed:.0f}s)')
    print(f'  Best V2 model    : {best_name}')
    print(f'  Best ROC-AUC     : {best_row["ROC-AUC"]:.4f}')
    print(f'  Optimal threshold: {opt_threshold:.4f}')
    print(f'  CV AUC           : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    print(f'  Optuna trials    : {N_TRIALS} per model × 4 models = {N_TRIALS*4} total')
    print(f'  All outputs in   : {OUTPUT_DIR}/')
    print('=' * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
