import pandas as pd
import numpy as np
import argparse
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from scipy.stats import sem, t
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
import optuna
import shap
import matplotlib.pyplot as plt
from tabpfn_extensions.interpretability.shap import get_shap_values
import os, os.path as osp
import json
import hashlib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import shapiq

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=100, help='Number of top features to select')
parser.add_argument('--num_trials', type=int, default=30, help='Number of Optuna trials')

parser.add_argument('--model_type', type=str, default='lightgbm',
    choices=['rf', 'lightgbm', 'catboost', 'xgboost', 'tabpfn'],
    help='Model type: rf, lightgbm, catboost, xgboost, or tabpfn')

parser.add_argument('--normalize', action='store_true', help='Enable feature normalization (StandardScaler)')
args = parser.parse_args()

# Load main data
file_path = 'data/SupplementaryData1.xlsx'
df = pd.read_excel(file_path, sheet_name="lipidomics_data_males")

# Prepare features and target
y_raw = df['Presence of adrenal insufficiency '].astype(str).str.strip()
y = (y_raw == 'AI').astype(int)
# Select all columns except the specified ones for X
exclude_cols = [
    'Sample ID',
    'Presence of Cerebral ALD ',
    'Presence of adrenal insufficiency ',
    'Severity of Spinal cord disease '
]
X = df.drop(columns=exclude_cols)

# Fix LightGBM compatibility by removing colon
X = X.rename(columns=lambda col: col.replace(":", "_"))

# Patient ID for deduplication
patient_id_col = 'Sample ID'

# Nested CV parameters
outer_k = 5
inner_k = 3
random_state = 42

# --- Experiment output folder setup ---
exp_root = "experiments"
os.makedirs(exp_root, exist_ok=True)
run_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
rand_hash = hashlib.sha1(os.urandom(8)).hexdigest()[:6]
exp_dir = os.path.join(exp_root, f"{run_time}-{rand_hash}")
os.makedirs(exp_dir, exist_ok=True)

# Store CLI args and run info
run_log = {
    'args': vars(args),
    'run_time': run_time,
    'random_hash': rand_hash,
    'metrics': {},
    'per_fold': [],
}

# Store results
outer_roc_aucs = []
outer_pr_aucs = []
per_fold_rows = []

# Outer CV
outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=random_state)
for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    # Deduplicate validation set by patient
    val_patients = df.iloc[test_idx][patient_id_col].drop_duplicates()
    val_mask = df[patient_id_col].isin(val_patients)
    val_idx = df[val_mask].index.intersection(test_idx)
    # Remove these patients from training set
    train_mask = ~df[patient_id_col].isin(val_patients)
    train_idx = df[train_mask].index.intersection(train_idx)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Normalization and imputation logic
    X_train_proc = X_train.copy()
    X_val_proc = X_val.copy()

    imputer = KNNImputer(n_neighbors=5)
    X_train_proc = pd.DataFrame(imputer.fit_transform(X_train_proc), columns=X_train.columns, index=X_train.index)
    X_val_proc = pd.DataFrame(imputer.transform(X_val_proc), columns=X_val.columns, index=X_val.index)
    
    if args.normalize:
        scaler = StandardScaler()
        X_train_proc = pd.DataFrame(scaler.fit_transform(X_train_proc), columns=X_train.columns, index=X_train.index)
        X_val_proc = pd.DataFrame(scaler.transform(X_val_proc), columns=X_val.columns, index=X_val.index)

    # Univariate feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=args.k)
    selector.fit(X_train_proc, y_train)

    selected_cols = X_train_proc.columns[selector.get_support()]
    X_train_final = X_train_proc[selected_cols]
    X_val_final = X_val_proc[selected_cols]

    X_train_imp = X_train_final
    X_val_imp = X_val_final

    if args.model_type == "tabpfn":
        model = TabPFNClassifier(device="cpu")
        model.fit(X_train_imp.values, y_train.values)
        y_pred_prob = model.predict_proba(X_val_imp.values)[:, 1]
        y_pred = model.predict(X_val_imp.values)
    else:
        # Inner CV for hyperparameter tuning (no feature selection here, features are pre-selected)
        def objective(trial):
            inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=random_state)
            scores = []

            for inner_train_idx, inner_test_idx in inner_cv.split(X_train_imp, y_train):
                X_inner_train, y_inner_train = X_train_imp.iloc[inner_train_idx], y_train.iloc[inner_train_idx]
                X_inner_test, y_inner_test = X_train_imp.iloc[inner_test_idx], y_train.iloc[inner_test_idx]
                
                if args.model_type == "rf":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    model_class = RandomForestClassifier
                    
                elif args.model_type == "catboost":
                    params = {
                        'iterations': trial.suggest_int('iterations', 100, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'depth': trial.suggest_int('depth', 3, 8),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                        'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
                        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                        'eval_metric': 'AUC',
                        'loss_function': 'Logloss',
                        'verbose': 0,
                        'random_seed': 42,
                        'allow_writing_files': False
                    }
                    model_class = CatBoostClassifier

                elif args.model_type == "xgboost":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 8),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'tree_method': 'hist',
                        'enable_categorical': True,
                        'eval_metric': 'auc',
                        'random_state': 42
                    }
                    model_class = XGBClassifier

                elif args.model_type == "lightgbm":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 8),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'objective': 'binary',
                        'boosting_type': 'gbdt',
                        'random_state': 42,
                        'verbose': -1
                    }
                    model_class = LGBMClassifier

                pipeline = Pipeline([
                    ('smote', SVMSMOTE(random_state=random_state)),
                    ('clf', model_class(**params))
                ])
                pipeline.fit(X_inner_train, y_inner_train)
                y_pred_prob = pipeline.predict_proba(X_inner_test)[:, 1]
                score = average_precision_score(y_inner_test, y_pred_prob)
                scores.append(score)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.num_trials, show_progress_bar=False)
        best_params = study.best_params

        # Train final model on full training set with best params
        if args.model_type == "catboost":
            best_params.update({
                'eval_metric': 'AUC',
                'loss_function': 'Logloss',
                'verbose': 0,
                'random_seed': 42,
                'allow_writing_files': False
            })
            model_class = CatBoostClassifier

        elif args.model_type == "xgboost":
            best_params.update({
                'tree_method': 'hist',
                'enable_categorical': True,
                'eval_metric': 'auc',
                'random_state': 42
            })
            model_class = XGBClassifier

        elif args.model_type == "lightgbm":
            best_params.update({
                'objective': 'binary',
                'boosting_type': 'gbdt',
                'random_state': 42,
                'verbose': -1
            })
            model_class = LGBMClassifier

        elif args.model_type == "rf":
            best_params.update({
                'random_state': 42,
                'n_jobs': -1
            })
            model_class = RandomForestClassifier

        model = model_class(**best_params)
        pipeline = Pipeline([
            ('smote', SVMSMOTE(random_state=random_state)),
            ('clf', model)
        ])
        pipeline.fit(X_train_imp, y_train)
        y_pred_prob = pipeline.predict_proba(X_val_imp)[:, 1]
        y_pred = pipeline.predict(X_val_imp)

    # SHAP summary plot for fold 1
    model_type = args.model_type
    if model_type != "tabpfn":
        print(f"Generating SHAP values for fold {outer_fold}...")
        if model_type == "lightgbm":
            explainer = shap.Explainer(model, X_train_imp)
            shap_values = explainer(X_train_imp).values
        elif model_type == "catboost":
            from catboost import Pool
            pool = Pool(X_train_imp, label=y_train, cat_features=None)
            shap_values = model.get_feature_importance(pool, type='ShapValues')
            shap_values = np.array(shap_values)[:, :-1]  # drop expected value column
        elif model_type == "xgboost":
            explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
            shap_values = explainer.shap_values(X_train_imp)
        elif model_type == "rf":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train_imp)[:,:,1]

            shapiq_exp = shapiq.TabularExplainer(model, X_train_imp.values, index="SII", max_order=2)
            # Predict for all training samples
            all_preds = model.predict(X_train_imp)
            # Find indices for class 0 and class 1
            idx_0 = np.where(all_preds == 0)[0]
            idx_1 = np.where(all_preds == 1)[0]
            # Randomly select 4 from each class (if available)
            rng = np.random.default_rng(0)
            sel_0 = rng.choice(idx_0, size=min(4, len(idx_0)), replace=False)
            sel_1 = rng.choice(idx_1, size=min(4, len(idx_1)), replace=False)
            sel_idx = np.concatenate([sel_0, sel_1])
            X_sel = X_train_imp.iloc[sel_idx]
            preds_sel = all_preds[sel_idx]
            # Run shapiq explain for each of the 8 samples individually
            for i, (row_idx, pred) in enumerate(zip(sel_idx, preds_sel)):
                x_instance = X_sel.iloc[[i]]  # keep as DataFrame for shapiq
                interactions = shapiq_exp.explain(x_instance, budget=100, random_state=0)
                shapiq.network_plot(interactions, feature_names=X_train_imp.columns, show=False)
                print(interactions.get_n_order(2))
                # shapiq.plot.upset_plot(interactions.get_n_order(2), feature_names=X_train_imp.columns, show=False)
                plt.title(f"Sample idx: {row_idx}, Prediction: {pred}")
                plt.show()

        plt.figure()
        shap.summary_plot(shap_values, X_train_imp, show=False)
        plt.tight_layout()
        plt.savefig(osp.join(exp_dir, f"{model_type}_{outer_fold}_shap_summary.png"), dpi=300)
        plt.close()

        mean_abs_shap = np.abs(shap_values).mean(axis=0)  # mean absolute SHAP per feature
        importance_df = pd.DataFrame({
            'feature': X_train_imp.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values(by='mean_abs_shap', ascending=False)

        importance_df.to_csv(osp.join(exp_dir, f"{model_type}_{outer_fold}_shap_feature_importance.csv"), index=False)

    # Metrics
    roc_auc = roc_auc_score(y_val, y_pred_prob)
    pr_auc = average_precision_score(y_val, y_pred_prob)
    outer_roc_aucs.append(roc_auc)
    outer_pr_aucs.append(pr_auc)
    print(f"Fold {outer_fold+1}: ROC AUC={roc_auc:.3f}, PR AUC={pr_auc:.3f}")
    per_fold_rows.append({'fold': outer_fold+1, 'roc_auc': roc_auc, 'pr_auc': pr_auc})

# 95% CI calculation
def mean_ci(data):
    m = np.mean(data)
    se_ = sem(data)
    ci = se_ * t.ppf((1 + 0.95) / 2., len(data)-1)
    return m, m-ci, m+ci

roc_mean, roc_low, roc_high = mean_ci(outer_roc_aucs)
pr_mean, pr_low, pr_high = mean_ci(outer_pr_aucs)

print("\n==== Nested CV Results ====")
print(f"ROC AUC: {roc_mean:.3f} (95% CI: {roc_low:.3f}-{roc_high:.3f})")
print(f"PR AUC: {pr_mean:.3f} (95% CI: {pr_low:.3f}-{pr_high:.3f})")

# --- Save experiment log and results ---
run_log['metrics'] = {
    'roc_auc': {'mean': roc_mean, 'ci_low': roc_low, 'ci_high': roc_high},
    'pr_auc': {'mean': pr_mean, 'ci_low': pr_low, 'ci_high': pr_high}
}
run_log['per_fold'] = per_fold_rows
with open(os.path.join(exp_dir, 'log.json'), 'w') as f:
    json.dump(run_log, f, indent=2)
