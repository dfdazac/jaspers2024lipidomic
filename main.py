import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline
import optuna
from scipy.stats import sem, t
from sklearn.impute import KNNImputer

# Load data
file_path = 'SupplementaryData1.xlsx'
df = pd.read_excel(file_path, sheet_name="lipidomics_data_males")

# Identify lipid columns (from '1-acyl LPC(14:1)' onwards)
lipid_start = list(df.columns).index('1-acyl LPC(14:1)')
lipid_cols = df.columns[lipid_start:]

# Prepare features and target
y_raw = df['Presence of adrenal insufficiency '].astype(str).str.strip()
y = (y_raw == 'AI').astype(int)
X = df[lipid_cols]

# Encode target (no longer needed, but keep for compatibility)
# le = LabelEncoder()
# y_enc = le.fit_transform(y)
y_enc = y.values

# Patient ID for deduplication
patient_id_col = 'Sample ID'

# Nested CV parameters
outer_k = 10
inner_k = 10
random_state = 42

# Store results
outer_roc_aucs = []
outer_pr_aucs = []
outer_f4s = []

# Outer CV
outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=random_state)
for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_enc)):
    # Deduplicate validation set by patient
    val_patients = df.iloc[test_idx][patient_id_col].drop_duplicates()
    val_mask = df[patient_id_col].isin(val_patients)
    val_idx = df[val_mask].index.intersection(test_idx)
    # Remove these patients from training set
    train_mask = ~df[patient_id_col].isin(val_patients)
    train_idx = df[train_mask].index.intersection(train_idx)

    X_train, y_train = X.iloc[train_idx], y_enc[train_idx]
    X_val, y_val = X.iloc[val_idx], y_enc[val_idx]

    # KNN imputation fit on training data, applied to both train and val
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)

    # Inner CV for hyperparameter tuning and feature selection
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=random_state)
        scores = []
        for inner_train_idx, inner_test_idx in inner_cv.split(X_train_imp, y_train):
            X_inner_train, y_inner_train = X_train_imp.iloc[inner_train_idx], y_train[inner_train_idx]
            X_inner_test, y_inner_test = X_train_imp.iloc[inner_test_idx], y_train[inner_test_idx]
            # Pipeline: SVMSMOTE -> RF
            pipeline = Pipeline([
                ('smote', SVMSMOTE(random_state=random_state)),
                ('clf', RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    n_jobs=-1
                ))
            ])
            pipeline.fit(X_inner_train, y_inner_train)
            rf = pipeline.named_steps['clf']
            feature_importances = rf.feature_importances_
            threshold = np.percentile(feature_importances, 30)
            selected_features = X_inner_train.columns[feature_importances > threshold]
            # Refit using only selected features
            pipeline.fit(X_inner_train[selected_features], y_inner_train)
            y_pred_prob = pipeline.predict_proba(X_inner_test[selected_features])[:, 1]
            score = average_precision_score(y_inner_test, y_pred_prob)
            scores.append(score)
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    best_params = study.best_params

    # Feature selection on full training set with best params
    pipeline = Pipeline([
        ('smote', SVMSMOTE(random_state=random_state)),
        ('clf', RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train_imp, y_train)
    rf = pipeline.named_steps['clf']
    feature_importances = rf.feature_importances_
    threshold = np.percentile(feature_importances, 30)
    selected_features = X_train_imp.columns[feature_importances > threshold]
    # Refit using only selected features
    pipeline.fit(X_train_imp[selected_features], y_train)
    y_pred_prob = pipeline.predict_proba(X_val_imp[selected_features])[:, 1]
    y_pred = pipeline.predict(X_val_imp[selected_features])

    # Metrics
    roc_auc = roc_auc_score(y_val, y_pred_prob)
    pr_auc = average_precision_score(y_val, y_pred_prob)
    f4 = fbeta_score(y_val, y_pred, beta=4)
    outer_roc_aucs.append(roc_auc)
    outer_pr_aucs.append(pr_auc)
    outer_f4s.append(f4)
    print(f"Fold {outer_fold+1}: ROC AUC={roc_auc:.3f}, PR AUC={pr_auc:.3f}, F4={f4:.3f}")

# 95% CI calculation
def mean_ci(data):
    m = np.mean(data)
    se_ = sem(data)
    ci = se_ * t.ppf((1 + 0.95) / 2., len(data)-1)
    return m, m-ci, m+ci

roc_mean, roc_low, roc_high = mean_ci(outer_roc_aucs)
pr_mean, pr_low, pr_high = mean_ci(outer_pr_aucs)
f4_mean, f4_low, f4_high = mean_ci(outer_f4s)

print("\n==== Nested CV Results ====")
print(f"ROC AUC: {roc_mean:.3f} (95% CI: {roc_low:.3f}-{roc_high:.3f})")
print(f"PR AUC: {pr_mean:.3f} (95% CI: {pr_low:.3f}-{pr_high:.3f})")
print(f"F4-score: {f4_mean:.3f} (95% CI: {f4_low:.3f}-{f4_high:.3f})") 