Analysis of the data presented in [Jaspers, Yorrick RJ, et al. "Lipidomic biomarkers in plasma correlate with disease severity in adrenoleukodystrophy." Communications Medicine 4.1 (2024): 175.](https://www.nature.com/articles/s43856-024-00605-9).

## Overview

This repository provides scripts for predicting the presence of adrenal insufficiency from lipidomics data using a variety of machine learning models. It includes:
- Feature selection based on statistical tests (p-value and Bonferroni correction)
- Support for multiple classifiers: Random Forest, LightGBM, CatBoost, XGBoost, and TabPFN
- Nested cross-validation with deduplication by patient
- Optional KNN imputation (enabled by default, can be disabled with a flag)
- SHAP feature importance and summary plots for model interpretability (for all supported models)

## Main Script

The main script is `src/predict.py`. It loads the data, performs feature selection, runs nested cross-validation, and outputs performance metrics and SHAP feature importances.

### Usage

```bash
python src/predict.py \
    --model_type {rf,lightgbm,catboost,xgboost,tabpfn} \
    [--pvalue_filter 0.05] \
    [--k 100] \
    [--no_imputation]
```

- `--model_type`: Choose the model to use (default: tabpfn)
- `--pvalue_filter`: p-value threshold for feature selection (default: 0.05)
- `--k`: Number of top features to select (default: 100)
- `--no_imputation`: Disable KNN imputation (enabled by default)

### Outputs
- Prints nested CV metrics (ROC AUC, PR AUC)
- Saves SHAP summary plot for the first fold as `{model_type}_shap_summary.png`
- Saves SHAP feature importances as `{model_type}_shap_feature_importance.csv`

## Dependencies
- pandas, numpy, scikit-learn, imbalanced-learn, optuna
- catboost, lightgbm, xgboost, tabpfn
- shap, matplotlib

Install dependencies with pip:
```bash
pip install pandas numpy scikit-learn imbalanced-learn optuna catboost lightgbm xgboost tabpfn shap matplotlib
```

## Data
- Expects input Excel files in the `data/` directory as described in the paper.

---

For questions or issues, please refer to the original publication or open an issue in this repository.

