import os
import json

import argparse

parser = argparse.ArgumentParser(description="Plot best ROC AUC per model from experiment logs.")
parser.add_argument("base_dir", nargs="?", default=".", help="Base directory containing experiment folders (default: current directory)")
args = parser.parse_args()
base_dir = args.base_dir

best_results = {}

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)

    if not os.path.isdir(folder_path) or not folder.startswith("2025-06"):
        continue

    log_file = os.path.join(folder_path, 'log.json')
    if not os.path.isfile(log_file):
        continue

    with open(log_file, 'r') as f:
        log = json.load(f)

    model_type = log['args']['model_type']
    roc_auc = log['metrics']['roc_auc']['mean']
    pr_auc = log['metrics']['pr_auc']['mean']

    if model_type not in best_results or roc_auc > best_results[model_type]['roc_auc']:
        best_results[model_type] = {
            'roc_auc': roc_auc,
            'roc_ci_low': log['metrics']['roc_auc']['ci_low'],
            'roc_ci_high': log['metrics']['roc_auc']['ci_high'],
            'pr_auc': pr_auc,
            'pr_ci_low': log['metrics']['pr_auc']['ci_low'],
            'pr_ci_high': log['metrics']['pr_auc']['ci_high'],
            'folder': folder
        }

# Print the results
for model_type, metrics in sorted(best_results.items(), key=lambda x: x[1]['roc_auc']):
    print(f"{model_type}:")
    print(f"  ROC AUC = {metrics['roc_auc']:.2%} "
          f"[{metrics['roc_ci_low']:.2%}, {metrics['roc_ci_high']:.2%}]")
    print(f"  PR AUC  = {metrics['pr_auc']:.2%} "
          f"[{metrics['pr_ci_low']:.2%}, {metrics['pr_ci_high']:.2%}]")
    print(f"  From folder: {metrics['folder']}\n")
