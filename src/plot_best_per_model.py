import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="Plot best ROC AUC per model from experiment logs.")
parser.add_argument("base_dir", nargs="?", default="experiments", help="Base directory containing experiment folders (default: experiments)")
args = parser.parse_args()
base_dir = args.base_dir

# Structure: results[model_type][k] = max_roc_auc
results = defaultdict(dict)

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    log_file = os.path.join(folder_path, 'log.json')
    if not os.path.isfile(log_file):
        continue

    with open(log_file, 'r') as f:
        log = json.load(f)

    model_type = log['args']['model_type']
    k = log['args']['k']
    roc_auc = log['metrics']['roc_auc']['mean']

    # Store the best ROC AUC for each model_type and k
    if k not in results[model_type] or roc_auc > results[model_type][k]:
        results[model_type][k] = roc_auc

# Plotting
plt.figure(figsize=(10, 6))
for model_type, k_roc_dict in results.items():
    ks = sorted(k_roc_dict.keys())
    roc_aucs = [k_roc_dict[k] for k in ks]
    plt.plot(ks, roc_aucs, marker='o', label=model_type)

plt.xlabel('Number of lipids used')
plt.ylabel('Best ROC AUC')
plt.title('Best ROC AUC vs k for Each Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

