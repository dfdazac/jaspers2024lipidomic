import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from tqdm import tqdm

# Experiment grid
k_values = [20, 50, 100, 150, 200]
model_types = ['rf', 'lightgbm', 'catboost', 'xgboost', 'tabpfn']
normalize_options = [True, False]
imputer_options = ['knn', 'min5']

MAX_PARALLEL = 4

def build_cmd(model, normalize, k, imputer):
    cmd = [
        sys.executable, "src/predict.py",
        "--model_type", model,
        "--k", str(k),
        "--imputer", imputer,
        "--exclude_controls",
        "--vlcfas_only",
    ]
    if normalize:
        cmd.append("--normalize")
    return cmd

def run_experiment(model, normalize, k, imputer):
    cmd = build_cmd(model, normalize, k, imputer)
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}")
        print(result.stderr)
    else:
        print(f"FINISHED: {' '.join(cmd)}")
    return result.returncode

def main():
    combos = list(itertools.product(model_types, normalize_options, k_values, imputer_options))
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = [
            executor.submit(run_experiment, model, normalize, k, imputer)
            for model, normalize, k, imputer in combos
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Experiments"):
            _.result()

if __name__ == "__main__":
    main() 