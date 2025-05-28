import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Experiment grid
model_types = ['rf', 'lightgbm', 'catboost', 'xgboost', 'tabpfn']
normalize_options = [True, False]

# Number of parallel jobs (adjust as needed)
MAX_PARALLEL = 4

def build_cmd(model, impute, normalize):
    cmd = [
        sys.executable, "src/predict.py",
        "--model_type", model
    ]
    if impute:
        cmd.append("--impute")
    if normalize:
        cmd.append("--normalize")
    return cmd

def run_experiment(model, impute, normalize):
    cmd = build_cmd(model, impute, normalize)
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}")
        print(result.stderr)
    else:
        print(f"FINISHED: {' '.join(cmd)}")
    return result.returncode

def main():
    combos = []
    for model in model_types:
        if model == 'tabpfn':
            impute_options = [True, False]
        else:
            impute_options = [True]
        for impute, normalize in itertools.product(impute_options, normalize_options):
            combos.append((model, impute, normalize))
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = [
            executor.submit(run_experiment, model, impute, normalize)
            for model, impute, normalize in combos
        ]
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    main() 