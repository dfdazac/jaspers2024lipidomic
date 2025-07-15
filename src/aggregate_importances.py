import argparse
import os
from datetime import datetime
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def aggregate_shap_importances(folders, top_k=None):
    # Load the Excel data for missing value statistics
    excel_path = os.path.join(os.path.dirname(__file__), '../data/SupplementaryData1.xlsx')
    try:
        df_excel = pd.read_excel(excel_path, sheet_name="lipidomics_data_males")
        exclude_cols = [
            'Sample ID',
            'Presence of Cerebral ALD ',
            'Presence of adrenal insufficiency ',
            'Severity of Spinal cord disease '
        ]
        X_excel = df_excel.drop(columns=exclude_cols)
        # Apply the same renaming as in the model pipeline
        X_excel = X_excel.rename(columns=lambda col: col.replace(":", "_"))
        missing_count_per_col = X_excel.isnull().sum()
        missing_percent_per_col = X_excel.isnull().mean() * 100
    except Exception as e:
        print(f"Warning: Could not load Excel data for missing value stats: {e}")
        missing_count_per_col = pd.Series(dtype=float, index=pd.Index([], dtype=str))
        missing_percent_per_col = pd.Series(dtype=float, index=pd.Index([], dtype=str))

    folder_sums = {}
    all_features = set()
    folder_labels = {}

    for folder in folders:
        # Find all *_shap_feature_importance.csv files
        csv_files = glob.glob(os.path.join(folder, '*_shap_feature_importance.csv'))
        if not csv_files:
            continue  # Skip folders with no CSVs
        # Print model_type and k from log.json if present
        log_path = os.path.join(folder, 'log.json')
        label = os.path.basename(folder)
        if os.path.exists(log_path):
            try:
                import json
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                model_type = log_data.get('args', {}).get('model_type', 'N/A')
                k = log_data.get('args', {}).get('k', 'N/A')
                print(f"Loaded {os.path.basename(folder)}: model_type={model_type}, k={k}")
                label = model_type
            except Exception as e:
                print(f"Warning: Could not read log.json in {folder}: {e}")
        folder_sums[label] = {}
        folder_labels[os.path.basename(folder)] = label
        feature_sums = {}
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            print(f"Loaded {os.path.basename(csv_file)} with {len(df)} rows")
            for _, row in df.iterrows():
                feature = row['feature']
                shap = row['mean_abs_shap']
                feature_sums[feature] = feature_sums.get(feature, 0) + shap
                all_features.add(feature)
        folder_sums[label] = feature_sums

    # Build dataframe
    all_features = sorted(all_features)
    df = pd.DataFrame(index=pd.Index(all_features, dtype=str))
    for folder, feature_sums in folder_sums.items():
        df[folder] = [feature_sums.get(f, 0) for f in all_features]

    # Compute overall importance and sort
    df['overall_sum'] = df.sum(axis=1)
    df = df.sort_values('overall_sum', ascending=False)

    # Add missing value statistics
    # Ensure index is string for .get and only call .get on Series
    def safe_get(series, key):
        if isinstance(series, pd.Series):
            return series.get(str(key), np.nan)
        return np.nan
    df['missing_count'] = df.index.map(lambda f: safe_get(missing_count_per_col, f))
    df['missing_percent'] = df.index.map(lambda f: safe_get(missing_percent_per_col, f))
    
    # Select top_k features if specified
    if top_k is not None:
        df_plot = df.head(top_k)
    else:
        df_plot = df
    
    # Define lipid classes and assign colors
    lipid_classes = [
        'PC', 'HexCer', 'Hex2Cer', 'Cer', 'LPC', '1-acyl', 'PE', 'SM', 'TG', '2-acyl', 'LPE', 'SM4', 'DG', 'CE'
    ]
    color_palette = sns.color_palette('tab10', n_colors=len(lipid_classes))
    class_to_color = {cls: color_palette[i] for i, cls in enumerate(lipid_classes)}

    def get_lipid_class(feature):
        for cls in lipid_classes:
            if str(feature).startswith(cls):
                return cls
        return None

    # For the features in the plot, get their classes and colors
    if top_k is not None:
        features_in_plot = df.head(top_k).index.tolist()
    else:
        features_in_plot = df.index.tolist()
    feature_classes = [get_lipid_class(f) for f in features_in_plot]
    feature_colors = [class_to_color[cls] if isinstance(cls, str) and cls in class_to_color else (0,0,0) for cls in feature_classes]

    # Plot two separate heatmaps: one for SHAP values, one for missing_count
    plot_cols = [col for col in df_plot.columns if col not in ['overall_sum', 'missing_count', 'missing_percent']]
    shap_data = df_plot[plot_cols]
    missing_data = df_plot[['missing_count']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, len(folders)*1.5)+4, max(8, len(df_plot)/3)))

    # SHAP heatmap
    sns.heatmap(shap_data, cmap='binary', annot=False, ax=ax1, cbar=True)
    ax1.set_title('Aggregated SHAP Feature Importances')
    ax1.set_xlabel('Model/Folders')
    ax1.set_ylabel('Feature')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    for ticklabel, color in zip(ax1.get_yticklabels(), feature_colors):
        ticklabel.set_color(color)
    legend_elements = [Patch(facecolor=class_to_color[cls], label=cls) for cls in lipid_classes]
    ax1.legend(handles=legend_elements, title="Lipid Class", bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=3, frameon=False)

    # Missing count heatmap
    sns.heatmap(missing_data, cmap='Reds', annot=True, fmt='.0f', ax=ax2, cbar=True)
    ax2.set_title('Missing Value Count')
    ax2.set_xlabel('Missing Count')
    ax2.set_ylabel('')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    # Color ytick labels by lipid class (same as ax1)
    for ticklabel, color in zip(ax2.get_yticklabels(), feature_colors):
        ticklabel.set_color(color)
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_yticklabels(ax1.get_yticklabels())

    plt.tight_layout()
    plt.show()

    return df


def main():
    parser = argparse.ArgumentParser(description='Aggregate SHAP feature importances from multiple folders.')
    parser.add_argument('folders', nargs='+', help='Folders containing *_shap_feature_importance.csv files')
    parser.add_argument('--top-k', type=int, default=None, help='Number of top features to plot in the heatmap (by overall importance)')
    args = parser.parse_args()
    
    df = aggregate_shap_importances(args.folders, top_k=args.top_k)
    
    # Create output/YYYY-mm-dd-HHmmss directory
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    output_dir = os.path.join('output', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save log.txt with all command line arguments
    log_path = os.path.join(output_dir, 'log.txt')
    with open(log_path, 'w') as f:
        f.write("Command line arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Save df[['overall_sum']] as aggregated_importances.csv
    agg_path = os.path.join(output_dir, 'aggregated_importances.csv')
    df[['overall_sum']].to_csv(agg_path)

if __name__ == '__main__':
    main() 