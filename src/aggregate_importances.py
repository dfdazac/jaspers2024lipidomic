import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def aggregate_shap_importances(folders, top_k=None):
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
            for _, row in df.iterrows():
                feature = row['feature']
                shap = row['mean_abs_shap']
                feature_sums[feature] = feature_sums.get(feature, 0) + shap
                all_features.add(feature)
        folder_sums[label] = feature_sums

    # Build dataframe
    all_features = sorted(all_features)
    df = pd.DataFrame(index=all_features)
    for folder, feature_sums in folder_sums.items():
        df[folder] = [feature_sums.get(f, 0) for f in all_features]

    # Compute overall importance and sort
    df['overall_sum'] = df.sum(axis=1)
    df = df.sort_values('overall_sum', ascending=False)
    
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
            if feature.startswith(cls):
                return cls
        return None

    # For the features in the plot, get their classes and colors
    if top_k is not None:
        features_in_plot = df.head(top_k).index.tolist()
    else:
        features_in_plot = df.index.tolist()
    feature_classes = [get_lipid_class(f) for f in features_in_plot]
    feature_colors = [class_to_color.get(cls, (0,0,0)) for cls in feature_classes]

    # Plot heatmap (excluding overall_sum column)
    plt.figure(figsize=(max(8, len(folders)*1.5), max(8, len(df_plot)/3)))
    ax = sns.heatmap(df_plot.drop(columns=['overall_sum']), cmap='binary', annot=False)
    plt.title('Aggregated SHAP Feature Importances')
    plt.xlabel('Model/Folders')
    plt.ylabel('Feature')
    plt.xticks(rotation=45, ha='right')

    # Color ytick labels by lipid class
    for ticklabel, color in zip(ax.get_yticklabels(), feature_colors):
        ticklabel.set_color(color)
    
    # Add a vertical legend for lipid classes
    legend_elements = [Patch(facecolor=class_to_color[cls], label=cls) for cls in lipid_classes]
    ax.legend(handles=legend_elements, title="Lipid Class", bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=3, frameon=False)
    plt.tight_layout()
    plt.show()

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate SHAP feature importances from multiple folders.')
    parser.add_argument('folders', nargs='+', help='Folders containing *_shap_feature_importance.csv files')
    parser.add_argument('--top-k', type=int, default=None, help='Number of top features to plot in the heatmap (by overall importance)')
    args = parser.parse_args()
    
    df = aggregate_shap_importances(args.folders, top_k=args.top_k)
    print(df.head(args.top_k))

if __name__ == '__main__':
    main() 