import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths
excel_path = os.path.join(os.path.dirname(__file__), '../data/SupplementaryData1.xlsx')
shap_path = os.path.join(os.path.dirname(__file__), '../output/2025-07-15-113125/aggregated_importances.csv')
output_pdf = os.path.join(os.path.dirname(__file__), '../output/lipid_boxplots_10x10.pdf')

# Clinical columns (with trailing spaces)
clinical_cols = [
    'Presence of Cerebral ALD ',
    'Presence of adrenal insufficiency ',
    'Severity of Spinal cord disease '
]

# Load top 100 lipids from SHAP importances
shap_df = pd.read_csv(shap_path, index_col=0)
top_lipids = shap_df.head(100).index.tolist()

# Load main data
main_df = pd.read_excel(excel_path, sheet_name="lipidomics_data_males")
main_df = main_df.rename(columns=lambda col: col.replace(":", "_"))

# Prepare figure
fig, axes = plt.subplots(10, 10, figsize=(30, 30), constrained_layout=True)
axes = axes.flatten()

for i, lipid in enumerate(top_lipids):
    ax = axes[i]
    if lipid not in main_df.columns:
        ax.set_title(f"{lipid}\n(N/A)")
        ax.axis('off')
        continue
    data = main_df[lipid]
    box_data = [data.dropna()]
    labels = ['All']
    # For each clinical column, add boxplots for each group
    for col in clinical_cols:
        if col in main_df.columns:
            for val in main_df[col].dropna().unique():
                group = main_df[main_df[col] == val][lipid]
                if not isinstance(group, pd.Series):
                    group = pd.Series(group)
                group = group.dropna()
                if len(group) > 0:
                    box_data.append(group)
                    labels.append(f"{col.strip()}={val}")
    # Boxplot
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False)
    # Min/max markers
    for j, group in enumerate(box_data):
        group = group.dropna()
        if len(group) > 0:
            ax.plot([j+1], [np.min(group)], marker='v', color='red', markersize=5)
            ax.plot([j+1], [np.max(group)], marker='^', color='blue', markersize=5)
    ax.set_title(lipid, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

# Hide any unused subplots
for j in range(len(top_lipids), 100):
    axes[j].axis('off')

fig.suptitle('Top 100 Lipids: Value Distributions (10x10 grid)', fontsize=18)
plt.savefig(output_pdf, bbox_inches='tight')
plt.close(fig)
print(f"Saved 10x10 boxplot grid to {output_pdf}")
