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
# --- BOX PLOT OF LIPID VALUES ---
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
    box_colors = ['#888888']  # gray for 'All'
    # Identify Control group (patients who are Control in any clinical variable)
    control_mask = (main_df[clinical_cols] == 'Control').any(axis=1)
    control_group = main_df[control_mask][lipid]
    if not isinstance(control_group, pd.Series):
        control_group = pd.Series(control_group)
    control_group = control_group.dropna()
    if len(control_group) > 0:
        box_data.append(control_group)
        labels.append('Control')
        box_colors.append('#9467bd')  # purple for Control
    # For each clinical column, add boxplots for each group except 'Control'
    color_map = {
        'Presence of Cerebral ALD ': '#1f77b4',           # blue
        'Presence of adrenal insufficiency ': '#2ca02c',  # green
        'Severity of Spinal cord disease ': '#ff7f0e'     # orange
    }
    for col in clinical_cols:
        if col in main_df.columns:
            for val in main_df[col].dropna().unique():
                if val == 'Control':
                    continue  # already handled
                group = main_df[(main_df[col] == val) & (~control_mask)][lipid]
                if not isinstance(group, pd.Series):
                    group = pd.Series(group)
                group = group.dropna()
                if len(group) > 0:
                    box_data.append(group)
                    labels.append(f"{col.strip()}={val}")
                    box_colors.append(color_map[col])
    # Boxplot
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False)
    # Set box colors
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    # Min/max markers
    for j, group in enumerate(box_data):
        group = group.dropna()
        if len(group) > 0:
            ax.plot([j+1], [np.min(group)], marker='v', color='red', markersize=5)
            ax.plot([j+1], [np.max(group)], marker='^', color='blue', markersize=5)
    ax.set_title(lipid, fontsize=8)
    # Show x labels only on bottom row
    if i // 10 == 9:
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    # Show y label for all plots
    ax.set_ylabel('Lipid Value', fontsize=8)

# Hide any unused subplots
for j in range(len(top_lipids), 100):
    axes[j].axis('off')

fig.suptitle('Top 100 Lipids: Value Distributions (10x10 grid)', fontsize=18)
fig.text(0.5, 0.04, 'Group', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Lipid Value', va='center', rotation='vertical', fontsize=14)
plt.savefig(output_pdf, bbox_inches='tight')
plt.close(fig)
print(f"Saved 10x10 boxplot grid to {output_pdf}")

# --- BARPLOT OF MISSING FRACTION PER GROUP ---
output_pdf_missing = os.path.join(os.path.dirname(__file__), '../output/lipid_missing_10x10.pdf')
fig2, axes2 = plt.subplots(10, 10, figsize=(30, 30), constrained_layout=True)
axes2 = axes2.flatten()

for i, lipid in enumerate(top_lipids):
    ax = axes2[i]
    if lipid not in main_df.columns:
        ax.set_title(f"{lipid}\n(N/A)")
        ax.axis('off')
        continue
    data = main_df[lipid]
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    group_fracs = []
    labels = ['All']
    bar_colors = ['#888888']
    # All
    frac_missing_all = data.isna().mean()
    group_fracs.append(frac_missing_all)
    # Control group
    control_mask = (main_df[clinical_cols] == 'Control').any(axis=1)
    control_group = main_df[control_mask][lipid]
    if not isinstance(control_group, pd.Series):
        control_group = pd.Series(control_group)
    frac_missing_control = control_group.isna().mean() if len(control_group) > 0 else np.nan
    if not np.isnan(frac_missing_control):
        group_fracs.append(frac_missing_control)
        labels.append('Control')
        bar_colors.append('#9467bd')
    # For each clinical column, add missing fraction for each group except 'Control'
    for col in clinical_cols:
        if col in main_df.columns:
            for val in main_df[col].dropna().unique():
                if val == 'Control':
                    continue  # already handled
                group = main_df[(main_df[col] == val) & (~control_mask)][lipid]
                if not isinstance(group, pd.Series):
                    group = pd.Series(group)
                frac_missing = group.isna().mean() if len(group) > 0 else np.nan
                group_fracs.append(frac_missing)
                labels.append(f"{col.strip()}={val}")
                bar_colors.append(color_map[col])
    # Bar plot
    ax.bar(range(len(group_fracs)), group_fracs, color=bar_colors)
    ax.set_ylim(0, 1)
    ax.set_title(lipid, fontsize=8)
    # Show x labels only on bottom row
    if i // 10 == 9:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    # Show y label for all plots
    ax.set_ylabel('Fraction Missing', fontsize=8)

# Hide any unused subplots
for j in range(len(top_lipids), 100):
    axes2[j].axis('off')

fig2.suptitle('Top 100 Lipids: Fraction of Missing Values (10x10 grid)', fontsize=18)
fig2.text(0.5, 0.04, 'Group', ha='center', fontsize=14)
fig2.text(0.04, 0.5, 'Fraction Missing', va='center', rotation='vertical', fontsize=14)
plt.savefig(output_pdf_missing, bbox_inches='tight')
plt.close(fig2)
print(f"Saved 10x10 missing fraction barplot grid to {output_pdf_missing}")
