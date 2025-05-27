import pandas as pd
import re

# Load lipid names from SupplementaryData2.xlsx
stat_path = 'data/SupplementaryData2.xlsx'
stat_df = pd.read_excel(stat_path, header=[0, 1])
lipids_supp2 = stat_df[stat_df.columns[0]].astype(str).str.strip().unique()

# Load lipid columns from SupplementaryData1.xlsx
main_path = 'data/SupplementaryData1.xlsx'
df = pd.read_excel(main_path, sheet_name="lipidomics_data_males")
supp1_cols = set(df.columns.astype(str))

# Improved function: handle dot or space before last part
def convert_lipid_name(name):
    if '.' in name:
        prefix, suffix = name.rsplit('.', 1)
        return f"{prefix}({suffix})"
    elif ' ' in name:
        prefix, suffix = name.rsplit(' ', 1)
        return f"{prefix}({suffix})"
    return name

# Build mapping for all lipids in supp2
mapping = {}
for lipid in stat_df[stat_df.columns[0]].astype(str).str.strip():
    converted_lipid = convert_lipid_name(lipid)
    if converted_lipid in supp1_cols:
        mapping[lipid] = converted_lipid
    else:
        raise ValueError(f"{converted_lipid} not in SupplementaryData1")

# Replace the first column in stat_df with the converted names
stat_df_matched = stat_df.copy()
first_col = stat_df.columns[0]
stat_df_matched[first_col] = stat_df_matched[first_col].astype(str).str.strip().map(mapping)

# Flatten MultiIndex columns before saving
stat_df_matched.columns = [' '.join([str(i) for i in col if str(i) != 'nan']).strip() for col in stat_df_matched.columns.values]

# Save the updated DataFrame
output_path = 'data/SupplementaryData2-matched-lipids.xlsx'
stat_df_matched.to_excel(output_path, index=False)
print(f"Saved matched lipid names to {output_path}")