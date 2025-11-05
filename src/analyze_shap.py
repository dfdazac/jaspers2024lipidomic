import argparse
import json
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate instance-level SHAP results to analyze age-stratified patterns "
            "and lipid interaction groups without re-running training."
        )
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="experiments",
        help=(
            "Directory containing experiment folders OR a specific experiment folder "
            "that itself contains instance_shap_table.csv and log.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="output/shap_analysis",
        help="Directory to write analysis outputs (CSVs/PNGs)",
    )
    parser.add_argument(
        "--model_types",
        nargs="*",
        default=None,
        help="Optional list of model types to include (e.g., lightgbm rf catboost)",
    )
    parser.add_argument("--k", type=int, default=None, help="Filter experiments by k if provided")
    parser.add_argument(
        "--normalize",
        choices=["true", "false"],
        default=None,
        help="Filter by normalize flag if provided",
    )
    parser.add_argument(
        "--imputer",
        choices=["knn", "min5"],
        default=None,
        help="Filter by imputer if provided",
    )
    parser.add_argument(
        "--exclude_controls",
        choices=["true", "false"],
        default=None,
        help="Filter by exclude_controls flag if provided",
    )
    parser.add_argument(
        "--vlcfas_only",
        choices=["true", "false"],
        default=None,
        help="Filter by vlcfas_only flag if provided",
    )
    parser.add_argument(
        "--age_bins",
        type=int,
        default=3,
        help="Number of quantile age bins for stratified analysis",
    )
    parser.add_argument(
        "--top_lipids_for_clustering",
        type=int,
        default=150,
        help=(
            "Limit clustering to top N lipids ranked by mean absolute SHAP across all instances"
        ),
    )
    parser.add_argument(
        "--corr_method",
        choices=["spearman", "pearson"],
        default="spearman",
        help="Correlation method for SHAP co-variation",
    )
    parser.add_argument(
        "--cluster_distance_threshold",
        type=float,
        default=0.6,
        help=(
            "Hierarchical clustering threshold on distance (1 - |corr|). Lower -> more clusters"
        ),
    )
    parser.add_argument(
        "--min_presence",
        type=int,
        default=10,
        help=(
            "Minimum number of non-NaN SHAP values required per lipid to include in analyses"
        ),
    )
    parser.add_argument(
        "--save_aggregated_csv",
        action="store_true",
        help="If set, save the full aggregated long-form table",
    )
    parser.add_argument(
        "--year_prefix",
        default=None,
        help="Optional folder name prefix filter (e.g., 2025)",
    )
    return parser.parse_args()


def _flag_to_bool_or_none(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return value.lower() == "true"


def discover_and_load(
    base_dir: str,
    model_types: Optional[List[str]],
    k: Optional[int],
    normalize: Optional[bool],
    imputer: Optional[str],
    exclude_controls: Optional[bool],
    vlcfas_only: Optional[bool],
    year_prefix: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[pd.DataFrame] = []
    used_folders: List[str] = []

    if not osp.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # If base_dir itself is a single experiment folder, load it directly and ignore filters
    shap_csv_direct = osp.join(base_dir, "instance_shap_table.csv")
    log_json_direct = osp.join(base_dir, "log.json")
    if osp.isfile(shap_csv_direct) and osp.isfile(log_json_direct):
        with open(log_json_direct, "r") as f:
            log = json.load(f)
        args = log.get("args", {})
        df = pd.read_csv(shap_csv_direct)
        df["model_type"] = args.get("model_type")
        df["k"] = args.get("k")
        df["normalize"] = bool(args.get("normalize"))
        df["imputer"] = args.get("imputer")
        df["exclude_controls"] = bool(args.get("exclude_controls"))
        df["vlcfas_only"] = bool(args.get("vlcfas_only"))
        df["exp_folder"] = osp.basename(base_dir.rstrip(os.sep))
        return df, [osp.basename(base_dir.rstrip(os.sep))]

    for folder in sorted(os.listdir(base_dir)):
        folder_path = osp.join(base_dir, folder)
        if not osp.isdir(folder_path):
            continue
        if year_prefix is not None and not folder.startswith(year_prefix):
            continue

        shap_csv = osp.join(folder_path, "instance_shap_table.csv")
        log_json = osp.join(folder_path, "log.json")
        if not (osp.isfile(shap_csv) and osp.isfile(log_json)):
            continue

        with open(log_json, "r") as f:
            log = json.load(f)
        args = log.get("args", {})

        if model_types is not None and args.get("model_type") not in model_types:
            continue
        if k is not None and args.get("k") != k:
            continue
        if normalize is not None and bool(args.get("normalize")) != normalize:
            continue
        if imputer is not None and args.get("imputer") != imputer:
            continue
        if exclude_controls is not None and bool(args.get("exclude_controls")) != exclude_controls:
            continue
        if vlcfas_only is not None and bool(args.get("vlcfas_only")) != vlcfas_only:
            continue

        df = pd.read_csv(shap_csv)

        # Attach metadata columns
        df["model_type"] = args.get("model_type")
        df["k"] = args.get("k")
        df["normalize"] = bool(args.get("normalize"))
        df["imputer"] = args.get("imputer")
        df["exclude_controls"] = bool(args.get("exclude_controls"))
        df["vlcfas_only"] = bool(args.get("vlcfas_only"))
        df["exp_folder"] = folder

        rows.append(df)
        used_folders.append(folder)

    if not rows:
        raise RuntimeError(
            "No matching experiments with instance_shap_table.csv found; adjust filters."
        )

    merged = pd.concat(rows, axis=0, ignore_index=True)
    return merged, used_folders


def melt_lipid_shap(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    meta_cols = {
        "k",
        "model_type",
        "normalize",
        "imputer",
        "fold",
        "sample_id",
        "age",
        "true_adrenal_insufficiency",
        "pred_adrenal_insufficiency",
        "exclude_controls",
        "vlcfas_only",
        "exp_folder",
    }
    lipid_cols = [c for c in df.columns if c not in meta_cols]
    long_df = df.melt(
        id_vars=[
            "k",
            "model_type",
            "normalize",
            "imputer",
            "fold",
            "sample_id",
            "age",
            "true_adrenal_insufficiency",
            "pred_adrenal_insufficiency",
            "exclude_controls",
            "vlcfas_only",
            "exp_folder",
        ],
        value_vars=lipid_cols,
        var_name="lipid",
        value_name="shap_value",
    )
    return long_df, lipid_cols


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    m = p_values.shape[0]
    order = np.argsort(p_values.values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q_values = p_values.values * m / np.maximum(ranks, 1)
    # Ensure monotonicity
    q_values_sorted = np.minimum.accumulate(q_values[order][::-1])[::-1]
    q_values_final = np.empty_like(q_values_sorted)
    q_values_final[order] = q_values_sorted
    q_series = pd.Series(q_values_final, index=p_values.index)
    return q_series.clip(upper=1.0)


def age_stratified_analysis(
    long_df: pd.DataFrame,
    output_dir: str,
    age_bins: int,
    min_presence: int,
) -> None:
    df = long_df.copy()
    df = df[~df["age"].isna()]

    # Create quantile age bins
    df["age_bin"], bin_edges = pd.qcut(df["age"], q=age_bins, retbins=True, duplicates="drop")

    # Compute mean absolute SHAP per lipid per age_bin
    df["abs_shap"] = df["shap_value"].abs()

    # Only keep lipids with sufficient presence
    lipid_presence = df.groupby("lipid")["abs_shap"].apply(lambda s: s.notna().sum())
    eligible_lipids = lipid_presence[lipid_presence >= min_presence].index
    df = df[df["lipid"].isin(eligible_lipids)]

    grouped = (
        df.groupby(["lipid", "age_bin"], observed=False)  # type: ignore
        ["abs_shap"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot(index="lipid", columns="age_bin", values="abs_shap").fillna(0.0)
    pivot.to_csv(osp.join(output_dir, "age_stratified_mean_abs_shap.csv"))

    # Statistical test: Kruskal-Wallis across bins per lipid
    kw_rows: List[Dict[str, object]] = []
    for lipid, sub in df.groupby("lipid", observed=False):
        groups: List[np.ndarray] = []
        for age_bin, sub_bin in sub.groupby("age_bin", observed=False):
            vals = sub_bin["abs_shap"].dropna().values
            if vals.size > 0:
                groups.append(vals)
        # Require at least two bins with data
        if len(groups) < 2:
            continue
        # Skip if all values across bins are identical (SciPy raises in this case)
        all_vals = np.concatenate(groups)
        if all_vals.size == 0 or np.nanstd(all_vals) == 0.0:
            continue
        stat, p = kruskal(*groups)
        kw_rows.append({"lipid": lipid, "kw_stat": stat, "p_value": p})
    kw_df = pd.DataFrame(kw_rows).sort_values("p_value")
    if not kw_df.empty:
        kw_df["q_value"] = benjamini_hochberg(kw_df["p_value"]).values
        kw_df.to_csv(osp.join(output_dir, "age_stratified_kw_results.csv"), index=False)

    # Plot: For top 20 lipids with strongest variation (lowest q or highest range)
    top_candidates: List[str]
    if not kw_df.empty:
        top_candidates = kw_df.nsmallest(10, "p_value")["lipid"].tolist()
    else:
        # fallback: pick by range across bins
        ranges = pivot.max(axis=1) - pivot.min(axis=1)
        top_candidates = ranges.nlargest(10).index.tolist()

    to_plot = pivot.loc[pivot.index.intersection(top_candidates)]
    if not to_plot.empty:
        plt.figure(figsize=(12, 8))
        for idx, (lipid, row) in enumerate(to_plot.iterrows()):
            plt.plot(range(len(row)), row.values, marker="o", label=lipid)
        plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45, ha="right")
        plt.ylabel("Mean |SHAP|")
        plt.title("Age-stratified mean |SHAP| for top lipids")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(osp.join(output_dir, "age_stratified_top_lipids.png"), dpi=300)
        plt.close()


def shap_covariance_and_clustering(
    long_df: pd.DataFrame,
    output_dir: str,
    corr_method: str,
    cluster_distance_threshold: float,
    min_presence: int,
    top_lipids_for_clustering: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Build wide matrix: rows = instances (sample_id+exp_folder+fold), columns = lipids, values = SHAP (signed)
    instance_id = long_df["exp_folder"].astype(str) + "|f" + long_df["fold"].astype(str) + "|" + long_df["sample_id"].astype(str)
    matrix = long_df.pivot_table(index=instance_id, columns="lipid", values="shap_value", aggfunc="mean")

    # Filter lipids by presence
    presence = matrix.notna().sum(axis=0)
    eligible = presence[presence >= min_presence].index
    matrix = matrix[eligible]

    # Rank lipids by mean absolute SHAP
    mean_abs = matrix.abs().mean(axis=0).sort_values(ascending=False)
    selected = mean_abs.head(top_lipids_for_clustering).index
    matrix = matrix[selected]

    # Compute correlation with pairwise complete observations
    if corr_method == "spearman":
        corr = matrix.corr(method="spearman", min_periods=5)
    else:
        corr = matrix.corr(method="pearson", min_periods=5)
    corr = corr.fillna(0.0)
    corr.to_csv(osp.join(output_dir, "shap_correlation_matrix.csv"))

    # Distance for clustering based on absolute correlation
    abs_corr = corr.abs()
    np.fill_diagonal(abs_corr.values, 1.0)
    dist = 1.0 - abs_corr
    # squareform expects condensed distance; ensure symmetry
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="average")

    # Assign clusters
    cluster_labels = fcluster(Z, t=cluster_distance_threshold, criterion="distance")
    clusters_df = pd.DataFrame({
        "lipid": corr.index,
        "cluster": cluster_labels,
        "mean_abs_shap": mean_abs.loc[corr.index].values,
    })
    clusters_df_sorted = clusters_df.sort_values("mean_abs_shap", ascending=False)
    clusters_df_sorted.to_csv(osp.join(output_dir, "lipid_clusters.csv"), index=False)

    # Heatmap ordered by dendrogram leaves
    dendro = dendrogram(Z, no_plot=True, labels=corr.index.tolist())
    order = dendro["leaves"]
    corr_ord = corr.values[order][:, order]
    labels_ord = corr.index[order]

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_ord, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels_ord)), labels_ord, rotation=90, fontsize=6)
    plt.yticks(range(len(labels_ord)), labels_ord, fontsize=6)
    plt.title("SHAP co-variation (correlation) heatmap")
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "shap_corr_heatmap.png"), dpi=300)
    plt.close()

    # Dendrogram plot (horizontal for readability)
    plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=corr.index.tolist(), orientation="right")
    plt.axvline(cluster_distance_threshold, color="gray", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "dendrogram.pdf"), dpi=300)
    plt.close()

    return matrix, corr, clusters_df


def module_scores_and_associations(
    matrix: pd.DataFrame,
    long_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    output_dir: str,
) -> None:
    # Build module score per instance: sum of |SHAP| within each cluster
    lipid_to_cluster: Dict[str, int] = (
        clusters_df.set_index("lipid")["cluster"].to_dict()
    )
    cluster_ids = sorted(set(lipid_to_cluster.values()))

    module_scores = pd.DataFrame(index=matrix.index)
    for cluster_id in cluster_ids:
        lipids_in_cluster = [l for l, c in lipid_to_cluster.items() if c == cluster_id and l in matrix.columns]
        if not lipids_in_cluster:
            continue
        module_scores[f"cluster_{cluster_id}"] = matrix[lipids_in_cluster].abs().sum(axis=1)

    # Attach metadata per instance
    meta_cols = [
        "exp_folder",
        "fold",
        "sample_id",
        "age",
        "true_adrenal_insufficiency",
        "pred_adrenal_insufficiency",
    ]
    instance_id = long_df["exp_folder"].astype(str) + "|f" + long_df["fold"].astype(str) + "|" + long_df["sample_id"].astype(str)
    meta = long_df.drop(columns=["lipid", "shap_value"], errors="ignore").drop_duplicates(
        subset=["exp_folder", "fold", "sample_id"]
    ).copy()
    meta["instance_id"] = (
        meta["exp_folder"].astype(str)
        + "|f"
        + meta["fold"].astype(str)
        + "|"
        + meta["sample_id"].astype(str)
    )
    meta = meta.set_index("instance_id")
    # Ensure alignment by reindexing
    meta = meta.reindex(module_scores.index)
    merged = pd.concat([meta[meta_cols], module_scores], axis=1)
    merged.to_csv(osp.join(output_dir, "module_scores_by_instance.csv"))

    # Age association per module (Spearman)
    age_assoc_rows: List[Dict[str, object]] = []
    for col in module_scores.columns:
        age = merged["age"].astype(float)
        vals = merged[col].astype(float)
        mask = age.notna() & vals.notna()
        if mask.sum() < 10:
            continue
        rho, p = spearmanr(age[mask], vals[mask])
        age_assoc_rows.append({"module": col, "spearman_rho": rho, "p_value": p})
    age_assoc = pd.DataFrame(age_assoc_rows).sort_values("p_value")
    if not age_assoc.empty:
        age_assoc["q_value"] = benjamini_hochberg(age_assoc["p_value"]).values
        age_assoc.to_csv(osp.join(output_dir, "module_age_associations.csv"), index=False)

    # Outcome association per module (Mann-Whitney U on |SHAP| sums)
    outcome_rows: List[Dict[str, object]] = []
    outcome = merged["true_adrenal_insufficiency"].astype(int)
    for col in module_scores.columns:
        vals = merged[col].astype(float)
        g0 = vals[outcome == 0].dropna()
        g1 = vals[outcome == 1].dropna()
        if g0.shape[0] >= 5 and g1.shape[0] >= 5:
            stat, p = mannwhitneyu(g0, g1, alternative="two-sided")
            outcome_rows.append({"module": col, "u_stat": stat, "p_value": p, "n0": g0.shape[0], "n1": g1.shape[0]})
    outcome_assoc = pd.DataFrame(outcome_rows).sort_values("p_value")
    if not outcome_assoc.empty:
        outcome_assoc["q_value"] = benjamini_hochberg(outcome_assoc["p_value"]).values
        outcome_assoc.to_csv(osp.join(output_dir, "module_outcome_associations.csv"), index=False)

    # Optional: ROC AUC using single-module score as predictor (training instances only)
    auc_rows: List[Dict[str, object]] = []
    y_true = outcome.values
    for col in module_scores.columns:
        scores = merged[col].astype(float).values
        if np.sum(~np.isnan(scores)) < 10:
            continue
        # Replace NaN with median for simple scoring
        s = pd.Series(scores)
        s = s.fillna(s.median())
        try:
            auc = roc_auc_score(y_true, s)
            auc_rows.append({"module": col, "roc_auc": auc})
        except Exception:
            continue
    if auc_rows:
        pd.DataFrame(auc_rows).sort_values("roc_auc", ascending=False).to_csv(
            osp.join(output_dir, "module_single_score_auc.csv"), index=False
        )

    # Boxplot of top 6 modules by outcome difference
    if not outcome_assoc.empty:
        top_mods = outcome_assoc.nsmallest(6, "p_value")["module"].tolist()
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
        axes = axes.flatten()
        for ax, mod in zip(axes, top_mods):
            vals0 = merged.loc[outcome == 0, mod]
            vals1 = merged.loc[outcome == 1, mod]
            ax.boxplot([vals0.dropna().values, vals1.dropna().values], labels=["No AI", "AI"])
            ax.set_title(mod)
        plt.tight_layout()
        plt.savefig(osp.join(output_dir, "module_scores_boxplots_by_outcome.png"), dpi=300)
        plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    normalize_flag = _flag_to_bool_or_none(args.normalize)
    exclude_controls_flag = _flag_to_bool_or_none(args.exclude_controls)
    vlcfas_only_flag = _flag_to_bool_or_none(args.vlcfas_only)

    print("Loading experiments...")
    merged, used_folders = discover_and_load(
        base_dir=args.base_dir,
        model_types=args.model_types,
        k=args.k,
        normalize=normalize_flag,
        imputer=args.imputer,
        exclude_controls=exclude_controls_flag,
        vlcfas_only=vlcfas_only_flag,
        year_prefix=args.year_prefix,
    )
    print(f"Loaded {len(merged)} rows from {len(used_folders)} experiments.")

    long_df, lipid_cols = melt_lipid_shap(merged)
    if args.save_aggregated_csv:
        long_df.to_csv(osp.join(args.output_dir, "all_instance_shap_long.csv"), index=False)

    print("Running age-stratified analysis...")
    age_stratified_analysis(
        long_df=long_df,
        output_dir=args.output_dir,
        age_bins=args.age_bins,
        min_presence=args.min_presence,
    )

    print("Computing SHAP co-variation and clustering...")
    matrix, corr, clusters_df = shap_covariance_and_clustering(
        long_df=long_df,
        output_dir=args.output_dir,
        corr_method=args.corr_method,
        cluster_distance_threshold=args.cluster_distance_threshold,
        min_presence=args.min_presence,
        top_lipids_for_clustering=args.top_lipids_for_clustering,
    )

    print("Computing module scores and associations...")
    module_scores_and_associations(
        matrix=matrix,
        long_df=long_df,
        clusters_df=clusters_df,
        output_dir=args.output_dir,
    )

    # Provenance
    provenance = {
        "filters": {
            "model_types": args.model_types,
            "k": args.k,
            "normalize": normalize_flag,
            "imputer": args.imputer,
            "exclude_controls": exclude_controls_flag,
            "vlcfas_only": vlcfas_only_flag,
            "year_prefix": args.year_prefix,
        },
        "parameters": {
            "age_bins": args.age_bins,
            "min_presence": args.min_presence,
            "corr_method": args.corr_method,
            "top_lipids_for_clustering": args.top_lipids_for_clustering,
            "cluster_distance_threshold": args.cluster_distance_threshold,
        },
        "used_folders": used_folders,
    }
    with open(osp.join(args.output_dir, "analysis_provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    print("Done. Outputs written to:", args.output_dir)


if __name__ == "__main__":
    main()


