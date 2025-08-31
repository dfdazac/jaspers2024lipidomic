import os
import os.path as osp
import re
from typing import Dict, Optional, Tuple, List

import pandas as pd


def _load_vlcfa_thresholds(csv_path: Optional[str] = None) -> Dict[str, int]:
    """
    Load VLCFA thresholds from the vlcfas CSV/TSV file.

    The file is expected to have at least the columns:
    - 'Lipid Class'
    - 'Minimum Sum Carbon Length for VLCFA'

    A single row can specify multiple classes separated by commas.
    Returns a mapping from lipid base class to minimum sum carbon length.
    """
    if csv_path is None:
        # Default path: project_root/data/vlcfas.csv
        csv_path = osp.abspath(osp.join(osp.dirname(__file__), "..", "data", "vlcfas.csv"))

    # File is TSV-formatted (tab-separated)
    df = pd.read_csv(csv_path, sep="\t")

    class_col = "Lipid Class"
    min_len_col = "Minimum Sum Carbon Length for VLCFA"

    thresholds: Dict[str, int] = {}
    for _, row in df.iterrows():
        classes_field = str(row[class_col])
        try:
            min_sum_carbon = int(row[min_len_col])
        except Exception:
            # Skip rows without a valid integer threshold
            continue

        for cls in classes_field.split(","):
            cls_clean = cls.strip()
            if cls_clean:
                thresholds[cls_clean] = min_sum_carbon

    return thresholds


_HEX_CER_CLASSES_WITHOUT_DT_VARIANT = {"HexCer", "Hex2Cer", "Hex3Cer"}
_DT_VARIANT_CLASSES = {"SM", "SM4", "Cer", "GM1", "GM2", "GM3"}


def _parse_lipid_name(lipid_name: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse a lipid column name into (base_class, sum_carbon_count).

    Examples:
    - "PC(O+P-34:2)" -> ("PC(O+P)", 34)
    - "LPE(O-24:1)" -> ("LPE(O)", 24)
    - "HexCer(d35:1)" -> ("HexCer", 35)
    - "Cer(d34:1)" -> ("Cer(d)", 34)
    - "SM(t42:2)" -> ("SM(t)", 42)
    - "1-acyl LPC(14:1)" -> ("1-acyl LPC", 14)
    - "LPC(19:1)" -> ("LPC", 19)
    """
    if not isinstance(lipid_name, str):
        return None, None

    name = lipid_name.strip()

    # Extract total carbon count: first occurrence of "<digits>:<digits>"
    carbon_match = re.search(r"(\d+):\d+", name)
    sum_carbons: Optional[int] = int(carbon_match.group(1)) if carbon_match else None

    # Extract head class and inside-parentheses part
    m = re.match(r"^([A-Za-z0-9\-\s]+)\(([^)]*)\)$", name)
    if not m:
        # If no parentheses, treat the entire name as the class label
        base_class = name
        return base_class, sum_carbons

    head = m.group(1).strip()
    inside = m.group(2).strip()

    # Variant token is the first segment before '-' if any (e.g., 'O+P' in 'O+P-34:2')
    variant_token = inside.split("-", 1)[0]

    # Handle plasmalogen/ether variants: O, P, O+P
    if variant_token in {"O", "P", "O+P"}:
        base_class = f"{head}({variant_token})"
        return base_class, sum_carbons

    # Handle d/t sphingoid variants (e.g., d35:1, t42:2)
    if re.match(r"^[dt]", variant_token):
        dt_char = variant_token[0]  # 'd' or 't'
        if head in _HEX_CER_CLASSES_WITHOUT_DT_VARIANT:
            # CSV uses bare names for HexCer subclasses (no (d)/(t))
            base_class = head
        elif head in _DT_VARIANT_CLASSES:
            # CSV encodes d/t variant at class level, e.g., 'Cer(d)', 'SM(t)'
            base_class = f"{head}({dt_char})"
        else:
            # Default: ignore d/t and use the head only
            base_class = head
        return base_class, sum_carbons

    # Default: numeric-only variant in parentheses (e.g., '34:2')
    base_class = head
    return base_class, sum_carbons


def select_vlcfas(df: pd.DataFrame, vlcfa_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Return a DataFrame with only VLCFA lipid columns based on rules in vlcfas.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with lipid measurements as columns.
    vlcfa_csv_path : Optional[str]
        Path to the VLCFA rules file (default: project_root/data/vlcfas.csv).

    Returns
    -------
    pd.DataFrame
        A view of `df` containing only columns that are VLCFAs.
    """
    thresholds = _load_vlcfa_thresholds(vlcfa_csv_path)

    selected_columns: List[str] = []
    for col in df.columns:
        base_class, sum_carbons = _parse_lipid_name(str(col))
        if base_class is None or sum_carbons is None:
            continue

        min_required = thresholds.get(base_class)
        if min_required is None:
            # Not a recognized lipid class in the VLCFA rules
            continue

        if sum_carbons >= min_required:
            selected_columns.append(col)

    return df[selected_columns]


__all__ = [
    "select_vlcfas",
]


