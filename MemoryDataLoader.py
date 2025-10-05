from __future__ import annotations
import io, re, pandas as pd, numpy as np, joblib
from typing import Optional, Sequence, List, Tuple


# columns we actually need for triage/vetting
USECOLS_DEFAULT: List[str] = [
    "pl_name","hostname","disc_facility","disposition",
    "pl_orbper","koi_period","koi_depth","pl_trandep",
    "pl_rade","st_rad","koi_duration","pl_trandurh",
    "koi_time0bk","pl_tranmid","koi_model_snr","koi_impact",
    "koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec",
    "default_flag","rowupdate"
]
NUMERIC_DEFAULT: List[str] = [
    "pl_orbper","koi_period","koi_depth","pl_trandep","pl_rade","st_rad",
    "koi_duration","pl_trandurh","koi_time0bk","pl_tranmid","koi_model_snr",
    "koi_impact","default_flag"
]
TEXT_DEFAULT: List[str] = ["pl_name","hostname","disc_facility","disposition"]

def _strip_html_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"<.*?>", "", regex=True)

def _coerce_float32_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce").astype("float32")

def _pick_existing(header: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in header.columns]

def load_uploaded_csv(
    uploaded, 
    usecols: Optional[Sequence[str]] = None, 
    prefer_arrow: bool = True
) -> pd.DataFrame:
    """
    Read a large exoplanet CSV from Streamlit's uploader with low RAM:
      - keeps only selected columns,
      - uses engine='pyarrow' when available (fast & compact),
      - coerces numerics to float32,
      - prefers default rows and de-dupes.
    """
    # Convert uploaded file to bytes if it's not already
    if hasattr(uploaded, 'read'):
        # If it's a file-like object, read its contents
        content = uploaded.read()
        if isinstance(content, str):
            content = content.encode('utf-8')
    else:
        content = uploaded

    # Create BytesIO object
    buf = io.BytesIO(content)
    
    # header sniff
    header = pd.read_csv(buf, nrows=0)
    usecols = list(usecols) if usecols else USECOLS_DEFAULT
    keep = _pick_existing(header, usecols)
    if not keep:
        raise ValueError("None of the expected columns are present; check your CSV header.")

    # Reset buffer position
    buf.seek(0)
    
    # Main read
    df = pd.read_csv(buf, low_memory=False)
   
    # clean text + numericize
    for c in TEXT_DEFAULT:
        if c in df.columns:
            df[c] = _strip_html_series(df[c]).astype("string")
    for c in NUMERIC_DEFAULT:
        if c in df.columns:
            df[c] = _coerce_float32_series(df[c])

    # stable sort & drop dups (prefer archive-recommended rows)
    if "rowupdate" in df.columns:
        df["rowupdate"] = pd.to_datetime(df["rowupdate"], errors="coerce")
    if "default_flag" in df.columns:
        df = df.sort_values(["default_flag","rowupdate"], ascending=[False, False], na_position="last")

    subset_keys = [k for k in ["hostname","pl_name","kepoi_name","toi"] if k in df.columns]
    if subset_keys:
        df = df.drop_duplicates(subset=subset_keys, keep="first")

    return df.reset_index(drop=True)

def load_joblib_model(uploaded) -> joblib:
    """Safely load a scikit-learn pipeline sent via Streamlit uploader."""
    return joblib.load(io.BytesIO(uploaded.getvalue()))

def pick_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["kepid","kepoi_name","koi_name","kic","epic","tic","toi","hostname","pl_name","name","source_id"]
    return next((c for c in candidates if c in df.columns), None)

def select_numeric_features(df: pd.DataFrame, label: Optional[str], id_col: Optional[str]) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = set(filter(None, [label, id_col]))
    num_cols = [c for c in num_cols if c not in drop]
    # keep columns with <95% missing
    miss = df[num_cols].isna().mean()
    feats = [c for c in num_cols if miss[c] < 0.95]
    return feats