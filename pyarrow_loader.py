
# pyarrow_loader.py
# Requires: pandas >= 2.1 and pyarrow installed (pip install pyarrow)
from __future__ import annotations
import re
from typing import List, Optional, Sequence

import pandas as pd

class PyarrowLoader:
    """Memory-friendly CSV loader for the vetting tool using pandas+pyarrow.

    - Reads only the columns needed for lc_summary + vetting.
    - Uses engine="pyarrow" and Arrow-backed dtypes (compact, fast).
    - Coerces numeric columns to float32.
    - Optionally prefers archive "default" rows and drops duplicates.
    - Can save the filtered, typed result to Parquet.
    """

    DEFAULT_USECOLS: List[str] = [
        "pl_name","hostname","disc_facility","disposition",
        "pl_orbper","koi_period","koi_depth","pl_trandep",
        "pl_rade","st_rad","koi_duration","pl_trandurh",
        "koi_time0bk","pl_tranmid","koi_model_snr","koi_impact",
        "koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec",
        "default_flag","rowupdate"
    ]

    DEFAULT_NUMERIC: List[str] = [
        "pl_orbper","koi_period","koi_depth","pl_trandep",
        "pl_rade","st_rad","koi_duration","pl_trandurh",
        "koi_time0bk","pl_tranmid","koi_model_snr","koi_impact",
        "default_flag"
    ]

    DEFAULT_TEXT: List[str] = ["pl_name","hostname","disc_facility","disposition"]

    def __init__(
        self,
        usecols: Optional[Sequence[str]] = None,
        numeric_cols: Optional[Sequence[str]] = None,
        text_cols: Optional[Sequence[str]] = None,
    ) -> None:
        self.usecols = list(usecols) if usecols is not None else list(self.DEFAULT_USECOLS)
        self.numeric_cols = list(numeric_cols) if numeric_cols is not None else list(self.DEFAULT_NUMERIC)
        self.text_cols = list(text_cols) if text_cols is not None else list(self.DEFAULT_TEXT)

    @staticmethod
    def _strip_html_series(s: pd.Series) -> pd.Series:
        # robust even on Arrow strings
        return s.astype("string").str.replace(r"<.*?>", "", regex=True)

    @staticmethod
    def _coerce_float32_series(s: pd.Series) -> pd.Series:
        # drop commas and coerce to float32
        s = s.astype("string").str.replace(",", "", regex=False)
        return pd.to_numeric(s, errors="coerce").astype("float32")

    def _existing_cols(self, path: str) -> List[str]:
        hdr = pd.read_csv(path, nrows=0)
        return [c for c in self.usecols if c in hdr.columns]

    def load(
        self,
        path: str,
        prefer_default: bool = True,
        dedupe: bool = True,
        subset_keys: Optional[Sequence[str]] = None,
        parse_rowupdate: bool = True,
    ) -> pd.DataFrame:
        """Load CSV with pyarrow engine and minimal memory.

        Args:
          path: CSV path.
          prefer_default: If True, sort to keep default_flag==1 where present.
          dedupe: Drop duplicate planet/host rows.
          subset_keys: Columns to define uniqueness (fallback uses existing of
                       ["hostname","pl_name","kepoi_name","toi"]). 
          parse_rowupdate: Attempt to parse rowupdate as datetime for stable sorting.
        """
        keep = self._existing_cols(path)
        if not keep:
            raise ValueError("None of the requested columns exist in the CSV. "
                             "Check your header or usecols.")

        # Arrow-backed read: compact and fast
        df = pd.read_csv(
            path,
            engine="pyarrow",
            usecols=keep,
            dtype_backend="pyarrow"
        )

        # Clean text columns (strip HTML anchors etc.) and standardize dtype
        for c in self.text_cols:
            if c in df.columns:
                df[c] = self._strip_html_series(df[c]).astype("string[pyarrow]")

        # Numeric coercion → float32
        for c in self.numeric_cols:
            if c in df.columns:
                df[c] = self._coerce_float32_series(df[c])

        # Parse rowupdate if present (helps when prefer_default=True)
        if parse_rowupdate and "rowupdate" in df.columns:
            df["rowupdate"] = pd.to_datetime(df["rowupdate"], errors="coerce")

        # Prefer archive "default" rows if available
        if prefer_default and "default_flag" in df.columns:
            sort_cols = ["default_flag"]
            ascending = [False]
            if "rowupdate" in df.columns:
                sort_cols.append("rowupdate"); ascending.append(False)
            df = df.sort_values(sort_cols, ascending=ascending, na_position="last")

        # Drop duplicates across sensible keys
        if dedupe:
            if subset_keys is None:
                subset_keys = [c for c in ["hostname","pl_name","kepoi_name","toi"] if c in df.columns]
            if subset_keys:
                df = df.drop_duplicates(subset=subset_keys, keep="first")

        # Make sure text columns are compact Arrow strings
        for c in self.text_cols:
            if c in df.columns and str(df[c].dtype) != "string[pyarrow]":
                df[c] = df[c].astype("string[pyarrow]")

        return df.reset_index(drop=True)

    def to_parquet(self, df: pd.DataFrame, path: str, compression: str = "zstd") -> None:
        df.to_parquet(path, index=False, compression=compression)
