import pandas as pd
import numpy as np
class DatasetSplitter:
    def __init__(self, test_size: float = 0.4, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame):
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_test = int(len(df) * self.test_size)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

def pseudo_label_from_row(row: pd.Series) -> str:
    if pd.isna(row.get("disposition")):
        return "Unknown"
    
    disp = str(row.get("disposition")).upper()
    if "CONFIRMED" in disp:
        return "Planet"
    if "FALSE" in disp:
        return "InstrumentalNoise"
    if "CANDIDATE" in disp:
        return "Unknown"
    return "Unknown"