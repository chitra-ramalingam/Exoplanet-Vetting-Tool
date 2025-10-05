from DataUtils import pseudo_label_from_row
import numpy as np
import pandas as pd

class Evaluator:
    LABELS = ["Planet","EclipsingBinary","VariableStar","InstrumentalNoise","Unknown"]

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        n = len(Evaluator.LABELS)
        mat = np.zeros((n, n), dtype=int)
        idx = {lab:i for i, lab in enumerate(Evaluator.LABELS)}
        for t, p in zip(y_true, y_pred):
            i = idx.get(t, idx["Unknown"])
            j = idx.get(p, idx["Unknown"])
            mat[i, j] += 1
        return mat

    @staticmethod
    def pseudo_labels_from_df(df: pd.DataFrame):
        return [pseudo_label_from_row(r) for _, r in df.iterrows()]
