import io
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint, uniform as sp_uniform

# ---------- Utilities ----------
class TransitVetter:
    def __init__(self):
        pass

    def first_non_null(self, row, cols, default=np.nan):
        for c in cols:
            if c in row and pd.notna(row[c]):
                return row[c]
        return default

    def map_four_class(self, row) -> Optional[str]:
        """
        Returns one of: Planet, EclipsingBinary, VariableStar, InstrumentalNoise
        or None if unlabeled.
        """
        disp = str(self.first_non_null(row, ["disposition", "koi_disposition"])).upper()
        tfop = str(row.get("tfopwg_disp", "")).upper()

        fp_nt = row.get("koi_fpflag_nt", np.nan)  # not transit-like
        fp_ss = row.get("koi_fpflag_ss", np.nan)  # centroid offset
        fp_co = row.get("koi_fpflag_co", np.nan)  # contamination
        fp_ec = row.get("koi_fpflag_ec", np.nan)  # eclipsing binary

        if "CONFIRMED" in disp or "CP" in tfop:
            return "Planet"
        if (pd.notna(fp_ec) and int(fp_ec) == 1) or "EB" in tfop or "ECLIPS" in tfop:
            return "EclipsingBinary"
        if pd.notna(fp_nt) and int(fp_nt) == 1:
            return "InstrumentalNoise"
        if (pd.notna(fp_ss) and int(fp_ss) == 1) or (pd.notna(fp_co) and int(fp_co) == 1):
            return "InstrumentalNoise"

        period = self.first_non_null(row, ["pl_orbper", "koi_period"])
        dur_h  = self.first_non_null(row, ["pl_trandurh", "koi_duration"])
        try:
            if pd.notna(period) and pd.notna(dur_h):
                ratio = float(dur_h) / (float(period) * 24.0)
                if ratio > 0.2:
                    return "VariableStar"
        except Exception:
            pass

        if "CANDIDATE" in disp:
            return None
        return None

    def build_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        feats["period_days"]    = df.apply(lambda r: self.first_non_null(r, ["pl_orbper", "koi_period"]), axis=1)
        feats["duration_hours"] = df.apply(lambda r: self.first_non_null(r, ["pl_trandurh", "koi_duration"]), axis=1)
        depth_ppm = df.apply(lambda r: self.first_non_null(r, ["pl_trandep", "koi_depth"]), axis=1)
        feats["depth_ppm"] = depth_ppm
        feats["depth_frac"] = pd.to_numeric(depth_ppm, errors="coerce") / 1e6

        feats["snr"]    = pd.to_numeric(df.get("koi_model_snr", np.nan), errors="coerce")
        feats["impact"] = pd.to_numeric(df.get("koi_impact", np.nan), errors="coerce")

        feats["teff"] = df.apply(lambda r: self.first_non_null(r, ["st_teff", "koi_steff"]), axis=1)
        feats["logg"] = df.apply(lambda r: self.first_non_null(r, ["st_logg", "koi_slogg"]), axis=1)
        feats["srad"] = df.apply(lambda r: self.first_non_null(r, ["st_rad", "koi_srad"]), axis=1)
        feats["met"]  = df.get("st_met", np.nan)

        feats["vmag"] = df.get("sy_vmag", np.nan)
        feats["dist_pc"] = df.get("sy_dist", np.nan)

        with np.errstate(invalid="ignore", divide="ignore"):
            feats["dur_over_period"] = pd.to_numeric(feats["duration_hours"], errors="coerce") / (
                pd.to_numeric(feats["period_days"], errors="coerce") * 24.0
            )
        return feats.apply(pd.to_numeric, errors="coerce")

    def make_labels(self, df: pd.DataFrame) -> pd.Series:
        labels = df.apply(self.map_four_class, axis=1)
        return labels

    def train_and_eval(
        self,
        csv_path: str,
        model_out: str = "classical_vet.joblib",
        algorithm: str = "lgbm",      # "lgbm" | "xgb" | "cat" | "rf"
        tune: bool = False,
        random_state: int = 42,
    ) -> Dict:
        df = pd.read_csv(csv_path, low_memory=False)
        X = self.build_feature_frame(df)
        y = self.make_labels(df)

        # Keep only labeled rows for training
        m = y.notna()
        Xl, yl = X[m], y[m]

        if yl.nunique() < 2 or len(yl) < 50:
            raise RuntimeError(f"Not enough labeled rows to train: classes={yl.value_counts().to_dict()}")

        Xtr, Xte, ytr, yte = train_test_split(
            Xl, yl, test_size=0.2, random_state=random_state, stratify=yl
        )

        clf = ClassicalVetTrainer(algorithm=algorithm, tune=tune, random_state=random_state)
        clf.fit(Xtr, ytr)

        ypred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)

        # Metrics
        report = classification_report(yte, ypred, digits=3)
        cm_labels = sorted(yl.unique())
        cm = confusion_matrix(yte, ypred, labels=cm_labels)
        f1m = f1_score(yte, ypred, average="macro")
        bacc = balanced_accuracy_score(yte, ypred)

        # Build results DF for your Streamlit streaming
        test_ids = df.loc[Xte.index, :]
        id_cols = [c for c in ["kepid","kepoi_name","koi_name","kic","epic","tic","toi","hostname","pl_name"] if c in test_ids.columns]
        keep_cols = id_cols + [c for c in ["lc_summary"] if c in test_ids.columns]
        res = pd.DataFrame(index=Xte.index)
        if keep_cols:
            res = test_ids[keep_cols].copy()
        res["pred"] = ypred
        if proba is not None and len(proba.shape) == 2:
            res["pred_conf"] = np.max(proba, axis=1)
        else:
            res["pred_conf"] = 1.0

        # Feature importance (best-effort)
        feat_imp = clf.feature_importances()
        feat_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:30]

        # Save model
        clf.save(model_out)

        return {
            "report": report,
            "labels": [str(x) for x in cm_labels],
            "confusion_matrix": cm.tolist(),
            "f1_macro": float(f1m),
            "balanced_accuracy": float(bacc),
            "feature_importance_top30": feat_imp,
            "results": res.reset_index(drop=True),
            "model_params": clf.get_params(),
            "algorithm": algorithm,
        }

# ---------- Trainer (Upgraded) ----------
class ClassicalVetTrainer:
    """
    Strong baseline with switchable algorithms:
      - LightGBM (default)
      - XGBoost
      - CatBoost
      - RandomForest (fallback)
    Handles: imputation, stratified CV (for tuning), class imbalance via sample_weight.
    """
    def __init__(self, algorithm: str = "lgbm", tune: bool = False, random_state: int = 42):
        self.algorithm = algorithm.lower()
        self.tune = tune
        self.random_state = random_state
        self.pipe = None
        self.feature_names_ = None
        self.label_names_ = None
        self.classes_ = None

    def _build_estimator(self, n_classes: int):
        alg = self.algorithm
        if alg == "rf":
            return RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=self.random_state,
                class_weight=None  # we pass sample_weight instead
            )
        elif alg == "lgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as e:
                raise ImportError("LightGBM not installed. `pip install lightgbm`") from e
            return LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif alg == "xgb":
            try:
                from xgboost import XGBClassifier
            except ImportError as e:
                raise ImportError("XGBoost not installed. `pip install xgboost`") from e
            return XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=-1
            )
        elif alg == "cat":
            try:
                from catboost import CatBoostClassifier
            except ImportError as e:
                raise ImportError("CatBoost not installed. `pip install catboost`") from e
            return CatBoostClassifier(
                loss_function="MultiClass",
                iterations=2000,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=self.random_state,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _param_distributions(self, n_classes: int):
        alg = self.algorithm
        if alg == "rf":
            return {
                "model__n_estimators": sp_randint(300, 1200),
                "model__max_depth": sp_randint(4, 24),
                "model__min_samples_split": sp_randint(2, 10),
                "model__min_samples_leaf": sp_randint(1, 6),
            }
        if alg == "lgbm":
            return {
                "model__n_estimators": sp_randint(400, 2000),
                "model__learning_rate": sp_uniform(0.01, 0.09),
                "model__num_leaves": sp_randint(31, 255),
                "model__subsample": sp_uniform(0.6, 0.4),
                "model__colsample_bytree": sp_uniform(0.6, 0.4),
                "model__reg_lambda": sp_uniform(0.0, 2.0),
            }
        if alg == "xgb":
            return {
                "model__n_estimators": sp_randint(400, 2000),
                "model__learning_rate": sp_uniform(0.01, 0.09),
                "model__max_depth": sp_randint(4, 16),
                "model__subsample": sp_uniform(0.6, 0.4),
                "model__colsample_bytree": sp_uniform(0.6, 0.4),
                "model__reg_lambda": sp_uniform(0.0, 2.0),
            }
        if alg == "cat":
            return {
                "model__iterations": sp_randint(800, 3000),
                "model__learning_rate": sp_uniform(0.01, 0.09),
                "model__depth": sp_randint(4, 10),
                "model__l2_leaf_reg": sp_uniform(1.0, 5.0),
            }
        return {}

    def _compute_sample_weights(self, y: pd.Series) -> np.ndarray:
        classes = np.array(sorted(pd.unique(y)))
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        mapping = {c: w for c, w in zip(classes, weights)}
        return y.map(mapping).astype(float).values

    def fit(self, X: pd.DataFrame, y: pd.Series):
        mask = y.notna()
        Xtr, ytr = X[mask], y[mask]
        self.feature_names_ = list(Xtr.columns)
        self.label_names_ = sorted(list(pd.unique(ytr)))
        self.classes_ = np.array(self.label_names_)
        n_classes = len(self.label_names_)

        # Build estimator now that we know n_classes
        estimator = self._build_estimator(n_classes)

        # Pipeline: impute -> model
        self.pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", estimator)
        ])

        # class imbalance → sample weights
        sw = self._compute_sample_weights(ytr)

        if self.tune:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            param_dist = self._param_distributions(n_classes)
            # macro-F1 focuses on minority classes
            search = RandomizedSearchCV(
                estimator=self.pipe,
                param_distributions=param_dist,
                n_iter=25,
                scoring="f1_macro",
                n_jobs=-1,
                cv=cv,
                verbose=0,
                random_state=self.random_state,
            )
            search.fit(Xtr, ytr, model__sample_weight=sw)
            self.pipe = search.best_estimator_
        else:
            self.pipe.fit(Xtr, ytr, model__sample_weight=sw)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        mdl = self.pipe.named_steps["model"]
        if hasattr(mdl, "predict_proba"):
            return self.pipe.predict_proba(X)
        return None

    def feature_importances(self) -> Dict[str, float]:
        mdl = self.pipe.named_steps["model"]
        importances = None
        # CatBoost
        try:
            from catboost import CatBoostClassifier  # noqa
            if isinstance(mdl, CatBoostClassifier):
                vals = mdl.get_feature_importance()
                importances = {f: float(v) for f, v in zip(self.feature_names_, vals)}
        except Exception:
            pass
        # Generic tree-based
        if importances is None and hasattr(mdl, "feature_importances_"):
            vals = getattr(mdl, "feature_importances_")
            importances = {f: float(v) for f, v in zip(self.feature_names_, vals)}
        # Linear backup
        if importances is None and hasattr(mdl, "coef_"):
            vals = np.abs(mdl.coef_).sum(axis=0)
            importances = {f: float(v) for f, v in zip(self.feature_names_, vals)}
        if importances is None:
            importances = {f: 0.0 for f in self.feature_names_}
        return importances

    def save(self, path: str):
        joblib.dump({
            "pipeline": self.pipe,
            "features": self.feature_names_,
            "labels": self.label_names_,
            "algorithm": self.algorithm,
            "random_state": self.random_state
        }, path)

    @staticmethod
    def load(path: str):
        blob = joblib.load(path)
        obj = ClassicalVetTrainer(algorithm=blob.get("algorithm","rf"),
                                  random_state=blob.get("random_state",42))
        obj.pipe = blob["pipeline"]
        obj.feature_names_ = blob["features"]
        obj.label_names_ = blob["labels"]
        obj.classes_ = np.array(obj.label_names_)
        return obj

    def get_params(self) -> Dict:
        mdl = self.pipe.named_steps["model"]
        return {"algorithm": self.algorithm, **mdl.get_params(deep=False)}
