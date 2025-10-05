# --- keep your existing imports ---

import streamlit as st
import pandas as pd
import numpy as np
import io, joblib, os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Optional

# NEW: bring in your service + evaluator helpers
from VettingService import VettingService
from Evaluator import Evaluator
from DataUtils import pseudo_label_from_row
from MLModel.TransitVetter import TransitVetter


st.set_page_config(page_title="Exoplanet Batch Triage", layout="wide")

# ---------- MENU ----------
st.sidebar.header("Mode")
mode = st.sidebar.radio("Select view", ["Batch Triage", "Classifier"], index=0)

# ---------- Shared sidebar inputs (reuse your uploader for both) ----------
st.sidebar.header("Inputs")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
model_file = st.sidebar.file_uploader("Optional: Upload trained model (.joblib)", type=["joblib"], accept_multiple_files=False)

# =====================================================================
# BATCH TRIAGE (your existing flow) — wrap your current code in this block
# =====================================================================
if mode == "Batch Triage":
    st.title("🪐 Exoplanet Batch Triage (CSV → Predictions)")
    st.caption("Upload a CSV, run batch inference, review a sortable table, and download the results.")

    with st.expander("How it works"):
        st.markdown("""
        - **Upload a CSV** similar to NASA/Kepler/TESS catalogs (your `Combined_Exoplanet_Data.csv` works).
        - **Optionally upload a trained model** (`.joblib`). If you don't upload one and your CSV has a `disposition` column,
          the app will **train a quick baseline** on the fly and then predict on the same CSV.
        - The app produces **prediction**, **confidence**, and a **priority score** to rank borderline cases.
        - Click **Download** to save a copy of the predictions as CSV.
        """)

    # IMPORTANT: move the "no CSV" check inside this branch
    if csv_file is None:
        st.info("Upload a CSV to begin.")
        st.stop()

    # ================= your existing TRIAGE code below (unchanged) =================
    # Read CSV
    content = csv_file.read()
    df = pd.read_csv(io.BytesIO(content), low_memory=False)
    st.success(f"Loaded CSV with shape {df.shape[0]:,} rows × {df.shape[1]:,} columns.")

    # Try to pick an ID column for display
    id_candidates = ["kepid","kepoi_name","koi_name","kic","epic","tic","toi","hostname","pl_name","name","source_id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)

    # Helper: build / train a baseline if needed
    def train_baseline_if_possible(df: pd.DataFrame, id_col: Optional[str]) -> Optional[Pipeline]:
        if "disposition" not in df.columns:
            return None
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ["disposition", id_col]]
        if not num_cols: return None
        dfl = df.dropna(subset=["disposition"]).copy()
        if len(dfl) < 20: return None
        miss = dfl[num_cols].isna().mean()
        feats = [c for c in num_cols if miss[c] < 0.95]
        if not feats: return None
        X = dfl[feats]; y = dfl["disposition"].astype(str)
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception:
            Xtr, Xte, ytr, yte = X, X, y, y
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ])
        pipe.fit(Xtr, ytr)
        try:
            ypred = pipe.predict(Xte)
            rep = classification_report(yte, ypred, output_dict=False)
            st.sidebar.text("Quick baseline quality:\n" + rep)
        except Exception:
            pass
        pipe.feat_names_in_ = np.array(feats)
        return pipe

    def compute_priority(conf):
        conf = np.clip(conf, 0.0, 1.0)
        return (1.0 - (np.abs(conf - 0.5) * 2.0)).clip(0, 1)

    # Load or train model
    model = None
    if model_file is not None:
        try:
            model = joblib.load(io.BytesIO(model_file.read()))
            st.sidebar.success("Loaded uploaded model.")
        except Exception as e:
            st.sidebar.error(f"Couldn't load model: {e}")

    if model is None:
        st.sidebar.warning("No model uploaded. Attempting to train a quick baseline from CSV (requires 'disposition' column).")
        model = train_baseline_if_possible(df, id_col)
        if model is None:
            st.error("A model is required. Upload a .joblib model, or provide a CSV with a 'disposition' column so a baseline can be trained.")
            st.stop()

    # Build feature matrix for prediction
    if hasattr(model, "feat_names_in_"):
        feat_cols = list(model.feat_names_in_)
    else:
        feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for bad in ["disposition", id_col] + [c for c in ["label","class"] if c in df.columns]:
            if bad in feat_cols:
                feat_cols.remove(bad)
    if not feat_cols:
        st.error("No usable numeric feature columns were found for prediction.")
        st.stop()

    Xall = df[feat_cols].copy()
    try:
        proba = model.predict_proba(Xall)
        classes = list(model.classes_)
        pred_idx = np.argmax(proba, axis=1)
        y_pred = np.array(classes)[pred_idx]
        conf = proba.max(axis=1)
    except Exception:
        y_pred = model.predict(Xall)
        conf = np.ones(len(y_pred))

    priority = compute_priority(conf)
    out = df.copy()
    out["prediction"] = y_pred
    out["confidence"] = np.round(conf, 4)
    out["priority_score"] = np.round(priority, 4)

    st.subheader("Review Table")
    display_cols = [id_col] if id_col else []
    display_cols += ["prediction", "confidence", "priority_score"]
    if "disposition" in out.columns:
        display_cols.insert(1, "disposition")
    review_df = out[display_cols].copy()
    st.dataframe(review_df, use_container_width=True, height=480)

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download predictions as CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

    with st.expander("Diagnostics (feature columns used)"):
        st.code(", ".join(feat_cols))
        if hasattr(getattr(model, "named_steps", {}).get("clf", None), "feature_importances_"):
            importances = getattr(model.named_steps["clf"], "feature_importances_")
            imp_df = pd.DataFrame({"feature": feat_cols, "importance": importances}).sort_values("importance", ascending=False).head(20)
            st.write(imp_df)

# =====================================================================
# CLASSIFIER (ML Vetting) — this calls your VettingService
# =====================================================================
elif mode == "Classifier":
    import io, os, tempfile
    import pandas as pd
    import matplotlib.pyplot as plt
    from typing import Optional

    st.title("🧪 ML Classifier (LLM Vetting)")
    st.caption("Streams one result at a time, then shows the full table when done.")

    if csv_file is None:
        st.info("Upload a CSV to begin.")
        st.stop()

    with st.expander("How it works"):
        # 1) Save uploaded file to a temporary path if your classifier needs a path
        #    (If your classifier can accept a DataFrame, see the alt path below.)
        st.caption("Calling the classifier and streaming results...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_file.getbuffer())
            tmp.flush()
            csv_path = tmp.name

        try:
    # ---- pick model + tuning ----
            algo_label = st.sidebar.selectbox("Algorithm", ["LightGBM", "XGBoost", "CatBoost", "RandomForest"], index=0)
            alg_key = {"LightGBM":"lgbm","XGBoost":"xgb","CatBoost":"cat","RandomForest":"rf"}[algo_label]
            tune = st.sidebar.checkbox("Tune (RandomizedSearchCV)", value=False)

            tClassifier = TransitVetter()
            metrics = tClassifier.train_and_eval(csv_path=csv_path, algorithm=alg_key, tune=tune)

            # ---- streaming containers ----
            ph_progress = st.progress(0)
            ph_count = st.empty()
            ph_table = st.empty()

            # ---- stream predictions from metrics["results"] ----
            if "results" in metrics and isinstance(metrics["results"], pd.DataFrame):
                res_df = metrics["results"]
                rows, total = [], len(res_df)

                for i, (_, row) in enumerate(res_df.iterrows()):
                    rows.append(row)
                    seen = i + 1
                    ph_count.write(f"Processed: {seen} / {total}")

                    partial_df = pd.DataFrame(rows)
                    id_cols = [c for c in ["kepid","kepoi_name","koi_name","kic","epic","tic","toi","hostname","pl_name"] if c in partial_df.columns]
                    cols_to_show = (id_cols[:1] if id_cols else []) + [c for c in ["pred","pred_conf","lc_summary"] if c in partial_df.columns]
                    ph_table.dataframe(partial_df[cols_to_show] if cols_to_show else partial_df,
                                    use_container_width=True, height=420)
                    ph_progress.progress(int(seen / total * 100))

                result_df = pd.DataFrame(rows).reset_index(drop=True)
                st.success(f"Done. Predicted {len(result_df)} rows.")
                st.download_button(
                    "⬇️ Download predictions as CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="vetting_predictions.csv",
                    mime="text/csv",
                )

                # simple chart
                if "pred" in result_df.columns:
                    import matplotlib.pyplot as plt
                    counts = result_df["pred"].value_counts().sort_index()
                    fig = plt.figure(figsize=(6,4))
                    counts.plot(kind="bar")
                    plt.title("Predicted label counts")
                    plt.xlabel("Label"); plt.ylabel("Count")
                    plt.tight_layout()
                    st.pyplot(fig)

            # ---- training metrics ----
            st.subheader("Training Metrics")

            top = st.columns(3)
            top[0].metric("F1 (macro)", f"{metrics.get('f1_macro', float('nan')):.3f}")
            top[1].metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', float('nan')):.3f}")
            top[2].write(f"**Algorithm:** {metrics.get('algorithm','?').upper()}")

            cols = st.columns(2)
            if "report" in metrics:
                with cols[0]:
                    st.text("Classification Report:")
                    st.text(metrics["report"])

            if "confusion_matrix" in metrics and "labels" in metrics:
                with cols[1]:
                    st.text("Confusion Matrix:")
                    cm_df = pd.DataFrame(metrics["confusion_matrix"],
                                        columns=metrics["labels"],
                                        index=metrics["labels"])
                    st.dataframe(cm_df, use_container_width=True)

            # feature importances (top 20)
            if "feature_importance_top30" in metrics:
                fi_pairs = metrics["feature_importance_top30"]
                fi_df = pd.DataFrame(fi_pairs, columns=["feature","importance"])
                with st.expander("Top feature importances"):
                    st.dataframe(fi_df, use_container_width=True)
                    import matplotlib.pyplot as plt
                    fig2 = plt.figure(figsize=(7,5))
                    fi_df.head(20).set_index("feature")["importance"].plot(kind="barh")
                    plt.gca().invert_yaxis()
                    plt.title("Top feature importances (head 20)")
                    plt.tight_layout()
                    st.pyplot(fig2)

            # model params
            if "model_params" in metrics:
                with st.expander("Model parameters"):
                    st.json(metrics["model_params"])

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            try:
                os.remove(csv_path)
            except Exception:
                pass
