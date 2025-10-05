
import json, re, math
from typing import List, Dict, Any, Optional, Tuple, Iterator
import numpy as np
import pandas as pd
import requests
import DataUtils as du
from DataUtils import pseudo_label_from_row
from Models.GeminiCompletion import GeminiClient
#from Classifier.HuggingFaceClient import HFVettingClient, HFClientConfig
from Classifier.OllamaClient import OllamaClient

# -----------------------------
# Config
# -----------------------------

class VettingConfig:
    VETTING_SCHEMA = {
        "type": "object",
        "properties": {
            "label": {"enum": ["Planet","EclipsingBinary","VariableStar","InstrumentalNoise","Unknown"]},
            "confidence": {"type":"number"},
            "tests": {"type":"object"},
            "rationale": {"type":"string"}
        },
        "required": ["label","confidence","rationale"]
    }
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral",
                 temperature: float = 0.2, timeout: int = 120, few_shot_k: int = 4):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.few_shot_k = few_shot_k

# -----------------------------
# Dataset utils
# -----------------------------



# -----------------------------
# Ollama client (with safe fallback)
# -----------------------------

# class OllamaClient:
#     def __init__(self, config: VettingConfig):
#         self.cfg = config

#     def _extract_json(self, text: str) -> dict:
#         m = re.search(r"\{.*\}", text, flags=re.S)
#         if not m:
#             raise ValueError("No JSON found in response")
#         return json.loads(m.group(0))

#     def classify(self, lc_summary: str, lc_fields: dict, few_shots: List[dict] = None) -> dict:
#         few_shots = few_shots or []
#         system = (
#             "You are an exoplanet vetting assistant. "
#             "Given a light-curve summary and associated fields, return ONLY a JSON object "
#             "that matches the provided schema."
#         )
#         user = f"Schema: {json.dumps(VettingConfig.VETTING_SCHEMA)}\n"
#         for ex in few_shots[: self.cfg.few_shot_k]:
#             user += f"\nExample Input: {ex['lc_summary']}\n"
#             user += f"Example Fields: {json.dumps(ex['lc_fields'])}\n"
#             user += f"Example Output: {json.dumps(ex['label_json'])}\n"
#         user += f"\nNow classify this target.\nInput: {lc_summary}\nFields: {json.dumps(lc_fields)}\n"
#         user += "Return JSON only."

#         body = {
#             "model": self.cfg.model,
#             "messages": [{"role":"system","content":system},{"role":"user","content":user}],
#             "options": {"temperature": self.cfg.temperature}
#         }
#         try:
#             r = requests.post(f"{self.cfg.base_url}/api/chat", json=body, timeout=self.cfg.timeout)
#             r.raise_for_status()
#             data = r.json()
#             text = data["message"]["content"]
#             return self._extract_json(text)
#         except Exception as e:
#             label, conf, rationale = self._fallback_rule(lc_summary, lc_fields)
#             return {"label": label, "confidence": conf, "tests": {}, "rationale": rationale}

#     def _fallback_rule(self, lc_summary: str, lc_fields: dict):
#         disp = (lc_fields.get("disposition") or "").upper()
#         depth = lc_fields.get("depth_fraction")
#         period = lc_fields.get("period_days")
#         label = "Unknown"
#         conf = 0.5
#         rationale = "Fallback heuristic used."

#         if "CONFIRMED" in disp:
#             label, conf = "Planet", 0.8
#             rationale = "Disposition indicates confirmed planet."
#         elif depth is not None and depth > 0.02:
#             label, conf = "EclipsingBinary", 0.7
#             rationale = "Very deep transit depth (>2%) suggests EB."
#         elif period is None:
#             label, conf = "VariableStar", 0.6
#             rationale = "No reliable period suggests variability."
#         else:
#             label, conf = "Unknown", 0.4
#             rationale = "Insufficient evidence."

#         return label, conf, rationale

# -----------------------------
# Classifier
# -----------------------------
# ...existing code...
# AstraVetClassifier.py


class AstraVetClassifier:
    """LLM-based vetter using Ollama; few-shot examples from training rows."""
    def __init__(self, config: VettingConfig):
        self.cfg = config
        cfg = {
            "model": "gemini-2.5-flash",
            "temperature": 0.2,
            "timeout": 30,
        }

        self.gc = GeminiClient(cfg)


        self.few_shots: List[dict] = []

    def fit(self, train_df: pd.DataFrame, few_shot_k: Optional[int] = None):
        """Build lightweight few-shot examples from training rows."""
        k = few_shot_k if few_shot_k is not None else self.cfg.few_shot_k
        examples: List[dict] = []
        # keep it tiny for local models; 0–2 is plenty
        for _, row in train_df.head(max(0, int(k or 0))).iterrows():
            label = du.pseudo_label_from_row(row)
            conf = 0.8 if label == "Planet" else 0.6
            ex_json = {
                "label": label,
                "confidence": conf,
                "tests": {},
                "rationale": f"Derived from disposition={row.get('disposition')}"
            }
            examples.append({
                "lc_summary": row["lc_summary"],
                "lc_fields": row["lc_fields"],
                "label_json": ex_json
            })
        self.few_shots = examples

    def _iter_rows(self, df: pd.DataFrame, top_n: Optional[int] = None):
        """Yield (idx, row) in a deterministic order, optionally capped by top_n."""
        if top_n is not None:
            df = df.head(int(top_n))
        for idx, row in df.reset_index(drop=True).iterrows():
            yield int(idx), row

    # ---------- BATCH (sequential) ----------
    def predict(self, df: pd.DataFrame, top_n: Optional[int] = None) -> List[Dict]:
        """
        Return a list of classification dicts matching the df order (or first top_n rows).
        """
        top_n = 20
        results: List[Dict] = []
        for _, row in self._iter_rows(df, top_n=top_n):
            res = self.gc.classify(
                lc_summary=row["lc_summary"],
                lc_fields=row["lc_fields"],
            )
            results.append(res)
        return results

    # ---------- STREAM (sequential row-by-row) ----------
    # def predict_stream(self, df: pd.DataFrame, top_n: Optional[int] = None) -> Iterator[Tuple[int, Dict]]:
    #     """
    #     Yields per-row streaming events. For each row, you may see multiple
    #     {'event':'delta','text':...} entries, followed by one
    #     {'event':'result','data':{...}}. The caller (UI) decides how to render.
    #     """
    #     for idx, row in self._iter_rows(df, top_n=top_n):
    #         for ev in self.client.classify_stream(
    #             lc_summary=row["lc_summary"],
    #             lc_fields=row["lc_fields"],
    #             few_shots=self.few_shots
    #         ):
    #             # pass the row index along with the event
    #             yield idx, ev

# class AstraVetClassifier:
#     """LLM-based vetter using Ollama; few-shot examples from training rows."""
#     def __init__(self, config: VettingConfig):
#         self.cfg = config
#         self.client = OllamaClient(config)
#         self.few_shots: List[dict] = []

#     def fit(self, train_df: pd.DataFrame):
#         examples = []
#         for _, row in train_df.iterrows():
#             label = pseudo_label_from_row(row)
#             conf = 0.8 if label == "Planet" else 0.6
#             ex_json = {"label": label, "confidence": conf, "tests": {}, "rationale": f"Derived from disposition={row.get('disposition')}"}
#             examples.append({
#                 "lc_summary": row["lc_summary"],
#                 "lc_fields": row["lc_fields"],
#                 "label_json": ex_json
#             })
#         self.few_shots = examples

#     def predict(self, df: pd.DataFrame):
#         preds = []
#         for _, row in df.iterrows():
#             out = self.client.classify(row["lc_summary"], row["lc_fields"], few_shots=self.few_shots)
#             preds.append(out)
#         return preds


