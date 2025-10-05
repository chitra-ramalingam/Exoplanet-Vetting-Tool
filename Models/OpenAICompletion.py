# openai_client.py
from openai import OpenAI
import json
from typing import List, Dict, Tuple
from Core.Constants import Constants
class OpenAICompletion:
    def __init__(self):
        # Ollama's OpenAI-compatible endpoint (no real key needed)
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.VETTING_SCHEMA = {
            "type": "object",
            "properties": {
                "label": {"enum": ["Planet","EclipsingBinary","VariableStar","InstrumentalNoise","Unknown"]},
                "confidence": {"type":"number"},
                "tests": {"type":"object"},
                "rationale": {"type":"string"}
            },
            "required": ["label","confidence","rationale"]
        }

    def _process_single(self, model: str, sys_msg: str, lc_summary: str, features: Dict, tests: Dict) -> Dict:
        user_msg = (
            f"Schema: {json.dumps(self.VETTING_SCHEMA)}\n"
            f"Light curve summary: {lc_summary}\n"
            f"Top features: {json.dumps(features)}\n"
            f"Checks: {json.dumps(tests)}\n"
            "Return JSON only."
        )
        sys = Constants.SystemMessage
        try:
            print (f"Calling {model}.. {user_msg}.")
            resp = self.client.chat.completions.create(
                model='phi3.5',
                messages=[{"role": "system", "content": sys},
                          {"role": "user",   "content": "WHat is my name"}],
                temperature=0.2,
            )
            txt = resp.choices[0].message.content
            try:
                out = json.loads(txt)
            except Exception:
                start, end = txt.find("{"), txt.rfind("}")
                out = json.loads(txt[start:end+1])
            # normalize missing fields
            out.setdefault("tests", {})
            return out
        except Exception as e:
            return {"label": "Unknown", "confidence": 0.3, "tests": {}, "rationale": f"Error: {e}"}

    # === BATCH (sequential) ===
    def CallCompletionBatch(self, model: str, sys_msg: str, items: List[Dict]) -> List[Dict]:
        results = []
        for item in items:
            res = self._process_single(
                model, sys_msg,
                item.get("lc_summary", ""),
                item.get("features", {}),
                item.get("tests", {})
            )
            results.append(res)
        return results

    # === STREAM (sequential) ===
    def CallCompletionStream(self, model: str, sys_msg: str, items: List[Dict]):
        """
        Generator: yields (row_index, result_dict) IN INPUT ORDER.
        No threads, each item is processed and yielded immediately.
        """
        for i, item in enumerate(items):
            idx = item.get("row_index", i)
            res = self._process_single(
                model, sys_msg,
                item.get("lc_summary", ""),
                item.get("features", {}),
                item.get("tests", {})
            )
            yield idx, res
