# models/ollama_client.py
import json, re, requests
from typing import Dict, List, Iterator, Tuple

class OllamaClient:
    def __init__(self, config):
        """
        config must have:
          - base_url: e.g. "http://localhost:11434"   (no /v1)
          - model: e.g. "mistral"
          - temperature, timeout (seconds), few_shot_k
        """
        self.cfg = config

    # ---- tiny helpers ----
    def _json_slice(self, text: str) -> dict:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m: raise ValueError("no JSON in output")
        return json.loads(m.group(0))

    def _fallback(self, lc_summary: str, lc_fields: dict) -> dict:
        # very small heuristic fallback
        disp = (lc_fields.get("disposition") or "").upper()
        depth = lc_fields.get("depth_fraction")
        period = lc_fields.get("period_days")
        if "CONFIRMED" in disp:
            return {"label":"Planet","confidence":0.8,"tests":{},"rationale":"Confirmed disposition"}
        if isinstance(depth, (int,float)) and depth and depth > 0.02:
            return {"label":"EclipsingBinary","confidence":0.7,"tests":{},"rationale":"Depth > 2%"}
        if period is None:
            return {"label":"VariableStar","confidence":0.6,"tests":{},"rationale":"No stable period"}
        return {"label":"Unknown","confidence":0.4,"tests":{},"rationale":"Insufficient evidence"}

    def _messages(self, lc_summary: str, lc_fields: dict) -> Tuple[str,str]:
        system = (
            "Return ONLY a JSON object with keys: label, confidence, rationale. "
            "label ∈ {Planet,EclipsingBinary,VariableStar,InstrumentalNoise,Unknown}. "
            "confidence ∈ [0,1]. No extra text."
        )
        user = (
            f"Input: {lc_summary}\n"
            f"Fields: {json.dumps({k:v for k,v in (lc_fields or {}).items() if v is not None})}\n"
            "Output JSON only."
        )
        return system, user

    # ---- SIMPLE STREAMING ----
    def classify_stream(self, lc_summary: str, lc_fields: dict) -> Iterator[dict]:
        """
        Yields:
          {'event':'delta','text': '...'}  many times
          {'event':'result','data': {...}} once at end
          (on error) also yields {'event':'error','error': '...'}
        """
        system, user = self._messages(lc_summary, lc_fields)
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role":"system","content":system},
                {"role":"user","content":user}
            ],
            "format": "json",              # ask Ollama to bias to JSON
            "stream": True,
            "options": {
                "temperature": float(self.cfg.temperature or 0.0),
                "num_predict": 160,
                "num_ctx": 1024
            }
        }

        buf = []
        try:
            # separate connect/read timeouts keeps it responsive on first load
            timeout = self.cfg.timeout if self.cfg.timeout else 30
            r = requests.post(
                f"{self.cfg.base_url}/api/chat",
                json=body, stream=False, timeout=(5, timeout)
            )
            r.raise_for_status()

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # Each line is a JSON object from Ollama
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue

                # append deltas (text tokens)
                delta = chunk.get("message", {}).get("content")
                if delta:
                    buf.append(delta)
                    yield {"event":"delta", "text": delta}

                # finish
                if chunk.get("done"):
                    full = "".join(buf).strip()
                    try:
                        data = self._json_slice(full) if full else {}
                    except Exception:
                        data = self._fallback(lc_summary, lc_fields)

                    # normalize fields
                    data.setdefault("tests", {})
                    try:
                        data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
                    except Exception:
                        data["confidence"] = 0.5

                    yield {"event":"result", "data": data}
                    return

        except Exception as e:
            yield {"event":"error","error": str(e)}
            yield {"event":"result","data": self._fallback(lc_summary, lc_fields)}

    # ---- NON-STREAM convenience ----
    def classify(self, lc_summary: str, lc_fields: dict) -> dict:
        last = None
        for ev in self.classify_stream(lc_summary, lc_fields):
            if ev.get("event") == "result":
                last = ev["data"]
        return last or self._fallback(lc_summary, lc_fields)
