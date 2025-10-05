# models/gemini_client.py
import json, re
from typing import Dict, List, Tuple, Optional

# models/gemini_client.py
from google import genai
import json, re
from google.genai import types

class GeminiClient:
    def __init__(self, config):
        # config: {"api_key": "...", "model": "gemini-2.0-flash-001", "temperature": 0.2}
        self.cfg = config
        self.client = genai.Client(api_key="")
        self.model='gemini-2.5-flash'

    def classify(self, lc_summary: str, lc_fields: dict) -> dict:
        system_prompt = (
            "You are an exoplanet light-curve vetter. "
            "Return ONLY one JSON object with keys: "
            'label (str), confidence (0..1 float), tests (object), rationale (str). '
            "Labels: Planet, False Positive, Eclipsing Binary, Unknown."
        )
        user_prompt = (
            "Light-curve summary:\n"
            f"{lc_summary}\n\n"
            "Parsed fields (JSON):\n"
            f"{json.dumps(lc_fields, ensure_ascii=False)}\n\n"
            "Return the JSON now."
        )
        schema_hint = (
            'Example shape: {"label":"Planet","confidence":0.82,'
            '"tests":{"odd_even_depth":"pass","secondary_eclipse":"fail"},'
            '"rationale":"Short, specific justification."}'
        )
        return self.complete_json(system_prompt, user_prompt, schema_hint)
    
    def complete_json(self, system_prompt: str, user_prompt: str, schema_hint=None) -> dict:
        system_prompt = "Strictly Reply only in JSON. in this format. IF you cannot find information set confidence as 0.  Should always start with paranthesis, \r\n" 
        system_prompt += """{
            "type": "OBJECT",
            "required": ["label","confidence","tests","rationale"],
            "properties": {
                "label": {"type": "STRING"},
                "confidence": {"type": "NUMBER"},
                "tests": {"type": "OBJECT"},
                "rationale": {"type": "STRING"},
            },
        }
    """
        resp = self.client.models.generate_content(
         model=self.model,
              contents=[system_prompt, user_prompt],
             )
        text = resp.text
        text = text[text.find("{"):text.rfind("}")+1]  # crude JSON extraction
        try:
            json = json.loads(text)
            
        except:
            return {"label":"Unknown","confidence":0.0,"tests":{},"rationale":"Could not parse LLM output."}
        return json
