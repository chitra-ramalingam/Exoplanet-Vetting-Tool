import json, re, torch, threading
from typing import Dict, List, Iterator, Tuple, Optional

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TextIteratorStreamer, GenerationConfig
)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))

def _compact_fields(fields: Dict) -> Dict:
    if not isinstance(fields, dict): return {}
    out = {}
    for k, v in fields.items():
        if v is None: continue
        if isinstance(v, float): out[k] = float(f"{v:.6g}")
        else: out[k] = v
    return out

class HFClientConfig:
    def __init__(
        self,
        model_id: str = "microsoft/Phi-3.1-mini-4k-instruct",
        device_map: str = "auto",
        dtype: Optional[str] = None,        # "auto"|"float16"|"bfloat16"|None
        load_in_4bit: bool = False,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
    ):
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

class HFVettingClient:
    """
    Hugging Face transformers client with classify() and classify_stream().
    """
    def __init__(self, cfg: HFClientConfig):
        self.cfg = cfg

        # --- load tokenizer
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, trust_remote_code=True, torch_dtype=torch.float32, device_map="auto")
    
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # --- load model
        model_kwargs = dict(
            device_map=cfg.device_map,
            trust_remote_code=True,
        )
        if cfg.load_in_4bit:
            # requires bitsandbytes and a CUDA GPU
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs.pop("device_map", None)  # let bnb decide
        elif cfg.dtype:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
                "auto": "auto",
            }
            model_kwargs["torch_dtype"] = dtype_map.get(cfg.dtype, "auto")

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, trust_remote_code=True, torch_dtype=torch.float32, device_map="auto")

        self.model.eval()

        self.gen_cfg = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=(cfg.temperature > 0.0),
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )

    # ---- prompt helpers
    def _messages(self, lc_summary: str, lc_fields: Dict, few_shots: List[dict]) -> List[Dict]:
        """
        Build chat messages list in the standard {role, content} format.
        """
        sys = (
            "Return ONLY a JSON object with keys: label, confidence, rationale. "
            "label ∈ {Planet,EclipsingBinary,VariableStar,InstrumentalNoise,Unknown}. "
            "confidence ∈ [0,1]. No extra text."
        )
        msgs: List[Dict] = [{"role":"system","content":sys}]

        # keep few-shots tiny for small models
        for i, ex in enumerate(few_shots[:1]):
            msgs.append({"role":"user", "content": f"Input: {ex['lc_summary']}\nFields: {json.dumps(_compact_fields(ex['lc_fields']))}"})
            msgs.append({"role":"assistant", "content": json.dumps(ex["label_json"])})

        user = f"Input: {lc_summary}\nFields: {json.dumps(_compact_fields(lc_fields))}\nOutput JSON only."
        msgs.append({"role":"user","content": user})
        return msgs

    def _build_inputs(self, messages: List[Dict]) -> torch.Tensor:
        if hasattr(self.tok, "apply_chat_template"):
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tok([text], return_tensors="pt")
        else:
            # fallback: very simple prompt
            prompt = ""
            for m in messages:
                role = m["role"].upper()
                prompt += f"{role}: {m['content']}\n"
            prompt += "ASSISTANT:"
            inputs = self.tok([prompt], return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    # ---- final JSON classify (non-streaming)
    def classify(self, lc_summary: str, lc_fields: dict, few_shots: List[dict] = None) -> Dict:
        few_shots = few_shots or []
        messages = self._messages(lc_summary, lc_fields, few_shots)
        inputs = self._build_inputs(messages)

        with torch.no_grad():
            out = self.model.generate(**inputs, generation_config=self.gen_cfg)
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        try:
            data = _extract_json(text)
        except Exception:
            # best-effort slice
            s, e = text.find("{"), text.rfind("}")
            data = json.loads(text[s:e+1]) if s != -1 and e != -1 else {"label":"Unknown","confidence":0.4,"rationale":"parse failed"}
        data.setdefault("tests", {})
        try:
            data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except Exception:
            data["confidence"] = 0.5
        return data

    # ---- streaming tokens (yields {'event':'delta'|'result', ...})
    def classify_stream(self, lc_summary: str, lc_fields: dict, few_shots: List[dict] = None) -> Iterator[Dict]:
        few_shots = few_shots or []
        messages = self._messages(lc_summary, lc_fields, few_shots)
        inputs = self._build_inputs(messages)

        streamer = TextIteratorStreamer(self.tok, skip_special_tokens=True, skip_prompt=True)
        kwargs = dict(**inputs, streamer=streamer, generation_config=self.gen_cfg)

        # run generation in a tiny background thread (not a pool)
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        buf = []
        for piece in streamer:
            buf.append(piece)
            yield {"event":"delta", "text": piece}

        thread.join()
        full = "".join(buf).strip()
        try:
            data = _extract_json(full)
        except Exception:
            s, e = full.find("{"), full.rfind("}")
            data = json.loads(full[s:e+1]) if s != -1 and e != -1 else {"label":"Unknown","confidence":0.4,"rationale":"parse failed"}
        data.setdefault("tests", {})
        try:
            data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except Exception:
            data["confidence"] = 0.5

        yield {"event":"result", "data": data}
