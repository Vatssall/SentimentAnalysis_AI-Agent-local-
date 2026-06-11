# src/service/models_registry.py
from pathlib import Path
import joblib
from ..config import MODEL_DIR
from .hf_runtime import HFWrapper

_MODELS = {}  # task -> {"hf": HFWrapper} or {"skl": bundle}

def load_models():
    global _MODELS
    _MODELS = {}
    for sub in MODEL_DIR.glob("*"):
        meta_p = sub / "meta.joblib"
        skl_p  = sub / "model.joblib"
        if meta_p.exists():
            meta = joblib.load(meta_p)
            _MODELS[meta["task"]] = {"hf": HFWrapper(sub), "meta": meta}
        elif skl_p.exists():
            bundle = joblib.load(skl_p)
            _MODELS[bundle["meta"]["task"]] = {"skl": bundle, "meta": bundle["meta"]}
    return list(_MODELS.keys())

def get(task: str):
    return _MODELS.get(task)

def available():
    return list(_MODELS.keys())