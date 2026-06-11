# src/service/hf_runtime.py
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from setfit import SetFitModel
import joblib

class HFWrapper:
    def __init__(self, task_dir: Path):
        meta = joblib.load(task_dir / "meta.joblib")
        self.kind = meta["kind"]       # "setfit" or "hf"
        self.labels = meta["labels"]
        if self.kind == "setfit":
            self.model = SetFitModel.from_pretrained(str(task_dir / "setfit_model"))
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(str(task_dir / "hf_model"))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(task_dir / "hf_model"))
            self.model.eval()

    def predict(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        if self.kind == "setfit":
            try:
                probs = self.model.predict_proba(texts)
                if isinstance(probs, list): probs = np.stack(probs, axis=0)
                preds = probs.argmax(axis=1)
            except Exception:
                # fallback: compute logits then softmax
                emb = self.model.body.encode(texts)
                logits = self.model.model_head(emb)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                preds = probs.argmax(axis=1)
        else:
            with torch.no_grad():
                enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
                out = self.model(**enc)
                probs = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()
                preds = probs.argmax(axis=1)
        labels = [self.labels[i] for i in preds]
        return labels, probs