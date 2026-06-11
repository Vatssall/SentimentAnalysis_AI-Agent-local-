from pathlib import Path

# Base paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "clean"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Default model choices
VECTORIZER = dict(ngram_range=(1, 2), min_df=1)
LOGREG = dict(max_iter=1000, class_weight="balanced")

# Heuristic column detection
TEXT_CANDIDATES = ["text", "sentence", "message", "content", "comment", "post", "utterance", "tweet", "review"]
LABEL_CANDIDATES = ["label", "emotion", "sentiment", "mood", "target", "category", "class", "mental_health", "status"]

# Tasks we expect to train
DEFAULT_TASKS = [
    {"csv": "mood.clean.csv", "task": "mood"},
    {"csv": "emotion.clean.csv", "task": "emotion"},
    {"csv": "sentiment.clean.csv", "task": "sentiment"},
]