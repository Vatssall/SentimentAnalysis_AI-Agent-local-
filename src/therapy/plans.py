# src/therapy/plans.py
from typing import Dict, List

def suggest_tasks(signals: Dict, text: str = "") -> List[dict]:
    """
    Suggest small, concrete exercises (2–10 min) based on the latest ML signals.
    """
    tasks: List[dict] = []
    if not signals:
        return tasks

    sent = (signals.get("sentiment") or {}).get("label", "").lower()
    emo  = (signals.get("emotion") or {}).get("label", "").lower()
    mood = (signals.get("mood") or {}).get("label", "").lower()

    negative = sent in {"anxiety","stress","depression","suicidal","negative"} \
        or emo in {"fear","worry","sad","anger","shame","guilt"} \
        or mood in {"sad","fear","anger"}

    if not negative:
        return tasks

    if sent in {"anxiety","stress"} or emo in {"fear","worry"}:
        tasks.append({
            "code": "breathing",
            "title": "Deep Breathing",
            "description": "Sit comfortably. Inhale for 4, hold 4, exhale for 4 — repeat 5 times.",
            "minutes": 5
        })

    if sent == "depression" or emo in {"sad","shame"}:
        tasks.append({
            "code": "behavioral_activation",
            "title": "Micro-task Activation",
            "description": "Pick one tiny task (e.g., drink water, open window, short walk) and do it now.",
            "minutes": 5
        })

    # Optional context hook for finance/college cues in user text
    t = text.lower()
    if any(k in t for k in ["exam", "interview", "presentation", "stage"]):
        tasks.append({
            "code": "grounding_5x5",
            "title": "5-Senses Grounding",
            "description": "Name 5 things you see, 4 touch, 3 hear, 2 smell, 1 taste.",
            "minutes": 3
        })

    return tasks[:3]  # keep it short