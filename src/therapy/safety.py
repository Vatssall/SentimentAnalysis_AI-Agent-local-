import re
SUICIDE_RE = re.compile(r"(kill myself|end my life|suicidal|don’t want to live|don't want to live|take my own life|end it all|wish I were dead)", re.I)

def detect_risk(text: str, signals: dict) -> bool:
    if SUICIDE_RE.search(text or ""):
        return True
    lab = (signals.get("sentiment") or {}).get("label","")
    proba = float((signals.get("sentiment") or {}).get("proba") or 0)
    return lab == "suicidal" and proba >= 0.65