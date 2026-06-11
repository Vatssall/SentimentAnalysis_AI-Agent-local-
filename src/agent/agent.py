# src/agent/agent.py
import json, re, requests, sys
from ollama import chat

API_BASE = "http://127.0.0.1:8000"  # your FastAPI

LLM_MODEL = "llama3"                # already pulled via ollama

# heuristics to detect explicit classification asks
CLASSIFY_HINTS = re.compile(
    r"\b(classif(y|y)|label|detect|is this (positive|negative)|emotion|mood|sentiment)\b",
    re.IGNORECASE
)

def call_api(task: str, text: str):
    r = requests.post(f"{API_BASE}/predict/{task}", json={"text": text}, timeout=10)
    r.raise_for_status()
    return r.json()  # {task,label,proba}

def classify_all(text: str):
    results = {}
    for task in ("sentiment", "emotion", "mood"):
        try:
            results[task] = call_api(task, text)
        except Exception as e:
            results[task] = {"task": task, "label": "unavailable", "proba": 0.0, "error": str(e)}
    return results

def style_directives(signals: dict) -> str:
    """Turn labels into guidance for the LLM."""
    emo = signals.get("emotion", {}).get("label", "")
    mood = signals.get("mood", {}).get("label", "")
    senti = signals.get("sentiment", {}).get("label", "")

    tone = []
    if senti in ("suicidal", "depression", "anxiety", "stress", "bipolar", "personality disorder"):
        tone.append("be calm, supportive, non-judgmental; suggest professional help resources if user asks; avoid clinical claims.")
    elif senti == "normal":
        tone.append("use a neutral, helpful tone.")
    if emo in ("sadness", "loneliness", "regret", "shame"):
        tone.append("acknowledge feelings, be empathetic, avoid toxic positivity.")
    if emo in ("joy", "gratitude", "pride", "love", "relief"):
        tone.append("be warm and encouraging; mirror positive affect briefly.")
    if mood in ("anger", "contempt", "disgust"):
        tone.append("de-escalate, validate frustration, avoid blame.")
    if not tone:
        tone.append("default to concise, helpful tone.")

    return " ".join(tone)

SYSTEM_TEMPLATE = """You are a helpful assistant that uses *provided ML signals* to adapt responses.

ML signals (JSON):
{signals}

Instructions:
- If the user explicitly asks to classify/label (sentiment, emotion, mood), return ONLY a compact JSON:
  {{"sentiment":"...", "emotion":"...", "mood":"...", "notes":"<optional>"}}
- Otherwise, use the signals to guide tone and content. Do not overrule the labels.
- Be concise. Avoid medical/diagnostic claims.
- If signals are unavailable, proceed normally.

Adaptive style guidance:
{style}
"""

def answer_with_llm(user_text: str, signals: dict, style: str):
    sys_prompt = SYSTEM_TEMPLATE.format(
        signals=json.dumps(signals, ensure_ascii=False),
        style=style
    )
    resp = chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text}
        ],
        options={"temperature": 0.3}  # steadier output
    )
    return resp["message"]["content"].strip()

def main():
    print("Agent ready. Type your message. Ctrl+C to exit.")
    for line in sys.stdin:
        user = line.strip()
        if not user:
            continue

        # 1) run your classifiers (customization happens here)
        signals = classify_all(user)

        # 2) if user asked to classify, return labels only
        if CLASSIFY_HINTS.search(user):
            out = {
                "sentiment": signals.get("sentiment", {}).get("label", "unavailable"),
                "emotion":   signals.get("emotion", {}).get("label",   "unavailable"),
                "mood":      signals.get("mood", {}).get("label",      "unavailable"),
                "notes": "probas: " + json.dumps({
                    k: round(v.get("proba", 0.0), 3) for k, v in signals.items() if isinstance(v, dict)
                })
            }
            print(json.dumps(out, ensure_ascii=False))
            continue

        # 3) otherwise, guide llama3 using the signals
        style = style_directives(signals)
        reply = answer_with_llm(user, signals, style)
        print(reply)

if __name__ == "__main__":
    main()