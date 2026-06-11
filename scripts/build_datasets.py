# scripts/build_datasets.py
"""
Generate large, balanced synthetic datasets for:
  - sentiment (mental health intent)
  - emotion
  - mood

Key goals:
  * Prevent 'suicidal' false positives by:
      - enforcing suicide keywords for suicidal examples
      - guaranteeing non-suicidal examples don't contain those keywords
  * Balance classes
  * Cover common domains (finance, school, family, work, health, uncertainty, under-confidence)

Outputs:
  data/aug2/sentiment_balanced.csv
  data/aug2/emotion_balanced.csv
  data/aug2/mood_balanced.csv
"""

from __future__ import annotations
import random, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "aug2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)

# ----------------- CONFIG: sizes per class (tune if you like) -----------------
SENTIMENT_ROWS_PER_CLASS = 2000   # 6 classes -> 12k rows
EMOTION_ROWS_PER_CLASS   = 1200   # 10 classes -> 12k rows
MOOD_ROWS_PER_CLASS      = 1000   # 8 classes  -> 8k rows

# ------------------------------------------------------------------------------
# Lexicons / templates
# ------------------------------------------------------------------------------

SUICIDE_PHRASES = [
    "end my life", "kill myself", "not want to live", "take my own life",
    "better if I were gone", "no reason to live", "suicidal thoughts",
    "end it all", "I wish I were dead"
]
# Make sure non-suicidal samples never contain ANY of these:
SUICIDE_RE = re.compile("|".join(re.escape(x) for x in SUICIDE_PHRASES), re.I)

FINANCE = [
    "lost money in stocks", "credit card debt piling up", "late on rent",
    "tuition fees are due", "unexpected medical bills", "investment went wrong",
    "startup is running out of runway", "family expenses growing",
    "salary not enough this month", "loan EMIs stressing me"
]
SCHOOL = [
    "midterms approaching", "failed an assignment", "group project issues",
    "confused about my major", "finals next week", "fell behind on lectures",
    "worried about grades", "admissions essay is due", "internship interview soon"
]
WORK = [
    "project deadline tomorrow", "boss rejected my leave", "team conflict",
    "performance review soon", "stack of unread emails", "urgent production bug",
    "overtime all week", "client escalations", "fear of being laid off"
]
RELATION = [
    "argued with my mom", "friend stopped replying", "breakup last week",
    "feel judged by family", "roommate issues", "lonely after moving",
    "worry about disappointing my parents"
]
HEALTH = [
    "can't sleep lately", "heart keeps racing", "headaches every day",
    "worry about test results", "feeling exhausted", "short of breath when stressed"
]
UNCERTAINTY = [
    "unsure which path to choose", "I feel underconfident", "struggling to decide",
    "second-guessing my choices", "afraid to take the next step", "overthinking everything"
]

CONTEXTS = FINANCE + SCHOOL + WORK + RELATION + HEALTH + UNCERTAINTY

# sentiment classes (NO 'normal')
SENTIMENT_CLASSES = ["depression", "anxiety", "stress", "suicidal", "bipolar", "personality disorder"]

def make_sentences(intent: str, ctx: str) -> list[str]:
    """Generate a handful of phrasings for each intent/context pair."""
    if intent == "suicidal":
        # Always include a suicide phrase
        s = random.choice(SUICIDE_PHRASES)
        variants = [
            f"I feel overwhelmed because {ctx} and sometimes I {s}. (seeking help)",
            f"{ctx.capitalize()} makes me think I might {s}. (seeking help)",
            f"Lately, with {ctx}, I have {s}. (seeking help)",
            f"When {ctx}, I start to {s}. (seeking help)",
        ]
        return variants

    # Non-suicidal buckets (ensure no suicide phrasing sneaks in)
    if intent == "depression":
        v = [
            f"I feel empty and low for weeks; {ctx}.",
            f"Nothing seems enjoyable anymore and {ctx}.",
            f"I'm losing motivation; {ctx}.",
            f"I struggle to get out of bed because {ctx}.",
        ]
    elif intent == "anxiety":
        v = [
            f"My chest feels tight and thoughts race about {ctx}.",
            f"I worry constantly about {ctx} and can't relax.",
            f"I feel nervous and shaky whenever {ctx}.",
            f"Overthinking {ctx} makes me anxious.",
        ]
    elif intent == "stress":
        v = [
            f"I'm overwhelmed by {ctx} and can't unwind.",
            f"Pressure is building due to {ctx}.",
            f"I feel burned out from {ctx}.",
            f"I can't stop thinking about {ctx} and it's exhausting.",
        ]
    elif intent == "bipolar":
        v = [
            f"My mood swings from highly energetic to exhausted; lately {ctx}.",
            f"Some days I'm unstoppable, other days I crash; {ctx}.",
            f"Periods of high activity followed by deep lows; now {ctx}.",
            f"I oscillate between extremes; {ctx}.",
        ]
    else:  # personality disorder
        v = [
            f"I struggle with unstable relationships and self-image; {ctx}.",
            f"People say I'm intense and unpredictable; {ctx}.",
            f"I fear abandonment and react strongly when {ctx}.",
            f"My sense of self shifts quickly; {ctx}.",
        ]

    # enforce: non-suicidal strings MUST NOT contain suicide phrases
    v = [x for x in v if not SUICIDE_RE.search(x)]
    return v

# Emotion classes
EMOTION_CLASSES = ["joy","trust","sad","anger","fear","disgust","surprise","anticipation","confusion","love"]
EMO_TEMPLATES = {
    "joy":[
        "I feel happy about {ctx}",
        "Today was wonderful; I'm thrilled about {ctx}",
        "{ctx} made my day amazing",
        "I can't stop smiling after {ctx}",
    ],
    "trust":[
        "I rely on {who}; they always show up",
        "I believe {who} will help with {ctx}",
        "I put my faith in {who} about {ctx}",
        "I feel safe trusting {who}",
    ],
    "sad":[
        "I feel down thinking about {ctx}",
        "I'm heartbroken after {ctx}",
        "It's a gloomy day because of {ctx}",
        "I can't shake this sadness about {ctx}",
    ],
    "anger":[
        "I'm furious about {ctx}",
        "{ctx} really ticks me off",
        "This makes me so mad: {ctx}",
        "I'm upset and angry with {who} over {ctx}",
    ],
    "fear":[
        "I'm worried and scared about {ctx}",
        "I'm afraid {ctx} might happen",
        "{ctx} gives me chills",
        "I feel anxious about {ctx}",
    ],
    "disgust":[
        "I'm disgusted by {ctx}",
        "That was gross: {ctx}",
        "{ctx} makes my stomach turn",
        "I feel repulsed when I see {ctx}",
    ],
    "surprise":[
        "I didn't expect {ctx} at all!",
        "Wow, {ctx} was so unexpected",
        "I'm shocked by {ctx}",
        "{ctx} caught me off guard",
    ],
    "anticipation":[
        "I can't wait for {ctx}",
        "I'm looking forward to {ctx}",
        "Counting the days until {ctx}",
        "I'm excited to see how {ctx} turns out",
    ],
    "confusion":[
        "I'm unsure how to proceed with {ctx}",
        "I feel puzzled about {ctx}",
        "This is confusing: {ctx}",
        "I can't decide what to do about {ctx}",
    ],
    "love":[
        "I adore {who} so much",
        "I feel deep love for {who}",
        "My heart is full when I'm with {who}",
        "I cherish {who} with all my heart",
    ],
}
WHO = ["my mom","my dad","my best friend","my mentor","my partner","my manager","my professor","my coach","my team"]

# Mood classes
MOOD_CLASSES = ["joy","calm","sad","anger","fear","disgust","surprise","anticipation"]
MOOD_TEMPLATES = {
    "joy":[
        "Feeling cheerful after {act}",
        "I'm in a good mood thanks to {act}",
        "Lighthearted and upbeat because of {act}",
    ],
    "calm":[
        "Feeling calm and relaxed after {act}",
        "My mind is peaceful following {act}",
        "I feel centered and calm today",
    ],
    "sad":[
        "Feeling low after {act}",
        "I'm gloomy today because of {act}",
        "My mood is blue since {act}",
    ],
    "anger":[
        "I feel irritable after {act}",
        "Short-tempered today due to {act}",
        "My mood is heated because of {act}",
    ],
    "fear":[
        "I'm on edge thinking about {act}",
        "Feeling tense and nervous before {act}",
        "Restless and uneasy due to {act}",
    ],
    "disgust":[
        "Feeling grossed out after {act}",
        "That left a bad taste after {act}",
        "I'm in a sour mood from {act}",
    ],
    "surprise":[
        "I'm startled by {act}",
        "Still in shock after {act}",
        "That surprise from {act} changed my mood",
    ],
    "anticipation":[
        "I'm eager and buzzing for {act}",
        "Feeling keyed up waiting for {act}",
        "Restless with anticipation about {act}",
    ],
}
ACTS = [
    "a long walk","yoga","meditation","a tough meeting","a workout",
    "a quiet afternoon","coffee with a friend","a noisy commute",
    "a family call","late-night study","coding sprint","movie night"
]

# ------------------------------------------------------------------------------
# Generators
# ------------------------------------------------------------------------------

def gen_sentiment():
    rows = []
    for label in SENTIMENT_CLASSES:
        target = SENTIMENT_ROWS_PER_CLASS
        while len([x for x in rows if x[1]==label]) < target:
            ctx = random.choice(CONTEXTS)
            samples = make_sentences(label, ctx)
            for s in samples:
                if label != "suicidal" and SUICIDE_RE.search(s):
                    continue  # guardrail
                rows.append((s, label))
                if len([x for x in rows if x[1]==label]) >= target:
                    break
    return rows

def fill(template_list, n, **slots):
    out = []
    while len(out) < n:
        tpl = random.choice(template_list)
        text = tpl.format(**{k: random.choice(v) for k, v in slots.items()})
        # small stylistic tweaks
        if random.random() < 0.35:
            text = text.replace("I ", "I really ").replace("I'm ", "I'm really ")
        if random.random() < 0.25:
            text += random.choice([".", "!", "..."])
        out.append(text)
    return out

def gen_emotion():
    rows = []
    for lab in EMOTION_CLASSES:
        tpls = EMO_TEMPLATES[lab]
        texts = fill(tpls, EMOTION_ROWS_PER_CLASS, ctx=CONTEXTS, who=WHO)
        rows.extend((t, lab) for t in texts)
    return rows

def gen_mood():
    rows = []
    for lab in MOOD_CLASSES:
        tpls = MOOD_TEMPLATES[lab]
        texts = fill(tpls, MOOD_ROWS_PER_CLASS, act=ACTS)
        rows.extend((t, lab) for t in texts)
    return rows

# ------------------------------------------------------------------------------
# Build & Save
# ------------------------------------------------------------------------------

def save_csv(rows, path):
    df = pd.DataFrame(rows, columns=["text","label"])
    # enforce constraints again for sanity
    #  - suicidal rows must contain a suicide phrase
    #  - non-suicidal rows must NOT contain a suicide phrase
    if "sentiment" in path.name:
        df_su = df[df["label"]=="suicidal"]
        df_nsu = df[df["label"]!="suicidal"]
        assert (df_su["text"].str.contains(SUICIDE_RE)).all(), "Some suicidal rows lack suicide phrases"
        assert (~df_nsu["text"].str.contains(SUICIDE_RE)).all(), "Non-suicidal rows contain suicide phrases"
    df = df.drop_duplicates(subset=["text","label"]).sample(frac=1, random_state=42).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] wrote {path} | rows={len(df)} | classes={df['label'].nunique()}")
    print("    counts:", df['label'].value_counts().to_dict())
    return df

def main():
    s_rows = gen_sentiment()
    e_rows = gen_emotion()
    m_rows = gen_mood()

    save_csv(s_rows, OUT_DIR / "sentiment_balanced.csv")
    save_csv(e_rows, OUT_DIR / "emotion_balanced.csv")
    save_csv(m_rows, OUT_DIR / "mood_balanced.csv")

if __name__ == "__main__":
    main()