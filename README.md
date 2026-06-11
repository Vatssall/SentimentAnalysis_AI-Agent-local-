<<<<<<< HEAD
# SentimentAnalysis_AI-Agent-local-
AI-powered mental health companion that detects sentiment, emotions, and mood in real time, provides empathetic conversations using Llama 3, suggests therapy-inspired exercises for stress and anxiety, and maintains session memory—all in a privacy-first, locally hosted environment.
=======


# Sentiment-Detection Therapist AI Agent

An AI-powered empathetic agent that detects **sentiment, emotion, and mood** from user inputs and responds like a supportive therapist.  
If negative signals such as **depression, anxiety, or stress** are detected, the agent suggests actionable **coping strategies** (breathing exercises, grounding, journaling, etc).  

Built with:
-  **Python + FastAPI** backend  
-  **Ollama (Llama3)** for natural conversation  
-  **Custom ML models (HuggingFace + scikit-learn)** for sentiment/emotion/mood classification  
-  **SQLite + SQLAlchemy** for memory and user sessions  
-  **Custom UI (HTML/CSS/JS)** inspired by ChatGPT  

---

##  Features
- **Emotion & Sentiment Detection**: Multi-label ML classifiers trained on custom datasets.  
- **Therapy-Like Responses**: AI suggests small actionable tasks when negative emotions appear.  
- **User Accounts & Sessions**: Login, register, and manage chat histories.  
- **Memory**: Rolling conversation memory + user-specific emotional state tracking.  
- **Modern UI**: Dark theme, chips for signals, session sidebar, smooth chat.  
- **Privacy First**: Runs entirely **locally** on your machine (Mac M3 in our case).  

---

##  Project Structure

ml-agent/
├── src/
│   ├── service/        # FastAPI app (api.py, auth, db, models_registry)
│   ├── models/         # Training scripts (train.py, train_hf.py)
│   ├── data/           # CSV datasets + cleaning scripts
│   └── static/         # styles.css, chat.js
│   └── templates/      # chat.html, login.html
├── models/             # Saved models (sentiment, emotion, mood)
├── reports/            # Training reports
└── README.md

---

##  Installation

### 1. Clone repo
```bash
git clone https://github.com/YOUR_USERNAME/ml-agent.git
cd ml-agent

2. Create environment

python -m venv venv
source venv/bin/activate   # macOS/Linux

3. Install dependencies

pip install -r requirements.txt

4. Train models (optional, already trained included)

python -m src.data.clean_and_standardize
python -m src.models.train_hf --csv data/clean/sentiment.clean.csv --task sentiment

5. Run server

uvicorn src.service.api:app --reload --port 8000

6. Access UI

Open http://127.0.0.1:8000/ui/login
	•	Register / Login
	•	Start chatting 

⸻

Example Use Cases

Positive

“I just got the job I was dreaming about! I’m so excited!”

	•	Sentiment: joy
	•	Emotion: happiness
	•	Mood: elated
	•	Agent: Congratulations! That’s amazing news…

Negative

“I feel my legs shake when I’m in a tough situation like an interview.”

	•	Sentiment: anxiety
	•	Emotion: fear
	•	Mood: nervous
	•	Agent: I sense that you’re anxious. Try this breathing exercise…

⸻

⚠️ Disclaimer

This project is intended for educational and research purposes only.
It is not a substitute for professional medical or psychological help.

⸻

 Authors
	•	Vatsal Shukla (Developer)

---

# 🎥 Demo Flow Script (for presentation)

### 🎬 Intro
- *“Welcome, this is our Sentiment-Detection Therapist AI Agent — a local AI tool that not only chats with you like ChatGPT but also tracks your emotional state and gives therapy-inspired suggestions.”*

---

### Step 1 → Login
- Open `http://127.0.0.1:8000/ui/login`  
- Register a new account  
- Show sidebar with sessions  

---

### Step 2 → Positive Case
Prompt:  
> “I just got the job I was dreaming about! I’m so excited!”  

**Expected Demo**:
- Chips: `joy / happiness / elated`  
- Assistant: Congratulates + encourages  

---

### Step 3 → Stress/Anxiety Case
Prompt:  
> “I feel my legs shake when I’m in a tough situation like an interview.”  

**Expected Demo**:
- Chips: `anxiety / fear / nervous`  
- Assistant: Reassures + gives **breathing exercise**  

---

### Step 4 → Depression Case
Prompt:  
> “I’ve been feeling sad and unmotivated for the past week.”  

**Expected Demo**:
- Chips: `depression / sadness / low`  
- Assistant: Offers **small actionable task** (journaling, walk, grounding exercise)  

---

### Step 5 → Gratitude Case
Prompt:  
> “I’m grateful for my friends, they really support me.”  

**Expected Demo**:
- Chips: `gratitude / joy / calm`  
- Assistant: Reinforces positive feelings  

---

### Step 6 → Session History
- Show how chat history + signals are saved per session  
- Show switching between past sessions  

---

### 
*“This project shows how AI can be more than a chatbot — it can detect your emotional state, remember your context, and provide supportive therapy-inspired guidance. While it’s not a replacement for real therapy, it’s a step towards empathetic AI companions.”*

---
>>>>>>> Initial commit
