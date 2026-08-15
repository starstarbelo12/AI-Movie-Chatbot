# 🎬 Movie Desk - Interactive Movie Chatbot

An intelligent, high-performance Streamlit movie chatbot that helps users query movie metadata (ratings, budget, revenue, runtime, summaries, etc.) using natural language. 

It features a **Multi-Model Intent Classifier** (MLP vs. Naive Bayes) paired with a **Hybrid Multi-Track Entity Matching Engine** for accurate movie title resolution.

---

## ✨ Key Features

* **🤖 Dual-Model Intent Classification**:
  * **MLP (Multi-Layer Perceptron)**: Captures complex non-linear word relationships for smooth probability distributions.
  * **Naive Bayes**: Fast, frequency-based statistical classifier for baseline benchmarking.
  * Real-time model switching in UI with inference latency tracking.

* **🔍 Hybrid Multi-Track Movie Matching Engine**:
  * Multi-track matching pipeline: Exact, Token Overlap, Fuzzy (`RapidFuzz`), and TF-IDF Cosine Similarity.
  * Includes spell-checking and alias resolution for phonetic queries (e.g., Chinese Pinyin titles).

* **🛡️ Robust Logic & Hardening**:
  * Entity stripping before classification to avoid intent-entity confusion.
  * Strictly calibrated fallback logic to prevent hallucinations.

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit, Custom CSS
* **NLP & Processing**: NLTK, WordNinja, PySpellChecker, RapidFuzz
* **Machine Learning**: Scikit-Learn (MLPClassifier, MultinomialNB, TfidfVectorizer), NumPy, Pandas

---

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have Python 3.9+ installed.

### 2. Installation & Setup

Clone the repository and install required dependencies:

```bash
git clone [https://github.com/your-username/movie-chatbot.git](https://github.com/your-username/movie-chatbot.git)
cd movie-chatbot
pip install -r requirements.txt
```

Download required NLTK packages (run once):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 3. Launch the Application

Make sure all model files (`*.pkl`) and `intents.json` are in the project root directory, then run:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
AI-Movie-Chatbot/
├── src/
│   └── chatbot/              # Main application package
│       ├── __init__.py
│       ├── paths.py          # Centralized path management
│       ├── core.py           # Main chatbot orchestration
│       ├── models.py         # Load ML models & data
│       ├── text_processing.py
│       ├── movie_matching.py
│       ├── intent_classifier.py
│       ├── mlp_classifier.py
│       ├── naive_bayes_classifier.py
│       └── response_formatter.py
│
├── data/
│   └── models/               # ML models & pickled data
│       ├── mlp_model.pkl
│       ├── nb_model.pkl
│       ├── words.pkl
│       ├── classes.pkl
│       └── df_cleaned.pkl
│
├── config/                   # Configuration files
│   └── intents.json
│
├── static/                   # Static assets
│   └── style.css
│
├── app.py                    # Entry point (updated imports)
├── requirements.txt
└── README.md
```
