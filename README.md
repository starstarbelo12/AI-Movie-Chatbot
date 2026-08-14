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
├── app.py              # Main Streamlit UI entry point
├── chatbot.py          # Core chatbot logic & inference pipeline
├── style.css           # Custom UI styling
├── intents.json        # Intent training patterns and responses
├── df_cleaned.pkl      # Cleaned movie database
├── mlp_model.pkl       # Trained MLP model checkpoint
├── nb_model.pkl        # Trained Naive Bayes model checkpoint
├── words.pkl           # Trained vocabulary dictionary
├── classes.pkl         # Intent class tags
└── requirements.txt    # Python package dependencies
```
