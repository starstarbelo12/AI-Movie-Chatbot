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
  * **Track 1**: Exact string & accent normalization matching.
  * **Track 2**: Token overlap & Stemming matching.
  * **Track 3**: Fuzzy matching powered by `RapidFuzz`.
  * **Track 4**: Sub-word Char-WB TF-IDF Vector Cosine Similarity.
  * Includes `WordNinja` segmentation and custom domain spell-checking.

* **🛡️ Robust Logic & Hardening**:
  * Title-word stripping before intent classification to avoid entity-intent collision.
  * Calibrated thresholds to prevent false positives/hallucinations (explicit "Not Found" handling).

* **🎨 Sleek Streamlit UI**:
  * Embedded bottom-fixed chat input layout.
  * Expandable intent confidence inspection panel (Predict Score).

---

## 🛠️ Tech Stack

* **Frontend / Framework**: Streamlit, Custom CSS
* **NLP & Processing**: NLTK, WordNinja, PySpellChecker, RapidFuzz
* **Machine Learning & Vectorization**: Scikit-Learn (MLPClassifier, MultinomialNB, TfidfVectorizer), NumPy, Pandas

---

---

## 📊 Model Comparison (MLP vs. Naive Bayes)

| Feature                | Naive Bayes (`nb`)                        | Multi-Layer Perceptron (`mlp`)             |
|  :---                  |                  :---                     |                       :---                 |
| **Type**               | Statistical Probabilistic Model           | Artificial Neural Network                  |
| **Probability Output** | Extreme / Polarized (e.g., 98.4% vs 0.1%) | Smooth / Calibrated (e.g., 41.3% vs 35.8%) |
| **Ambiguity Handling** | Overconfident on single key terms         | Evaluates competing context keywords       |
| **Inference Speed**    | Ultra Fast                                | Extremely Fast                             |

---

---

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have Python 3.9+ installed.

### 2. Installation

Clone the repository and install required dependencies:

```
git clone https://github.com/your-username/movie-chatbot.git
cd movie-chatbot
pip install -r requirements.txt
```

### 3. Launch the Application

```
streamlit run app.py
```
