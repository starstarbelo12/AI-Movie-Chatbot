# Movie Desk - Interactive Movie Chatbot

A Streamlit movie chatbot for querying movie metadata with natural-language questions. It uses two selectable intent-classification models and a multi-step movie-title matching pipeline.

## Features

- Select between Hybrid and Naive Bayes intent-classification models, with optional prediction and latency details.
- Find movie titles through exact, token-overlap, fuzzy, spell-corrected, and TF-IDF similarity matching.
- Ask for a movie's genre, rating, runtime, budget, revenue, languages, release date, production details, popularity, vote count, collection, and plot summary.
- Request multiple details in one question, such as `Titanic rating, runtime, and budget`.
- Rank movies by rating, revenue, budget, popularity, runtime, or vote count.
- Compare two or more movies, including budget-versus-revenue questions.

## Tech Stack

- Streamlit and custom CSS
- Python, Pandas, NumPy, and scikit-learn
- NLTK, WordNinja, PySpellChecker, and RapidFuzz
- Matplotlib and Seaborn for model evaluation charts

## Quick Start

### Prerequisites

- Python 3.9 or later
- Git (only if cloning the repository)

### Installation

```bash
git clone <your-repository-url>
cd AI-Movie-Chatbot
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and download the required NLTK data:

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Run the App

The required model files are included in `data/models/`, and the intent configuration is in `config/intents.json`.

```bash
streamlit run app.py
```

### Evaluate the Classifiers

```bash
python evaluate_and_plot.py
```

This generates accuracy, precision, recall, F1-score, confusion-matrix, and latency charts in the project root.

## Example Questions

```text
What is the rating and runtime of Interstellar?
Tell me the plot of Toy Story.
Compare Titanic and Avatar.
Did Barbie make back its budget?
Show the top 5 highest-rated horror movies.
```

## Project Structure

```text
AI-Movie-Chatbot/
|-- src/
|   `-- chatbot/
|       |-- __init__.py                 # Public package interface
|       |-- paths.py                    # Centralized path management
|       |-- core.py                     # Chatbot orchestration and routing
|       |-- models.py                   # Loads models, data, and intents
|       |-- text_processing.py          # Text normalization and spell correction
|       |-- movie_matching.py           # Movie title matching
|       |-- intent_classifier.py        # Intent classifier routing
|       |-- hybrid_classifier.py        # Hybrid model inference
|       |-- naive_bayes_classifier.py   # Naive Bayes model inference
|       |-- ranking.py                  # Movie ranking helpers
|       |-- comparison.py               # Movie comparison helpers
|       `-- response_formatter.py       # Response formatting
|
|-- data/
|   `-- models/
|       |-- hybrid_model.pkl
|       |-- nb_model.pkl
|       |-- words.pkl
|       |-- classes.pkl
|       `-- df_cleaned.pkl
|
|-- config/
|   `-- intents.json                    # Intent patterns and responses
|
|-- static/
|   `-- style.css                       # Streamlit styling
|
|-- app.py                              # Streamlit entry point
|-- evaluate_and_plot.py                # Evaluation and chart generation
|-- audit_intent_keywords.py            # Intent-keyword audit utility
|-- requirements.txt
`-- README.md
```
