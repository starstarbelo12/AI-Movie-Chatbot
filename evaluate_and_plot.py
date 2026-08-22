import time
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# Import the project's real intent prediction function.
from src.chatbot.intent_classifier import predict_class
from src.chatbot.paths import INTENTS_PATH

# ------------------------------------------------------------------------------
# 1. Load the labeled evaluation data used by the chatbot
# ------------------------------------------------------------------------------
def load_test_dataset():
    """Use every labeled pattern in intents.json as a ground-truth example."""
    with Path(INTENTS_PATH).open("r", encoding="utf-8") as file:
        intent_config = json.load(file)

    dataset = []
    for intent in intent_config.get("intents", []):
        tag = intent.get("tag")
        for pattern in intent.get("patterns", []):
            if isinstance(pattern, str) and pattern.strip() and tag:
                dataset.append((pattern, tag))

    if not dataset:
        raise ValueError(f"No labeled patterns found in {INTENTS_PATH}")

    return dataset, [intent["tag"] for intent in intent_config["intents"]]


TEST_DATASET, INTENT_CLASSES = load_test_dataset()

# ------------------------------------------------------------------------------
# 2. Dynamic inference and evaluation function
# ------------------------------------------------------------------------------
def evaluate_model(algorithm_name):
    y_true = []
    y_pred = []
    latencies = []

    print(f"⏳ Running dynamic evaluation for algorithm: '{algorithm_name}'...")

    for query, true_intent in TEST_DATASET:
        # Measure the exact runtime in milliseconds.
        start_time = time.perf_counter()
        predictions = predict_class(query, algorithm=algorithm_name)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000

        pred_intent = predictions[0][0] if predictions else "N/A"

        y_true.append(true_intent.lower())
        y_pred.append(str(pred_intent).lower())
        latencies.append(latency_ms)

    # Calculate metrics from the live predictions.
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    # Calculate the F1 score for each intent class.
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=INTENT_CLASSES, average=None, zero_division=0
    )
    
    # Calculate the normalized confusion matrix.
    cm = confusion_matrix(y_true, y_pred, labels=INTENT_CLASSES, normalize="true")
    avg_latency = np.mean(latencies)

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "f1_per_class": f1_per_class,
        "cm": cm,
        "avg_latency": avg_latency
    }

# ------------------------------------------------------------------------------
# 3. Run the live evaluation
# ------------------------------------------------------------------------------
nb_res = evaluate_model("nb")
hybrid_res = evaluate_model("hybrid")

print("\n=== Summary of Measured Results ===")
print(f"Naive Bayes -> Acc: {nb_res['acc']:.3f}, Latency: {nb_res['avg_latency']:.2f} ms")
print(f"Hybrid Model -> Acc: {hybrid_res['acc']:.3f}, Latency: {hybrid_res['avg_latency']:.2f} ms\n")

# ------------------------------------------------------------------------------
# 4. Plot charts from the measured data
# ------------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "Arial", "font.size": 11, "figure.autolayout": True})

# --- Figure 1: Overall measured metrics comparison ---
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
nb_scores = [nb_res["acc"], nb_res["prec"], nb_res["rec"], nb_res["f1"]]
hybrid_scores = [hybrid_res["acc"], hybrid_res["prec"], hybrid_res["rec"], hybrid_res["f1"]]

x = np.arange(len(metrics))
width = 0.35

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
r1 = ax1.bar(x - width/2, nb_scores, width, label="Naive Bayes", color="#6baed6")
r2 = ax1.bar(x + width/2, hybrid_scores, width, label="Hybrid Model", color="#2171b5")

ax1.set_ylabel("Score")
ax1.set_title("Figure 1: Measured Performance Metrics Comparison", fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.set_ylim(0.0, 1.05)
ax1.legend(loc="lower right")
ax1.bar_label(r1, fmt='%.3f', padding=3, fontsize=9)
ax1.bar_label(r2, fmt='%.3f', padding=3, fontsize=9)
plt.savefig("Fig1_Real_Overall_Metrics.png", dpi=300)
plt.close()

# --- Figure 2: Measured confusion matrices ---
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 4.8))
sns.heatmap(nb_res["cm"], annot=True, fmt=".2f", cmap="Blues", cbar=False,
            xticklabels=INTENT_CLASSES, yticklabels=INTENT_CLASSES, ax=ax2a)
ax2a.set_title("(a) Measured Naive Bayes CM", fontweight='bold')
ax2a.set_xlabel("Predicted")
ax2a.set_ylabel("True")

sns.heatmap(hybrid_res["cm"], annot=True, fmt=".2f", cmap="Blues", cbar=False,
            xticklabels=INTENT_CLASSES, yticklabels=INTENT_CLASSES, ax=ax2b)
ax2b.set_title("(b) Measured Hybrid Model CM", fontweight='bold')
ax2b.set_xlabel("Predicted")

plt.savefig("Fig2_Real_Confusion_Matrices.png", dpi=300)
plt.close()

# --- Figure 3: Measured per-class F1 comparison ---
fig3, ax3 = plt.subplots(figsize=(8, 4.5))
y = np.arange(len(INTENT_CLASSES))
r3a = ax3.barh(y + width/2, nb_res["f1_per_class"], width, label="Naive Bayes", color="#9ecae1")
r3b = ax3.barh(y - width/2, hybrid_res["f1_per_class"], width, label="Hybrid Model", color="#08519c")

ax3.set_xlabel("F1-Score")
ax3.set_title("Figure 3: Measured Per-Intent F1-Score Breakdown", fontweight='bold')
ax3.set_yticks(y)
ax3.set_yticklabels([i.capitalize() for i in INTENT_CLASSES])
ax3.set_xlim(0.0, 1.05)
ax3.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=2,
    frameon=True,
)
ax3.bar_label(r3a, fmt='%.2f', padding=3, fontsize=9)
ax3.bar_label(r3b, fmt='%.2f', padding=3, fontsize=9)
fig3.subplots_adjust(bottom=0.22)
plt.savefig("Fig3_Real_Per_Intent_F1.png", dpi=300)
plt.close()

# --- Figure 4: Measured accuracy and latency ---
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(9, 4))
models = ["Naive Bayes", "Hybrid Model"]
accs = [nb_res["acc"] * 100, hybrid_res["acc"] * 100]
lats = [nb_res["avg_latency"], hybrid_res["avg_latency"]]

bars1 = ax4a.bar(models, accs, color='#2171b5', width=0.4)
ax4a.set_ylabel("Accuracy (%)", fontweight='bold')
ax4a.set_ylim(80, 100)
ax4a.set_title("(a) Classification Accuracy", fontweight='bold')
ax4a.bar_label(bars1, fmt='%.1f%%', padding=3)

bars2 = ax4b.bar(models, lats, color='#d95f02', width=0.4)
ax4b.set_ylabel("Inference Latency (ms)", fontweight='bold')
ax4b.set_ylim(0, max(lats) * 1.25 if lats else 1)
ax4b.set_title("(b) Average Inference Latency", fontweight='bold')
ax4b.bar_label(bars2, fmt='%.2f ms', padding=3)

fig4.tight_layout()
plt.savefig("Fig4_Improved_Subplots.png", dpi=300)
plt.close()

print("✅ All charts have been calculated and generated based on your actual algorithm execution results!")