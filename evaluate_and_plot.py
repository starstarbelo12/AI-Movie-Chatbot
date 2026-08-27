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

from src.chatbot.intent_classifier import predict_class
from src.chatbot.paths import INTENTS_PATH


def load_test_dataset():
    """Load intent patterns and normalize target class names."""
    with Path(INTENTS_PATH).open("r", encoding="utf-8") as file:
        intent_config = json.load(file)

    dataset = []
    for intent in intent_config.get("intents", []):
        tag = intent.get("tag")
        for pattern in intent.get("patterns", []):
            if isinstance(pattern, str) and pattern.strip() and tag:
                dataset.append((pattern, tag.lower()))

    if not dataset:
        raise ValueError(f"No labeled patterns found in {INTENTS_PATH}")

    intent_classes = [intent["tag"].lower() for intent in intent_config["intents"]]
    return dataset, intent_classes


TEST_DATASET, INTENT_CLASSES = load_test_dataset()


def evaluate_model(algorithm_name):
    """Evaluate accuracy, macro metrics, per-class F1, and latency."""
    y_true, y_pred, latencies = [], [], []

    print(f"Evaluating algorithm: '{algorithm_name}'...")

    for query, true_intent in TEST_DATASET:
        start_time = time.perf_counter()
        predictions = predict_class(query, algorithm=algorithm_name)
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000)
        pred_intent = predictions[0][0] if predictions else "n/a"

        y_true.append(str(true_intent).lower())
        y_pred.append(str(pred_intent).lower())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=INTENT_CLASSES, average="macro", zero_division=0
    )
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=INTENT_CLASSES, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=INTENT_CLASSES, normalize="true")

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "f1_per_class": f1_per_class,
        "cm": cm,
        "avg_latency": np.mean(latencies)
    }


# Execute evaluation
nb_res = evaluate_model("nb")
hybrid_res = evaluate_model("hybrid")

print("\n=== Summary of Results ===")
print(f"Naive Bayes -> Acc: {nb_res['acc']:.3f}, Macro F1: {nb_res['f1']:.3f}, Latency: {nb_res['avg_latency']:.2f} ms")
print(f"Hybrid Model -> Acc: {hybrid_res['acc']:.3f}, Macro F1: {hybrid_res['f1']:.3f}, Latency: {hybrid_res['avg_latency']:.2f} ms\n")

# Global style configurations
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "Arial", "font.size": 10})

# Keep all 30 labels in the evaluation, but group them in plots so that
# the figures remain readable.  These groups are operational families,
# not a reduction of the evaluated intent set.
INTENT_FAMILIES = {
    "General FAQ": [
        intent for intent in INTENT_CLASSES
        if intent.startswith("ask_")
        or intent in {"greeting", "goodbye", "search_movie"}
    ],
    "Ranking": [intent for intent in INTENT_CLASSES if intent.startswith("rank_")],
    "Comparison": [intent for intent in INTENT_CLASSES if intent.startswith("compare_")],
}

print("Intent family sizes:", {name: len(labels) for name, labels in INTENT_FAMILIES.items()})

display_classes = [i.replace("_", " ").title() for i in INTENT_CLASSES]

# Figure 1: Overall Performance Metrics Comparison
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

# Figures 2 and 3: Separate confusion matrices
# A 30 x 30 matrix is intentionally unannotated so the labels remain readable.
def save_confusion_matrix(result, title, filename):
    # Give the 30 intent labels enough room for report-sized output.
    fig, ax = plt.subplots(figsize=(24, 22), constrained_layout=True)
    sns.heatmap(result["cm"], annot=False, cmap="Blues", cbar=True,
                xticklabels=display_classes, yticklabels=display_classes,
                ax=ax, vmin=0, vmax=1)
    ax.set_title(title, fontweight="bold", fontsize=24, pad=20)
    ax.set_xlabel("Predicted", fontsize=18, labelpad=14)
    ax.set_ylabel("True", fontsize=18, labelpad=14)
    ax.tick_params(axis="x", labelrotation=90, labelsize=18, pad=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=18, pad=7)
    fig.savefig(filename, dpi=300)
    plt.close(fig)

save_confusion_matrix(nb_res, "Figure 2: Naive Bayes Confusion Matrix",
                      "Fig2_Real_Naive_Bayes_Confusion_Matrix.png")
save_confusion_matrix(hybrid_res, "Figure 3: Hybrid Model Confusion Matrix",
                      "Fig3_Real_Hybrid_Confusion_Matrix.png")

# Figures 4, 5, and 6: Separate per-intent F1 charts by family
f1_by_intent = (
    dict(zip(INTENT_CLASSES, nb_res["f1_per_class"])),
    dict(zip(INTENT_CLASSES, hybrid_res["f1_per_class"])),
)

def save_family_f1_chart(number, family_name, family_labels, filename):
    positions = np.arange(len(family_labels))
    nb_values = [f1_by_intent[0][label] for label in family_labels]
    hybrid_values = [f1_by_intent[1][label] for label in family_labels]
    family_display = [label.replace("_", " ").title() for label in family_labels]

    fig, ax = plt.subplots(figsize=(10, max(4.5, len(family_labels) * 0.48)))
    nb_bars = ax.barh(positions + width / 2, nb_values, width,
                      label="Naive Bayes", color="#9ecae1")
    hybrid_bars = ax.barh(positions - width / 2, hybrid_values, width,
                          label="Hybrid Model", color="#08519c")
    ax.set_yticks(positions)
    ax.set_yticklabels(family_display, fontsize=9)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("F1-Score")
    ax.set_title(f"Figure {number}: Per-Intent F1-Score - {family_name}",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    ax.bar_label(nb_bars, fmt="%.2f", padding=2, fontsize=7)
    ax.bar_label(hybrid_bars, fmt="%.2f", padding=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

save_family_f1_chart(4, "General FAQ", INTENT_FAMILIES["General FAQ"],
                     "Fig4_Real_F1_General_FAQ.png")
save_family_f1_chart(5, "Ranking", INTENT_FAMILIES["Ranking"],
                     "Fig5_Real_F1_Ranking.png")
save_family_f1_chart(6, "Comparison", INTENT_FAMILIES["Comparison"],
                     "Fig6_Real_F1_Comparison.png")

# Figure 7: Accuracy & Latency Trade-Off
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(9, 4))
models = ["Naive Bayes", "Hybrid Model"]
accs = [nb_res["acc"] * 100, hybrid_res["acc"] * 100]
lats = [nb_res["avg_latency"], hybrid_res["avg_latency"]]

bars1 = ax4a.bar(models, accs, color='#2171b5', width=0.4)
ax4a.set_ylabel("Accuracy (%)", fontweight='bold')
ax4a.set_ylim(80, 100)
ax4a.set_title("(a) Classification Accuracy", fontweight='bold')
ax4a.bar_label(bars1, fmt='%.1f%%', padding=3)

bars2 = ax4b.bar(models, lats, color='#9ecae1', width=0.4)
ax4b.set_ylabel("Classifier Inference Latency (ms)", fontweight='bold')
max_lat = max(lats) if lats else 1.0
ax4b.set_ylim(0, max(max_lat * 1.25, 1.0))
ax4b.set_title("(b) Average Classifier Latency", fontweight='bold')
ax4b.bar_label(bars2, fmt='%.2f ms', padding=3)

fig4.tight_layout()
fig4.suptitle("Figure 7: Accuracy and Classifier Latency Trade-Off",
              fontweight="bold")
fig4.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig("Fig7_Accuracy_Latency_Tradeoff.png", dpi=300)
plt.close()

print("Execution complete. All evaluation charts generated.")
