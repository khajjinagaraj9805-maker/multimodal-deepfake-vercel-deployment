
from datasets import load_dataset
from modules.text_detector import detect_text
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

print("Loading MDAIGT dataset...")
dataset = load_dataset("CogniSAL/MDAIGT")

print("Available splits:", list(dataset.keys()))

split_name = "train_news"

if split_name not in dataset:
    raise ValueError(
        f"Expected '{split_name}' split. "
        f"Available splits: {list(dataset.keys())}"
    )

data = dataset[split_name]

print("Columns:", data.column_names)
print("Rows available:", len(data))

# Select a reproducible sample of 1,000 rows.
sample_count = min(1000, len(data))
data = data.shuffle(seed=42).select(range(sample_count))

# --------------------------------------------------
# 2. Run predictions and save AI probabilities
# --------------------------------------------------

y_true = []
ai_scores = []

for i, row in enumerate(data, start=1):
    text = str(row["text"])
    true_label = str(row["label"]).strip().lower()

    if true_label not in ("human", "machine"):
        continue

    if not text.strip():
        continue

    result = detect_text(text)

    actual = 1 if true_label == "machine" else 0
    ai_probability = float(result["ai_probability"]) / 100.0

    y_true.append(actual)
    ai_scores.append(ai_probability)

    if i % 20 == 0 or i == sample_count:
        print(f"Evaluated {i}/{sample_count}")

if not y_true:
    raise ValueError(
        "No valid labelled examples found. Check the dataset labels."
    )

# --------------------------------------------------
# 3. Evaluate current 50% threshold
# --------------------------------------------------

baseline_threshold = 0.50

y_pred = [
    1 if score >= baseline_threshold else 0
    for score in ai_scores
]

print("\n" + "=" * 50)
print("BASELINE EVALUATION")
print("=" * 50)

print("Evaluated samples:", len(y_true))
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Human", "AI"],
        zero_division=0,
        digits=4,
    )
)

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

# --------------------------------------------------
# 4. Compare different thresholds
# --------------------------------------------------

thresholds = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]

print("\n" + "=" * 90)
print("THRESHOLD COMPARISON")
print("=" * 90)

print(
    f"{'Threshold':>10} | "
    f"{'Accuracy':>8} | "
    f"{'AI Precision':>12} | "
    f"{'AI Recall':>9} | "
    f"{'AI F1':>7} | "
    f"{'Human FPR':>9}"
)
print("-" * 90)

for threshold in thresholds:
    threshold_predictions = [
        1 if score >= threshold else 0
        for score in ai_scores
    ]

    accuracy = accuracy_score(y_true, threshold_predictions)
    ai_precision = precision_score(
        y_true, threshold_predictions, pos_label=1, zero_division=0
    )
    ai_recall = recall_score(
        y_true, threshold_predictions, pos_label=1, zero_division=0
    )
    ai_f1 = f1_score(
        y_true, threshold_predictions, pos_label=1, zero_division=0
    )

    # Human false-positive rate:
    # proportion of actual human samples incorrectly classified as AI.
    human_indices = [
        index for index, actual in enumerate(y_true)
        if actual == 0
    ]

    human_false_positives = sum(
        threshold_predictions[index] == 1
        for index in human_indices
    )

    human_fpr = (
        human_false_positives / len(human_indices)
        if human_indices else 0.0
    )

    print(
        f"{threshold:10.2f} | "
        f"{accuracy:8.4f} | "
        f"{ai_precision:12.4f} | "
        f"{ai_recall:9.4f} | "
        f"{ai_f1:7.4f} | "
        f"{human_fpr:9.4f}"
    )

# --------------------------------------------------
# 5. Detailed report for each threshold
# --------------------------------------------------

print("\n" + "=" * 50)
print("DETAILED THRESHOLD REPORTS")
print("=" * 50)

for threshold in thresholds:
    threshold_predictions = [
        1 if score >= threshold else 0
        for score in ai_scores
    ]

    print(f"\n--- Threshold: {threshold:.2f} ---")
    print(
        "Accuracy:",
        round(accuracy_score(y_true, threshold_predictions), 4),
    )
    print(
        classification_report(
            y_true,
            threshold_predictions,
            labels=[0, 1],
            target_names=["Human", "AI"],
            zero_division=0,
            digits=4,
        )
    )
    print(
        "Confusion matrix (rows=actual, columns=predicted):"
    )
    print(
        confusion_matrix(
            y_true, threshold_predictions, labels=[0, 1]
        )
    )

print("\nEvaluation completed.")
print(
    "Note: thresholds were compared on the same evaluation sample. "
    "Use a separate validation set before selecting a final threshold."
)
