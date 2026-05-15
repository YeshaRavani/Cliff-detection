# baselineclassifier.py

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


TRAIN_PATH = "train_pairs.csv"
VAL_PATH = "val_pairs.csv"
TEST_PATH = "test_pairs.csv"

OUTPUT_DIR = "baseline_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMILES_COL_1 = "smiles_1"
SMILES_COL_2 = "smiles_2"
TARGET_COL = "label"


# Load CSV files
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train columns:", list(train_df.columns))


# Create pair-wise SMILES text input
train_df["smiles_pair"] = train_df[SMILES_COL_1].astype(str) + " [SEP] " + train_df[SMILES_COL_2].astype(str)
val_df["smiles_pair"] = val_df[SMILES_COL_1].astype(str) + " [SEP] " + val_df[SMILES_COL_2].astype(str)
test_df["smiles_pair"] = test_df[SMILES_COL_1].astype(str) + " [SEP] " + test_df[SMILES_COL_2].astype(str)


X_train = train_df["smiles_pair"]
y_train = train_df[TARGET_COL].astype(int)

X_val = val_df["smiles_pair"]
y_val = val_df[TARGET_COL].astype(int)

X_test = test_df["smiles_pair"]
y_test = test_df[TARGET_COL].astype(int)


print("\nClass distribution:")
print("Train:\n", y_train.value_counts())
print("Validation:\n", y_val.value_counts())
print("Test:\n", y_test.value_counts())


# Convert SMILES pair strings into character n-gram TF-IDF features
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

print("\nTF-IDF train shape:", X_train_vec.shape)


# Train simple baseline classifier
clf = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="liblinear"
)

clf.fit(X_train_vec, y_train)


def evaluate_split(name, X_vec, y_true):
    y_pred = clf.predict(X_vec)
    y_prob = clf.predict_proba(X_vec)[:, 1]

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{name} Results")
    print("-" * 35)
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return y_pred, y_prob, {
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


val_pred, val_prob, val_metrics = evaluate_split("Validation", X_val_vec, y_val)
test_pred, test_prob, test_metrics = evaluate_split("Test", X_test_vec, y_test)


# ROC curve
RocCurveDisplay.from_predictions(y_test, test_prob)
plt.title("ROC Curve - SMILES Pair Baseline")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=300)
plt.close()


# Precision-Recall curve
PrecisionRecallDisplay.from_predictions(y_test, test_prob)
plt.title("Precision-Recall Curve - SMILES Pair Baseline")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"), dpi=300)
plt.close()


# Confusion matrix
cm = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - SMILES Pair Baseline")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()


# Save metrics
metrics_df = pd.DataFrame([test_metrics])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_metrics.csv"), index=False)


print("\nFinal Test Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nGraphs and metrics saved in:", OUTPUT_DIR)