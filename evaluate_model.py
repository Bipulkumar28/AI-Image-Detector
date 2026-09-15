import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_fscore_support
)

# Define directories and parameters
BASE_DIR = os.path.join("dataset", "processed")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_PATH = os.path.join("models", "best_discriminator_model.keras")
OUTPUT_DIR = "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("=== STEP 11: Loading Saved Model & Test Dataset ===")

# Load best saved model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Ensure training completed successfully.")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model successfully loaded from '{MODEL_PATH}'.")

# Load held-out test set (shuffle MUST be False to align predictions with true labels)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

class_names = test_ds.class_names
print(f"Detected Class Mapping: {class_names}")  # Should show ['ai', 'real'] or ['0', '1']

# Extract ground truth labels
y_true = []
for images, labels in test_ds:
    y_true.extend(labels.numpy().flatten())
y_true = np.array(y_true)

print("\nRunning model inference on unseen test set...")
# Predict probabilities (0.0 to 1.0)
y_pred_probs = model.predict(test_ds, verbose=1).flatten()

# Convert probabilities to binary predictions (Threshold = 0.5)
y_pred = (y_pred_probs >= 0.5).astype(int)

# --- 1. PRINT CLASSIFICATION REPORT ---
print("\n" + "="*55)
print("             DETAILED CLASSIFICATION REPORT             ")
print("="*55)
report = classification_report(y_true, y_pred, target_names=['AI-Generated (0)', 'Real Image (1)'], digits=4)
print(report)

# --- 2. GENERATE CONFUSION MATRIX ---
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix Raw Values:")
print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=['AI-Generated', 'Real'], 
    yticklabels=['AI-Generated', 'Real'],
    annot_kws={"size": 14}
)
plt.title("Confusion Matrix — AI vs Real Discriminator", fontsize=14, pad=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
cm_plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"Confusion Matrix saved to '{cm_plot_path}'.")

# --- 3. GENERATE ROC-AUC CURVE ---
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=12)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
roc_plot_path = os.path.join(OUTPUT_DIR, "roc_auc_curve.png")
plt.savefig(roc_plot_path, dpi=300)
plt.close()
print(f"ROC-AUC Curve saved to '{roc_plot_path}'.")

# --- 4. SUMMARY STATS ---
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
accuracy = np.mean(y_true == y_pred)

print("\n" + "="*55)
print("               SUMMARY PERFORMANCE METRICS             ")
print("="*55)
print(f"Overall Test Accuracy : {accuracy * 100:.2f}%")
print(f"Precision             : {precision * 100:.2f}%")
print(f"Recall                : {recall * 100:.2f}%")
print(f"F1-Score              : {f1 * 100:.2f}%")
print(f"ROC-AUC Score         : {roc_auc:.4f}")
print("="*55 + "\n")