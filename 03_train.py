import sys
import time
import json
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import glob
import pyarrow.parquet as pq
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Start timing
start_time = time.time()

# --- CONFIGURACIÓN ---
INPUT_FOLDER = "embeddings_final"
MODEL_OUTPUT = "jailbreak_detector.json"
# Peso manual: 10.0 fuerza al modelo a ser muy sensible a ataques (High Recall)
# Esto reduce drásticamente los Falsos Negativos (ataques que pasan).
PARANOID_WEIGHT = 9.0 
# ---------------------

print(f">>> Loading Embeddings from {INPUT_FOLDER}...")

try:
    # 1. Cargar Datos
    parquet_files = glob.glob(f"{INPUT_FOLDER}/*.parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {INPUT_FOLDER}")

    dfs = []
    for file in parquet_files:
        if not file.endswith("_SUCCESS"):
            try:
                table = pq.read_table(file)
                df = table.to_pandas()
                dfs.append(df)
            except Exception as e:
                print(f"Skipping corrupt file {file}: {e}")

    if not dfs:
        raise FileNotFoundError("No valid parquet files found")

    combined_df = pd.concat(dfs, ignore_index=True)

    # 2. Limpieza de Datos (Solo Binario)
    # Eliminamos cualquier etiqueta que no sea 0 (Safe) o 1 (Jailbreak)
    print(f"Original shape: {combined_df.shape}")
    combined_df = combined_df[combined_df['label'].isin([0, 1])]
    print(f"Filtered shape (Binary 0/1 only): {combined_df.shape}")

    X = np.stack(combined_df["features"].values)
    y = combined_df["label"].values

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Estadísticas
jailbreak_count = sum(y)
total_count = len(y)
print(f"Jailbreak Count (Label 1): {jailbreak_count} ({jailbreak_count/total_count:.2%})")

if jailbreak_count < 10:
    print("ERROR: Not enough jailbreak samples to train!")
    sys.exit(1)

# 3. Split Data (Estratificado para mantener la proporción)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Hyperparameter Optimization with RandomizedSearchCV (faster than GridSearchCV)
print(">>> Starting Hyperparameter Optimization (RandomizedSearchCV)...")
print("(Sampling random parameter combinations with cross-validation)")

# Define parameter distributions for random sampling
param_distributions = {
    'learning_rate': uniform(0.03, 0.07),      # Uniform between 0.03 and 0.10
    'max_depth': randint(4, 8),                 # Random int between 4 and 7
    'scale_pos_weight': uniform(5.0, 8.0),      # Uniform between 5.0 and 13.0
    'n_estimators': randint(600, 1000)          # Random int between 600 and 999
}

# Base estimator with early stopping for faster training
base_model = xgb.XGBClassifier(
    tree_method='hist',
    eval_metric='auc',
    random_state=42,
    n_jobs=1  # RandomizedSearchCV handles parallelization
)

# Randomized search - much faster than exhaustive grid
N_ITER = 20  # Only test 20 random combinations (vs 81 for grid)
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    cv=3,  # 3-fold stratified cross-validation
    scoring='roc_auc',
    n_jobs=4,  # Match M4 performance cores
    verbose=2,
    refit=True,  # Refit on full training set with best params
    random_state=42
)

print(f"\n>>> Testing {N_ITER} random parameter combinations...")
print(f"    (Each tested with 3-fold cross-validation = {N_ITER * 3} total fits)")
print("    (This is ~4x faster than exhaustive grid search)")

# Fit randomized search
random_search.fit(X_train, y_train)

# Get best model and parameters
model = random_search.best_estimator_
best_params = random_search.best_params_

# Extract CV results with variance
cv_results = random_search.cv_results_
best_idx = random_search.best_index_
cv_mean = cv_results['mean_test_score'][best_idx]
cv_std = cv_results['std_test_score'][best_idx]

print(f"\n>>> Best Hyperparameters Found:")
print(f"    Learning Rate: {best_params['learning_rate']:.4f}")
print(f"    Max Depth: {best_params['max_depth']}")
print(f"    Scale Pos Weight: {best_params['scale_pos_weight']:.2f}")
print(f"    N Estimators: {best_params['n_estimators']}")
print(f"    Best CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")

# 5. Comprehensive Evaluation (Enhanced Metrics)
print("\n>>> Evaluating Final Model...")
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# Detailed classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, preds, target_names=["Safe", "Jailbreak"]))

# AUC-ROC Score
auc = roc_auc_score(y_test, probs)
print(f"\nAUC-ROC Score: {auc:.4f}")

# Confusion Matrix Analysis
print("\n=== Confusion Matrix Analysis ===")
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp} (benign flagged as jailbreak)")
print(f"False Negatives: {fn} (jailbreak missed)")
print(f"True Positives:  {tp}")
print(f"\nFalse Positive Rate: {fp/(fp+tn):.2%}")
print(f"False Negative Rate: {fn/(fn+tp):.2%}")

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_test, preds)
print(f"\nMatthews Correlation Coefficient: {mcc:.3f}")
print("(MCC is a balanced measure for imbalanced datasets: -1 to +1, +1 = perfect)")

# 6. Save Model
# Usamos get_booster() para compatibilidad con versiones nuevas de sklearn/xgboost
model.get_booster().save_model(MODEL_OUTPUT)
print(f"\n>>> Model saved successfully to '{MODEL_OUTPUT}'")

# =============================================================================
# 7. PAPER ARTIFACTS: Metrics JSON + Visualizations
# =============================================================================
print("\n>>> Generating Paper Artifacts...")

# Calculate additional metrics for paper
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision_jailbreak = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_jailbreak = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_jailbreak = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

# Save comprehensive metrics to JSON
paper_metrics = {
    "dataset": {
        "total_samples": int(total_count),
        "jailbreak_samples": int(jailbreak_count),
        "safe_samples": int(total_count - jailbreak_count),
        "jailbreak_percentage": round(jailbreak_count / total_count * 100, 2)
    },
    "train_test_split": {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_ratio": 0.2,
        "stratified": True,
        "random_state": 42
    },
    "hyperparameter_search": {
        "method": "RandomizedSearchCV",
        "n_iter": N_ITER,
        "cv_folds": 3,
        "scoring": "roc_auc",
        "best_params": {k: float(v) if isinstance(v, (int, float)) else v for k, v in best_params.items()},
        "cv_auc_mean": round(float(cv_mean), 4),
        "cv_auc_std": round(float(cv_std), 4)
    },
    "test_metrics": {
        "auc_roc": round(float(auc), 4),
        "mcc": round(float(mcc), 4),
        "accuracy": round(float(accuracy), 4),
        "precision_jailbreak": round(float(precision_jailbreak), 4),
        "recall_jailbreak": round(float(recall_jailbreak), 4),
        "f1_jailbreak": round(float(f1_jailbreak), 4),
        "false_positive_rate": round(float(fpr), 4),
        "false_negative_rate": round(float(fnr), 4)
    },
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }
}

with open("paper_metrics.json", "w") as f:
    json.dump(paper_metrics, f, indent=2)
print("    ✓ Metrics saved to 'paper_metrics.json'")

# --- Figure 1: ROC Curve ---
fpr_curve, tpr_curve, _ = roc_curve(y_test, probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr_curve, tpr_curve, 'b-', linewidth=2, label=f'XGBoost (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve: Jailbreak Detection', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ ROC curve saved to 'roc_curve.png'")

# --- Figure 2: Precision-Recall Curve ---
precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, 'g-', linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve: Jailbreak Detection', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Precision-Recall curve saved to 'precision_recall_curve.png'")

# --- Figure 3: Confusion Matrix Heatmap ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Jailbreak"])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix: Jailbreak Detection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Confusion matrix saved to 'confusion_matrix.png'")

# --- Print LaTeX Table for Paper ---
print("\n" + "="*70)
print("LATEX TABLE: Classification Metrics (copy to paper)")
print("="*70)
print(r"""
\begin{table}[h]
\centering
\caption{Classification Performance Metrics}
\begin{tabular}{|l|c|}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline""")
print(f"AUC-ROC & {auc:.3f} \\\\")
print(r"\hline")
print(f"MCC & {mcc:.3f} \\\\")
print(r"\hline")
print(f"Accuracy & {accuracy:.3f} \\\\")
print(r"\hline")
print(f"Precision (Jailbreak) & {precision_jailbreak:.3f} \\\\")
print(r"\hline")
print(f"Recall (Jailbreak) & {recall_jailbreak:.3f} \\\\")
print(r"\hline")
print(f"F1-Score (Jailbreak) & {f1_jailbreak:.3f} \\\\")
print(r"\hline")
print(f"False Positive Rate & {fpr:.3f} \\\\")
print(r"\hline")
print(f"False Negative Rate & {fnr:.3f} \\\\")
print(r"""\hline
\end{tabular}
\label{tab:metrics}
\end{table}""")

# Final timing
elapsed = time.time() - start_time
print(f"\n>>> Training completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print(">>> Optimized model ready for deployment!")