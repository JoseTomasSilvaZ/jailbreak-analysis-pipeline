import sys
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# --- CONFIG ---
# Point this to the folder created by Step 2
INPUT_FOLDER = "embeddings_final"
# ----------------

print(f">>> Loading Embeddings from {INPUT_FOLDER}...")
try:
    import glob
    import pyarrow.parquet as pq

    # Read all parquet files and combine them
    parquet_files = glob.glob(f"{INPUT_FOLDER}/*.parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {INPUT_FOLDER}")

    # Read all files into a list of dataframes, then concatenate
    dfs = []
    for file in parquet_files:
        if not file.endswith("_SUCCESS"):  # Skip metadata file
            table = pq.read_table(file)
            df = table.to_pandas()
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No valid parquet files found")

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Extract embeddings and labels
    X = np.stack(combined_df["features"].values)
    y = combined_df["label"].values

except Exception as e:
    print(f"ERROR: Could not find data in {INPUT_FOLDER}. Did Step 2 finish?")
    raise e

print(f"Data Shape: {X.shape}")
print(f"Jailbreak Count: {sum(y)} ({sum(y)/len(y):.2%})")

# 1. Split Data
# Use stratified split only if we have enough samples of both classes
min_class_count = min(sum(y), len(y) - sum(y))
if min_class_count >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
else:
    print("WARNING: Not enough samples in minority class for stratified split. Using random split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# 2. Calculate Class Weight (Crucial for Imbalance)
# If 0 jailbreaks exist in the sample, we cannot train a classifier
if sum(y) == 0:
    print("ERROR: No positive samples (jailbreaks) found in the dataset!")
    print("This usually means the labeling in 01_ingest.py failed.")
    print("Please check the ingestion pipeline and re-run.")
    sys.exit(1)

ratio = (len(y) - sum(y)) / sum(y)

print(f"Using scale_pos_weight: {ratio:.2f}")

# 3. Train XGBoost
print(">>> Training Classifier on M4 CPU...")
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=ratio, 
    tree_method='hist', 
    eval_metric='auc',
    early_stopping_rounds=50,
    n_jobs=-1 
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# 4. Evaluate
print(">>> Evaluating...")
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# Handle edge case where sample is too small to have both classes
try:
    print(classification_report(y_test, preds))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, probs):.4f}")
except Exception as e:
    print("Metrics skipped (Sample might be too small/homogenous).")

# Save model
model.save_model("jailbreak_detector.json")
print(">>> Model saved to 'jailbreak_detector.json'")