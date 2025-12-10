import xgboost as xgb
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# --- CONFIG ---
# Point this to the folder created by Step 2
INPUT_FOLDER = "embeddings_10k_test"
# ----------------

print(f">>> Loading Embeddings from {INPUT_FOLDER}...")
try:
    import glob
    parquet_files = glob.glob(f"{INPUT_FOLDER}/*.parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {INPUT_FOLDER}")
    table = pq.read_table(parquet_files[0])  # Read the first (and likely only) file
    # Extract embeddings and labels from Ray's tensor format
    X = np.stack(table["embedding"].to_numpy())  # Stack into 2D array
    y = table["label"].to_numpy()
except Exception as e:
    print(f"ERROR: Could not find data in {INPUT_FOLDER}. Did Step 2 finish?")
    raise e

print(f"Data Shape: {X.shape}")
print(f"Jailbreak Count: {sum(y)} ({sum(y)/len(y):.2%})")

# 1. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Calculate Class Weight (Crucial for Imbalance)
# If 0 jailbreaks exist in the sample (possible in small 10k tests), default to 1.0
if sum(y) > 0:
    ratio = (len(y) - sum(y)) / sum(y)
else:
    ratio = 1.0
    print("WARNING: No jailbreaks found in this small sample. Ratio set to 1.0")

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