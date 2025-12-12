import sys
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import glob
import pyarrow.parquet as pq

# --- CONFIGURACIÓN ---
INPUT_FOLDER = "embeddings_final"
MODEL_OUTPUT = "jailbreak_detector.json"
# Peso manual: 10.0 fuerza al modelo a ser muy sensible a ataques (High Recall)
# Esto reduce drásticamente los Falsos Negativos (ataques que pasan).
PARANOID_WEIGHT = 10.0 
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

# 4. Entrenamiento (XGBoost)
print(f">>> Training Classifier with PARANOID_WEIGHT={PARANOID_WEIGHT}...")
print("(This prioritizes detecting attacks over avoiding false alarms)")

model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,            # Profundidad media para evitar overfitting
    scale_pos_weight=PARANOID_WEIGHT, # <--- LA CLAVE DEL MODO PARANOICO
    tree_method='hist',     # Optimizado para CPU rápida
    eval_metric='auc',
    early_stopping_rounds=50,
    n_jobs=-1 
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# 5. Evaluación
print("\n>>> Evaluating Final Model...")
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# Reporte detallado
print(classification_report(y_test, preds, target_names=["Safe", "Jailbreak"]))
auc = roc_auc_score(y_test, probs)
print(f"AUC-ROC Score: {auc:.4f}")

# 6. Guardado Seguro
# Usamos get_booster() para compatibilidad con versiones nuevas de sklearn/xgboost
model.get_booster().save_model(MODEL_OUTPUT)
print(f"\n>>> Model saved successfully to '{MODEL_OUTPUT}'")
print(">>> Ready for testing!")