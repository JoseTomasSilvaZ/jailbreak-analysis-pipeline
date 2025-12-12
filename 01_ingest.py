import os
import sys
import pandas as pd
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("JailbreakIngestion_BigData") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

print(">>> Loading Datasets...")

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

# --- DATASET 1: Jayavibhav (El original) ---
print("1. Downloading jayavibhav/prompt-injection-safety...")
ds1 = load_dataset("jayavibhav/prompt-injection-safety", split="train")
df1 = ds1.to_pandas()
# Aseguramos nombres de columnas estándar
df1 = df1[["text", "label"]] 

# --- DATASET 2: Imoxto (El nuevo limpio) ---
print("2. Downloading imoxto/prompt_injection_cleaned_dataset-v2...")
# Nota: A veces los datasets tienen splits diferentes, probamos 'train'
ds2 = load_dataset("imoxto/prompt_injection_cleaned_dataset-v2", split="train")
df2 = ds2.to_pandas()
# Renombrar columnas si es necesario (inspecciona si falla)
# Asumimos que también trae 'text' y 'label'. Si trae 'prompt', cambiamos:
if 'prompt' in df2.columns:
    df2 = df2.rename(columns={'prompt': 'text'})
df2 = df2[["text", "label"]]

print(f"   -> Dataset 1 size: {len(df1)}")
print(f"   -> Dataset 2 size: {len(df2)}")

# --- FUSIÓN (MERGE) ---
print(">>> Merging datasets...")
pdf = pd.concat([df1, df2], ignore_index=True)

# Limpieza General
pdf = pdf.dropna(subset=["text"])
pdf["text"] = pdf["text"].astype(str)
pdf = pdf[pdf["text"].str.len() > 5]
pdf = pdf.drop_duplicates(subset=["text"]) # Eliminamos duplicados entre ambos sets

# Balanceo (Opcional: Si quieres usar TODO, comenta la línea de sample)
# Para Big Data real, intentemos usar todo lo posible, o un sample más grande
if len(pdf) > 20000:
    print(f"Sampling 20,000 examples from total of {len(pdf)}...")
    pdf = pdf.sample(n=20000, random_state=42).reset_index(drop=True)
else:
    print(f"Using full combined dataset: {len(pdf)} samples")

print(f"\n>>> Final Dataset Statistics:")
print(f"    Total samples: {len(pdf)}")
print(f"    Jailbreaks (1): {pdf['label'].sum()} ({pdf['label'].mean():.1%})")
print(f"    Safe (0): {(pdf['label'] == 0).sum()}")

# Guardar en Parquet (Spark maneja la escritura optimizada)
print("\n>>> Saving to Parquet...")
df_final = spark.createDataFrame(pdf)
# Reparticionamos a 1 para tener un solo archivo o pocos archivos grandes
df_final.repartition(1).write.mode("overwrite").parquet("processed_data")

print(">>> Ingestion Complete! Ready for Step 2 (Embedding).")