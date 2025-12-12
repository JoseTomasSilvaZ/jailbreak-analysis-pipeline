import os
import sys
import pandas as pd
from pyspark.sql import SparkSession
from datasets import load_dataset

# Inicializar Spark
spark = SparkSession.builder \
    .appName("Ingesta_Jailbreak_BigData") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

print(">>> INICIANDO INGESTA (Jayavibhav + Imoxto)...")

# --- Función Auxiliar para Estandarizar (Esta función arregla el error de Imoxto) ---
def estandarizar_df(df, nombre_fuente):
    print(f"   -> Procesando {nombre_fuente} (Columnas originales: {list(df.columns)})...")
    
    # 1. Normalizar columna de TEXTO
    candidatos_texto = ['prompt', 'data', 'sentence', 'text_prompts', 'question']
    for col in candidatos_texto:
        if col in df.columns:
            df = df.rename(columns={col: 'text'})
            break
            
    # 2. Normalizar columna de ETIQUETA (LABEL)
    # Buscamos variantes como 'Label', 'labels', 'ground_truth'
    candidatos_label = ['Label', 'labels', 'ground_truth', 'class', 'target', 'is_jailbreak']
    for col in candidatos_label:
        if col in df.columns:
            df = df.rename(columns={col: 'label'})
            break

    # 3. Validación
    if 'text' not in df.columns or 'label' not in df.columns:
        print(f"      [SALTAR] No se encontró 'text' o 'label' en {nombre_fuente}")
        return pd.DataFrame()

    # 4. Limpiar tipos de datos
    # Mapear strings 'safe'/'injection' a 0/1 si es necesario
    if df['label'].dtype == 'object':
        mapeo = {'safe': 0, 'benign': 0, 'injection': 1, 'jailbreak': 1, 'unsafe': 1}
        df['label'] = df['label'].map(mapeo).fillna(df['label'])
    
    # Asegurar que sean números enteros
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df['text'] = df['text'].astype(str)
    
    # Seleccionar solo lo que nos importa
    df = df[['text', 'label']]
    df['source'] = nombre_fuente
    return df

# --- 1. JAYAVIBHAV ---
print("\n1. Descargando jayavibhav/prompt-injection-safety...")
try:
    df1 = load_dataset("jayavibhav/prompt-injection-safety", split="train").to_pandas()
    df1 = estandarizar_df(df1, "jayavibhav")
except Exception as e:
    print(f"Error cargando Jayavibhav: {e}")
    df1 = pd.DataFrame()

# --- 2. IMOXTO ---
print("\n2. Descargando imoxto/prompt_injection_cleaned_dataset-v2...")
try:
    df2 = load_dataset("imoxto/prompt_injection_cleaned_dataset-v2", split="train").to_pandas()
    # Usamos la función estandarizar para arreglar el nombre de la columna 'labels'
    df2 = estandarizar_df(df2, "imoxto")
except Exception as e:
    print(f"Error cargando Imoxto: {e}")
    df2 = pd.DataFrame()

# --- FUSIÓN DE DATOS ---
print("\n>>> FUSIONANDO DATASETS (MERGE)...")
dfs = [d for d in [df1, df2] if not d.empty]

if not dfs:
    print("ERROR CRÍTICO: No se cargaron datos.")
    sys.exit(1)

pdf = pd.concat(dfs, ignore_index=True)

# Limpieza Final
pdf = pdf.dropna(subset=["text"])
pdf = pdf[pdf["text"].str.len() > 5]
pdf = pdf.drop_duplicates(subset=["text"]) 

print(f"\n>>> ESTADÍSTICAS DEL DATASET FINAL:")
print(f"    Total de Muestras: {len(pdf)}")
print(f"    Fuentes: {pdf['source'].unique()}")
print(f"    Ataques (Label 1): {pdf['label'].sum()} ({pdf['label'].mean():.1%})")
print(f"    Seguros (Label 0): {len(pdf) - pdf['label'].sum()}")

# Guardar
print("\n>>> Guardando en formato Parquet...")
df_final = spark.createDataFrame(pdf)
# Repartimos en 4 archivos para aprovechar tus núcleos
df_final.repartition(4).write.mode("overwrite").parquet("processed_data")

print(">>> ¡Ingesta Completa! Ahora ejecuta el paso 02_embed.py")