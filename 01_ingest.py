import sys
import time
import json
import pandas as pd
from pyspark.sql import SparkSession
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

# Start timing
start_time = time.time()

# Inicializar Spark
spark = SparkSession.builder \
    .appName("Ingesta_Jailbreak_BigData") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

print(">>> INICIANDO INGESTA (Jayavibhav + Imoxto)...")

# --- Función para Deduplicación Semántica con MinHash LSH ---
def semantic_deduplicate(df, text_col='text', threshold=0.8, num_perm=128):
    """
    Deduplica semánticamente usando MinHash LSH.
    Identifica textos que son ~80% similares (por defecto) y conserva solo uno.
    
    Args:
        df: DataFrame con columna de texto
        text_col: Nombre de la columna de texto
        threshold: Umbral de similitud (0.8 = 80% similar se considera duplicado)
        num_perm: Número de permutaciones para MinHash (más = más preciso pero más lento)
    
    Returns:
        DataFrame deduplicado
    """
    print(f"\n>>> DEDUPLICACIÓN SEMÁNTICA (MinHash LSH, threshold={threshold})...")
    original_count = len(df)
    
    def get_minhash(text):
        """Genera MinHash para un texto usando n-gramas de palabras."""
        m = MinHash(num_perm=num_perm)
        # Usar shingles de 3 palabras para capturar contexto
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = ' '.join(words[i:i+3])
            m.update(shingle.encode('utf8'))
        # Fallback para textos muy cortos
        if len(words) < 3:
            for word in words:
                m.update(word.encode('utf8'))
        return m
    
    # Crear índice LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_indices = []
    
    texts = df[text_col].tolist()
    for idx, text in enumerate(tqdm(texts, desc="   Deduplicando")):
        mh = get_minhash(str(text))
        # Si no hay textos similares ya indexados, este es único
        if not lsh.query(mh):
            lsh.insert(f"doc_{idx}", mh)
            unique_indices.append(idx)
    
    df_dedup = df.iloc[unique_indices].reset_index(drop=True)
    removed = original_count - len(df_dedup)
    print(f"   -> Eliminados {removed} near-duplicates ({removed/original_count:.1%})")
    print(f"   -> Conservados {len(df_dedup)} muestras únicas")
    
    return df_dedup

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

# --- 1. JAYAVIBHAV PROMPT-INJECTION-SAFETY ---
print("\n1. Descargando jayavibhav/prompt-injection-safety...")
try:
    df1 = load_dataset("jayavibhav/prompt-injection-safety", split="train").to_pandas()
    df1 = estandarizar_df(df1, "jayavibhav-safety")
except Exception as e:
    print(f"Error cargando Jayavibhav Safety: {e}")
    df1 = pd.DataFrame()

# --- 2. JAYAVIBHAV PROMPT-INJECTION ---
print("\n2. Descargando jayavibhav/prompt-injection...")
try:
    df2 = load_dataset("jayavibhav/prompt-injection", split="train").to_pandas()
    df2 = estandarizar_df(df2, "jayavibhav-injection")
except Exception as e:
    print(f"Error cargando Jayavibhav Injection: {e}")
    df2 = pd.DataFrame()

# --- FUSIÓN DE DATOS ---
print("\n>>> FUSIONANDO DATASETS (MERGE)...")
dfs = [d for d in [df1, df2] if not d.empty]

if not dfs:
    print("ERROR CRÍTICO: No se cargaron datos.")
    sys.exit(1)

pdf = pd.concat(dfs, ignore_index=True)

# Track statistics for paper
raw_count = len(pdf)
source_counts_raw = pdf['source'].value_counts().to_dict()

# Limpieza Básica
pdf = pdf.dropna(subset=["text"])
pdf = pdf[pdf["text"].str.len() > 5]
after_cleaning = len(pdf)

# Paso 1: Deduplicación exacta (rápida)
pre_exact = len(pdf)
pdf = pdf.drop_duplicates(subset=["text"])
exact_duplicates_removed = pre_exact - len(pdf)
print(f"   -> Exact duplicates eliminados: {exact_duplicates_removed}")

# Paso 2: Deduplicación semántica con MinHash LSH (elimina near-duplicates)
pre_semantic = len(pdf)
pdf = semantic_deduplicate(pdf, text_col='text', threshold=0.8)
semantic_duplicates_removed = pre_semantic - len(pdf)

print("\n>>> ESTADÍSTICAS DEL DATASET FINAL:")
print(f"    Total de Muestras: {len(pdf)}")
print(f"    Fuentes: {pdf['source'].unique()}")
print(f"    Ataques (Label 1): {pdf['label'].sum()} ({pdf['label'].mean():.1%})")
print(f"    Seguros (Label 0): {len(pdf) - pdf['label'].sum()}")

# Save dataset statistics for paper
dataset_stats = {
    "sources": {
        "jayavibhav-safety": "https://huggingface.co/datasets/jayavibhav/prompt-injection-safety",
        "jayavibhav-injection": "https://huggingface.co/datasets/jayavibhav/prompt-injection"
    },
    "raw_counts": {
        "total_raw": int(raw_count),
        "by_source": {k: int(v) for k, v in source_counts_raw.items()}
    },
    "deduplication": {
        "after_basic_cleaning": int(after_cleaning),
        "exact_duplicates_removed": int(exact_duplicates_removed),
        "semantic_duplicates_removed": int(semantic_duplicates_removed),
        "semantic_threshold": 0.8,
        "total_duplicates_removed": int(exact_duplicates_removed + semantic_duplicates_removed)
    },
    "final_dataset": {
        "total_samples": int(len(pdf)),
        "jailbreak_samples": int(pdf['label'].sum()),
        "safe_samples": int(len(pdf) - pdf['label'].sum()),
        "jailbreak_percentage": round(pdf['label'].mean() * 100, 2),
        "by_source": {k: int(v) for k, v in pdf['source'].value_counts().to_dict().items()}
    }
}

with open("dataset_stats.json", "w") as f:
    json.dump(dataset_stats, f, indent=2)
print(">>> Dataset statistics saved to 'dataset_stats.json'")

# Print LaTeX table for paper
print("\n" + "="*70)
print("LATEX TABLE: Dataset Statistics (copy to paper)")
print("="*70)
print(r"""
\begin{table}[h]
\centering
\caption{Dataset Composition and Preprocessing Statistics}
\begin{tabular}{|l|r|}
\hline
\textbf{Metric} & \textbf{Count} \\
\hline""")
print(f"Raw samples (combined) & {raw_count:,} \\\\")
print(r"\hline")
print(f"After basic cleaning & {after_cleaning:,} \\\\")
print(r"\hline")
print(f"Exact duplicates removed & {exact_duplicates_removed:,} \\\\")
print(r"\hline")
print(f"Semantic duplicates removed & {semantic_duplicates_removed:,} \\\\")
print(r"\hline")
print(f"\\textbf{{Final dataset size}} & \\textbf{{{len(pdf):,}}} \\\\")
print(r"\hline")
print(f"Jailbreak samples (label=1) & {int(pdf['label'].sum()):,} ({pdf['label'].mean():.1%}) \\\\")
print(r"\hline")
print(f"Safe samples (label=0) & {int(len(pdf) - pdf['label'].sum()):,} ({1-pdf['label'].mean():.1%}) \\\\")
print(r"""\hline
\end{tabular}
\label{tab:dataset}
\end{table}""")

# Guardar
print("\n>>> Guardando en formato Parquet...")
df_final = spark.createDataFrame(pdf)
# Repartimos en 4 archivos para aprovechar tus núcleos
df_final.repartition(4).write.mode("overwrite").parquet("processed_data")

# Final timing
elapsed = time.time() - start_time
print(f"\n>>> Ingesta completada en {elapsed:.2f} segundos ({elapsed/60:.2f} minutos)")
print(">>> ¡Ingesta Completa! Ahora ejecuta el paso 02_embed.py")