import os
import sys
import time # Necesario para la pausa de cortesía
# Desactivar paralelismo de tokenizers para evitar bloqueos
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

# --- CONFIGURACIÓN ANTI-FREEZE ---
# Limitamos Spark para que no se coma toda tu RAM de 16GB
spark = SparkSession.builder \
    .appName("Jailbreak_Safe_GPU") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "1") \
    .config("spark.cores.max", "2") \
    .getOrCreate()

print(">>> [MODO SEGURO] Cargando datos...")
df = spark.read.parquet("processed_data")

# --- CLAVE 1: MENOS CONCURRENCIA ---
NUM_PARTITIONS = 2 
df = df.repartition(NUM_PARTITIONS)
print(f"   -> Estrategia: {NUM_PARTITIONS} procesos (para evitar colapso de RAM).")

def embed_partition_safe(iterator):
    import torch
    print(f"--- Iniciando Worker GPU (PID: {os.getpid()}) ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cargar modelo
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # --- CLAVE 2: BATCH PEQUEÑO ---
    batch_size = 32
    
    batch_texts = []
    batch_labels = []
    
    count = 0
    
    for row in iterator:
        batch_texts.append(row.text)
        batch_labels.append(row.label)
        
        if len(batch_texts) >= batch_size:
            # Procesar lote
            embeddings = model.encode(
                batch_texts, 
                batch_size=batch_size,
                normalize_embeddings=True, 
                show_progress_bar=False,
                device=device
            )
            
            for i, emb in enumerate(embeddings):
                yield (emb.tolist(), batch_labels[i])
            
            # --- CLAVE 3: DESCANSO PARA LA UI ---
            time.sleep(0.01)
            
            # Limpiar memoria VRAM si es necesario (opcional, ayuda en casos extremos)
            # torch.cuda.empty_cache() 
            
            batch_texts = []
            batch_labels = []
            
            count += batch_size
            if count % 1000 == 0:
                print(f"[Worker {os.getpid()}] Progreso: {count} filas...")

    # Procesar remanentes
    if batch_texts:
        embeddings = model.encode(batch_texts, device=device)
        for i, emb in enumerate(embeddings):
            yield (emb.tolist(), batch_labels[i])

print(">>> Ejecutando Pipeline con Protección de UI...")
rdd = df.rdd.mapPartitions(embed_partition_safe)
df_embeddings = rdd.toDF(["features", "label"])

print(">>> Guardando...")
df_embeddings.write.mode("overwrite").parquet("embeddings_final")
print(">>> Terminado")