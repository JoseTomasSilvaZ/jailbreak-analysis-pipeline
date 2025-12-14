import os
import time

# --- FIX CRÍTICO: Evita que Spark y HuggingFace se bloqueen mutuamente ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer

# Start timing
start_time = time.time()

# Inicializar Spark
spark = SparkSession.builder \
    .appName("JailbreakEmbedding_Turbo") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print(">>> Cargando datos procesados...")
df = spark.read.parquet("processed_data")

# OPTIMIZACIÓN: Reparticionar según tus núcleos de CPU
# Si tienes 8 núcleos, usa 8 o 16 particiones.
NUM_PARTITIONS = 4
df = df.repartition(NUM_PARTITIONS)
print(f"   -> Datos repartidos en {NUM_PARTITIONS} procesos paralelos.")

def embed_partition_batch(iterator):
    """
    Procesa los datos en LOTES (Batches).
    Esto es mucho más rápido que hacerlo fila por fila.
    MODIFICADO: Ahora también guarda el texto original para análisis posterior.
    """
    print(f"--- Iniciando Worker en proceso {os.getpid()} ---")
    
    # Cargar el modelo una sola vez por proceso (en CPU)
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    batch_size = 128  # Procesamos 64 textos de golpe
    batch_texts = []
    batch_labels = []
    
    count = 0
    
    for row in iterator:
        batch_texts.append(row.text)
        batch_labels.append(row.label)
        
        # Cuando llenamos el lote, procesamos
        if len(batch_texts) >= batch_size:
            # La magia: encode() es vectorizado, muy rápido con listas
            embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
            
            # Devolvemos los resultados con el texto original incluido
            for i, emb in enumerate(embeddings):
                yield (emb.tolist(), batch_labels[i], batch_texts[i])
            
            count += len(batch_texts)
            if count % 1000 == 0:
                print(f"[Worker {os.getpid()}] Procesadas {count} filas...")
                
            # Limpiar lote
            batch_texts = []
            batch_labels = []
            
    # Procesar el último lote (remanente)
    if batch_texts:
        embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
        for i, emb in enumerate(embeddings):
            yield (emb.tolist(), batch_labels[i], batch_texts[i])
        print(f"[Worker {os.getpid()}] ¡Terminado! Total: {count + len(batch_texts)}")

print(">>> Iniciando Embedding Distribuido (Modo Batch)...")
print("    (Esto puede tomar unos minutos, pero verás logs de progreso)")

# MapPartitions permite manejar la iteración manualmente para hacer batching
rdd = df.rdd.mapPartitions(embed_partition_batch)

# Convertir de nuevo a DataFrame (ahora con texto original)
df_embeddings = rdd.toDF(["features", "label", "text"])

print(">>> Guardando Embeddings Finales...")
df_embeddings.write.mode("overwrite").parquet("embeddings_final")

# Final timing
elapsed = time.time() - start_time
print(f"\n>>> Embedding completado en {elapsed:.2f} segundos ({elapsed/60:.2f} minutos)")
print(">>> ¡Proceso Terminado Exitosamente!")