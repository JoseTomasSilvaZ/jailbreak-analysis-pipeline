# 02_embed.py
import os
import sys
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
import pandas as pd

spark = SparkSession.builder \
    .appName("JailbreakEmbedding") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print(">>> Loading processed data...")
df = spark.read.parquet("processed_data")

# Repartition ensures we use all your cores (e.g., 10 partitions for 10 cores)
df = df.repartition(10)

def embed_partition(iterator):
    # This runs INSIDE the worker process.
    # We load the model once per partition (Efficient)
    print("--- Initializing Model in Worker ---")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    for row in iterator:
        # Encode returns a numpy array, we convert to list for Spark
        vector = model.encode(row.text, normalize_embeddings=True).tolist()
        yield (vector, row.label)

print(">>> Starting Distributed Embedding (MapPartitions)...")

# Apply the function to each partition
rdd = df.rdd.mapPartitions(embed_partition)

# Convert back to DataFrame
# Schema: features (List of Floats), label (Int)
df_embeddings = rdd.toDF(["features", "label"])

print(">>> Saving Embeddings...")
df_embeddings.write.mode("overwrite").parquet("embeddings_final")
print(">>> Done.")