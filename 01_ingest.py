# 01_ingest.py
"""
Jailbreak Detection Data Ingestion
Uses jayavibhav/prompt-injection-safety dataset (60k samples)
"""
import os
import sys
import pandas as pd
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("JailbreakIngestion") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

print(">>> Loading Jailbreak Dataset...")

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

# Load jayavibhav/prompt-injection-safety (60k samples)
# Has 'text' and 'label' columns (label: 0=safe, 1=injection)
print(">>> Downloading jayavibhav/prompt-injection-safety...")
ds_train = load_dataset("jayavibhav/prompt-injection-safety", split="train")
ds_test = load_dataset("jayavibhav/prompt-injection-safety", split="test")

# Combine train and test
pdf_train = ds_train.to_pandas()
pdf_test = ds_test.to_pandas()
pdf = pd.concat([pdf_train, pdf_test], ignore_index=True)

print(f"    ✓ Loaded {len(pdf)} samples")

# Clean up
pdf = pdf.sample(n=10000, random_state=42).reset_index(drop=True)
pdf = pdf.dropna(subset=["text"])
pdf["text"] = pdf["text"].astype(str)
pdf = pdf[pdf["text"].str.len() > 5]
pdf = pdf[pdf["text"].str.len() < 10000]
pdf = pdf.drop_duplicates(subset=["text"])

# Shuffle
pdf = pdf.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n>>> Final Dataset Statistics:")
print(f"    Total samples: {len(pdf)}")
print(f"    Jailbreaks (label=1): {pdf['label'].sum()} ({pdf['label'].mean():.1%})")
print(f"    Safe (label=0): {(pdf['label'] == 0).sum()} ({(pdf['label'] == 0).mean():.1%})")

# Save
print("\n>>> Saving to Parquet...")
df_final = spark.createDataFrame(pdf)
df_final.write.mode("overwrite").parquet("processed_data")

print(">>> Ingestion Complete!")
print("Next: Run 02_embed.py")
