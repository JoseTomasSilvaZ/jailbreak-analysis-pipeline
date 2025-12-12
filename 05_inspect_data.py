# 05_inspect_data.py
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Inspector") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print(">>> Leyendo datos crudos...")
df = spark.read.parquet("processed_data")

print("\n=== ¿QUÉ ES LA ETIQUETA 1 (SUPUESTO JAILBREAK)? ===")
# Mostramos 5 ejemplos de lo que el dataset considera un ataque
df.filter(df.label == 1).select("text").show(5, truncate=False)

print("\n=== ¿QUÉ ES LA ETIQUETA 0 (SUPUESTO SEGURO)? ===")
# Mostramos 5 ejemplos de lo que el dataset considera seguro
df.filter(df.label == 0).select("text").show(5, truncate=False)

print("\n>>> Estadística de clases:")
total = df.count()
jailbreaks = df.filter(df.label == 1).count()
print(f"Total: {total}")
print(f"Etiqueta 1 (Jailbreak): {jailbreaks} ({jailbreaks/total:.1%})")