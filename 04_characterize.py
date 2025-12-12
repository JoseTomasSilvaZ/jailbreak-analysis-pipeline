import pyarrow.parquet as pq
import numpy as np
# CAMBIO: Usamos el HDBSCAN nativo de Scikit-Learn (más moderno y estable)
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob
import pandas as pd
import sys

print(">>> Loading Data...")
# Use the embeddings created by 02_embed.py
parquet_files = glob.glob("embeddings_final/*.parquet")
if not parquet_files:
    print("ERROR: No parquet files found in embeddings_final.")
    sys.exit(1)

# Read all files and combine them
dfs = []
for file in parquet_files:
    if not file.endswith("_SUCCESS"):  # Skip metadata file
        try:
            table = pq.read_table(file)
            df = table.to_pandas()
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {file}: {e}")

if not dfs:
    print("ERROR: No valid data found inside parquet files")
    sys.exit(1)

combined_df = pd.concat(dfs, ignore_index=True)

# Filtro de seguridad (igual que en 03_train)
combined_df = combined_df[combined_df['label'].isin([0, 1])]

X = np.stack(combined_df["features"].values)
y = combined_df["label"].values

# 1. Filter: Analyze ONLY the Jailbreaks
jailbreak_vectors = X[y == 1]
print(f"Analyzing {len(jailbreak_vectors)} jailbreak attempts...")

if len(jailbreak_vectors) == 0:
    print("ERROR: No jailbreak samples found!")
    sys.exit(1)

if len(jailbreak_vectors) < 20:
    print(f"WARNING: Only {len(jailbreak_vectors)} jailbreaks. Clustering might be boring.")

# 2. PCA Compression (Speed Hack)
print(">>> Running PCA...")
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(jailbreak_vectors)

# 3. HDBSCAN Clustering (Usando Sklearn nativo)
print(">>> Running HDBSCAN (Sklearn Implementation)...")
# Nota: La implementación de sklearn usa los mismos parámetros
clusterer = HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
labels = clusterer.fit_predict(X_reduced)

# 4. Report
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Found {n_clusters} distinct types of Jailbreak attacks.")

# 5. Visualization
pca_2d = PCA(n_components=2)
plot_data = pca_2d.fit_transform(jailbreak_vectors)

plt.figure(figsize=(10, 8))
# Usamos un mapa de colores que maneje bien el ruido (-1)
scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=labels, cmap='viridis', s=2, alpha=0.6)
plt.title(f"Map of {n_clusters} Jailbreak Clusters (Noise in purple/dark)")
plt.colorbar(scatter, label='Cluster ID')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig("jailbreak_clusters.png")
print(">>> Cluster map saved to 'jailbreak_clusters.png'")