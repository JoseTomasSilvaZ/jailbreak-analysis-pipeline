import pyarrow.parquet as pq
import numpy as np
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob

print(">>> Loading Data...")
# Use the embeddings we actually created (10k test sample)
parquet_files = glob.glob("embeddings_10k_test/*.parquet")
if not parquet_files:
    raise FileNotFoundError("No parquet files found in embeddings_10k_test. Did Step 2 finish?")
table = pq.read_table(parquet_files[0])
X = np.stack(table["embedding"].to_numpy())
y = table["label"].to_numpy()

# 1. Filter: Analyze ONLY the Jailbreaks
# We don't care about clustering safe conversations.
jailbreak_vectors = X[y == 1]
print(f"Analyzing {len(jailbreak_vectors)} jailbreak attempts...")

# 2. PCA Compression (Speed Hack)
# Reduce 384 dims -> 50 dims (retains ~95% variance)
print(">>> Running PCA...")
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(jailbreak_vectors)

# 3. HDBSCAN Clustering
# Adjust parameters for smaller dataset (10k samples instead of full dataset)
# min_cluster_size=10 means "A group must have 10 similar attacks to be a cluster"
print(">>> Running HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
labels = clusterer.fit_predict(X_reduced)

# 4. Report
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Found {n_clusters} distinct types of Jailbreak attacks.")

# 5. Visualization (Optional - saves a plot)
# Reduce to 2D just for the picture
pca_2d = PCA(n_components=2)
plot_data = pca_2d.fit_transform(jailbreak_vectors)

plt.figure(figsize=(10, 8))
plt.scatter(plot_data[:, 0], plot_data[:, 1], c=labels, cmap='viridis', s=1, alpha=0.5)
plt.title(f"Visual Map of {n_clusters} Jailbreak Strategies")
plt.colorbar(label='Cluster ID')
plt.savefig("jailbreak_clusters.png")
print(">>> Cluster map saved to 'jailbreak_clusters.png'")