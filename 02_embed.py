import ray
import numpy as np

# --- FIX: LOGICAL GPU HACK ---
# We tell Ray we have 4 "tickets" for the GPU. 
# This bypasses the fractional (0.25) math that is causing the hang.
ray.init(num_gpus=4, ignore_reinit_error=True)

class Embedder:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        
        # 1. Detect Apple Silicon GPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Worker initialized on device: {self.device}")
        
        # 2. Load Model
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.model.half()

    def __call__(self, batch):
        embeddings = self.model.encode(
            batch["text"].tolist(),
            batch_size=256,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return {"embedding": embeddings, "label": batch["label"]}

print(">>> Loading processed data...")
ds = ray.data.read_parquet("processed_data")

# --- TEST LIMIT ---
print(">>> LIMITING TO 10,000 ROWS FOR TESTING...")
ds = ds.limit(300_000)
# ------------------

print(f">>> Starting Distributed Embedding on M4 Neural Engine...")
embedding_ds = ds.map_batches(
    Embedder,  
    compute=ray.data.ActorPoolStrategy(size=4),
    num_gpus=1, # Each worker takes 1 "Logical Ticket"
    batch_size=1024
)

print(">>> Saving Embeddings...")
embedding_ds.write_parquet("embeddings_10k_test")
print(">>> Done. Vectors saved to ./embeddings_10k_test")