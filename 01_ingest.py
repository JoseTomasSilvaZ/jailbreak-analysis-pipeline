import ray
import ray.data
import numpy as np

# Initialize Ray. 
# Ray will automatically detect your M4 Pro's CPU cores (likely 10 or 14).
ray.init(ignore_reinit_error=True)

print(">>> Step 1: Connecting to Hugging Face Dataset Stream...")

# 1. Lazy Read from Hugging Face
# We use the direct Parquet path for speed. 
# This does NOT download the whole file to RAM; it streams it.
ds = ray.data.read_parquet("hf://datasets/lmsys/lmsys-chat-1m")

# 2. Define the Processing Logic
def extract_user_turns(batch):
    """
    Flattens the conversation tree.
    Input: A batch of conversations (List of Lists)
    Output: A batch of specific User Prompts + Safety Label
    """
    prompts = []
    labels = []
    
    # Iterate over the batch
    for i in range(len(batch["conversation"])):
        conv = batch["conversation"][i]
        # The dataset provides 'openai_moderation' usually as a parallel list or metadata.
        # For robustness, we assume the dataset structure aligns moderation tags with messages
        # or we check if the conversation *as a whole* was flagged if per-message isn't available.
        # (Adapting to the specific LMSYS structure where moderation is often per message)
        moderation = batch["openai_moderation"][i] 
        
        for j, message in enumerate(conv):
            if message["role"] == "user":
                prompts.append(message["content"])
                
                # Check Ground Truth Label
                # If the corresponding moderation tag shows flagged=True, mark as 1 (Unsafe)
                # Note: Logic depends on exact dataset version; this is the standard heuristic.
                is_unsafe = 0
                if j < len(moderation):
                    # Check if the message was flagged as unsafe by OpenAI
                    if moderation[j] and moderation[j].get('flagged', False):
                        is_unsafe = 1
                
                labels.append(is_unsafe)
                
    return {"text": prompts, "label": labels}

# 3. Apply the Transform
# batch_format="numpy" is faster for Ray to serialize
print(">>> Step 2: Extracting User Prompts (Lazy Evaluation)...")
processed_ds = ds.map_batches(extract_user_turns, batch_format="numpy")

# 4. Save Intermediate Result
# We save this to disk so we don't have to re-download if the next step fails.
# This creates a folder "processed_data" with parquet parts.
print(">>> Step 3: Saving pre-processed data to disk...")
processed_ds.write_parquet("processed_data")
print(">>> Ingestion Complete. Data saved to ./processed_data")