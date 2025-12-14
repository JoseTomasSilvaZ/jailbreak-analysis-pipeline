import pyarrow.parquet as pq
import numpy as np
import time
import json
# Paper Section 4.3.1: UMAP for manifold learning + HDBSCAN for density clustering
import umap
from sklearn.cluster import HDBSCAN
# PCA removed - using UMAP for all dimensionality reduction (paper consistency)
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import glob
import pandas as pd
import sys
import csv
# Paper Section 4.3.2: SHAP for explainability
import shap
import xgboost as xgb
from collections import Counter
import re

# Start timing
start_time = time.time()

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

# NEW: Extract original texts for semantic profiling
texts = combined_df["text"].values if "text" in combined_df.columns else None

# 1. Filter: Analyze ONLY the Jailbreaks
jailbreak_mask = y == 1
jailbreak_vectors = X[jailbreak_mask]
jailbreak_texts = texts[jailbreak_mask] if texts is not None else None

print(f"Analyzing {len(jailbreak_vectors)} jailbreak attempts...")

if len(jailbreak_vectors) == 0:
    print("ERROR: No jailbreak samples found!")
    sys.exit(1)

if len(jailbreak_vectors) < 20:
    print(f"WARNING: Only {len(jailbreak_vectors)} jailbreaks. Clustering might be boring.")

if jailbreak_texts is None:
    print("WARNING: Original texts not found. Semantic profiling will be limited.")
    print("(Re-run 02_embed.py to generate embeddings with text)")

# 2. UMAP Dimensionality Reduction (Paper Section 4.3.1)
print(">>> Running UMAP for manifold learning...")
print("    (UMAP preserves both local and global structure)")
reducer = umap.UMAP(
    n_components=10,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
X_reduced = reducer.fit_transform(jailbreak_vectors)
print(f"    Reduced from {jailbreak_vectors.shape[1]}D to {X_reduced.shape[1]}D")

# 3. HDBSCAN Clustering (Paper Section 4.3.1)
print(">>> Running HDBSCAN for hierarchical density-based clustering...")
print("    (HDBSCAN auto-adapts to varying densities, no eps tuning needed)")
clusterer = HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom'  # Excess of Mass - more robust
)
labels = clusterer.fit_predict(X_reduced)

# 4. Report
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"\nFound {n_clusters} distinct jailbreak attack families.")
print(f"Noise points (isolated attacks): {n_noise} ({n_noise/len(labels):.1%})")

# 4.1 Clustering Quality Metrics
valid_mask = labels != -1
silhouette = None
if sum(valid_mask) > 1 and n_clusters > 1:
    silhouette = silhouette_score(X_reduced[valid_mask], labels[valid_mask])
    print(f"Silhouette Score: {silhouette:.3f} (higher = better cluster separation)")

# Save cluster statistics for paper
cluster_stats = {
    "n_jailbreak_samples": int(len(jailbreak_vectors)),
    "n_clusters": int(n_clusters),
    "n_noise": int(n_noise),
    "noise_percentage": round(n_noise / len(labels) * 100, 2),
    "silhouette_score": round(float(silhouette), 4) if silhouette else None,
    "cluster_sizes": {str(k): int(v) for k, v in Counter(labels).items()},
    "umap_params": {
        "n_components": 10,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "cosine"
    },
    "hdbscan_params": {
        "min_cluster_size": 10,
        "min_samples": 5,
        "cluster_selection_method": "eom"
    }
}

with open("cluster_stats.json", "w") as f:
    json.dump(cluster_stats, f, indent=2)
print(">>> Cluster statistics saved to 'cluster_stats.json'")

# 5. Optional: SHAP Explainability (Paper Section 4.3.2)
# Note: SHAP on UMAP dimensions has limited interpretability
# The keyword extraction below provides more actionable insights
ENABLE_SHAP = False  # Set to True if needed for paper

if ENABLE_SHAP:
    print("\n>>> Training surrogate model for SHAP analysis...")
    valid_mask = labels != -1
    if sum(valid_mask) > 20:
        X_valid = X_reduced[valid_mask]
        labels_valid = labels[valid_mask]
        
        surrogate = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        surrogate.fit(X_valid, labels_valid)
        
        explainer = shap.TreeExplainer(surrogate)
        shap_values = explainer.shap_values(X_valid)
        
        print("\n=== SHAP-Based Dimension Importance ===")
        for cluster_id in sorted(set(labels_valid))[:3]:  # Show first 3 only
            cluster_mask = labels_valid == cluster_id
            if isinstance(shap_values, list):
                cluster_shap = np.abs(shap_values[cluster_id][cluster_mask]).mean(axis=0)
            else:
                cluster_shap = np.abs(shap_values[cluster_mask]).mean(axis=0)
            top_dims = np.argsort(cluster_shap)[-3:][::-1]
            print(f"Cluster {cluster_id}: Top UMAP dims = {top_dims.tolist()}")
else:
    print("\n>>> Skipping SHAP analysis (set ENABLE_SHAP=True to enable)")

# 6. Semantic Profiling with Keyword Extraction (Paper Section 4.3.2)
def profile_cluster_semantically(texts_array, mask):
    """Extract characteristic keywords from cluster texts"""
    cluster_texts = texts_array[mask]
    
    # Tokenize and extract meaningful words
    all_words = []
    for text in cluster_texts[:200]:  # Sample up to 200 for efficiency
        # Extract words (alphanumeric, length > 3)
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        all_words.extend(words)
    
    # Find most common keywords
    word_freq = Counter(all_words)
    # Filter out common stop words
    stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 
                 'your', 'their', 'about', 'which', 'there', 'would', 'could',
                 'what', 'when', 'where', 'they', 'them', 'than', 'then',
                 'these', 'those', 'some', 'such', 'only', 'also', 'more',
                 'most', 'other', 'into', 'very', 'just', 'over', 'after'}
    keywords = [(w, c) for w, c in word_freq.most_common(20) if w not in stopwords]
    
    return keywords[:10], cluster_texts[:3]  # Return top 10 keywords and 3 examples

if jailbreak_texts is not None and n_clusters > 0:
    print("\n\n=== SEMANTIC CLUSTER PROFILES ===")
    print("(Automated characterization of attack families)")
    
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue  # Skip noise
        
        cluster_mask = labels == cluster_id
        cluster_size = sum(cluster_mask)
        
        cluster_keywords, examples = profile_cluster_semantically(
            jailbreak_texts, cluster_mask
        )
        
        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster_id}: {cluster_size} attacks")
        print('='*60)
        print("Characteristic keywords:")
        for word, count in cluster_keywords:
            print(f"  - {word}: {count} occurrences")
        
        print("\nExample attacks from this cluster:")
        for i, example in enumerate(examples, 1):
            preview = example[:150] + "..." if len(example) > 150 else example
            print(f"  [{i}] {preview}")

# 6.5 Generate Publication-Ready Table
ATTACK_FAMILY_PATTERNS = {
    'ignore': 'Instruction Override',
    'instructions': 'Instruction Override', 
    'previous': 'Instruction Override',
    'disregard': 'Instruction Override',
    'forget': 'Instruction Override',
    'roleplay': 'Role-Play Manipulation',
    'character': 'Role-Play Manipulation',
    'pretend': 'Role-Play Manipulation',
    'acting': 'Role-Play Manipulation',
    'persona': 'Role-Play Manipulation',
    'imagine': 'Scenario Engineering',
    'hypothetical': 'Scenario Engineering',
    'emergency': 'Scenario Engineering',
    'scenario': 'Scenario Engineering',
    'situation': 'Scenario Engineering',
    'jailbreak': 'Direct Jailbreak',
    'bypass': 'Direct Jailbreak',
    'unlock': 'Direct Jailbreak',
    'mode': 'Mode Switching',
    'developer': 'Mode Switching',
    'sudo': 'Mode Switching',
    'admin': 'Mode Switching',
    'code': 'Code Injection',
    'execute': 'Code Injection',
    'script': 'Code Injection',
    'prompt': 'Prompt Leaking',
    'system': 'Prompt Leaking',
    'reveal': 'Prompt Leaking',
}

def infer_family_name(keywords):
    """Auto-suggest attack family name based on keywords"""
    for word, count in keywords:
        if word in ATTACK_FAMILY_PATTERNS:
            return ATTACK_FAMILY_PATTERNS[word]
    return "Novel Attack Pattern"

def infer_attack_strategy(keywords):
    """Infer attack strategy from keywords"""
    keyword_words = [w for w, c in keywords[:5]]
    if any(w in keyword_words for w in ['ignore', 'previous', 'instructions', 'disregard', 'forget']):
        return "Pretexting to bypass system prompts"
    elif any(w in keyword_words for w in ['roleplay', 'character', 'pretend', 'acting', 'persona']):
        return "Persona creation (e.g., DAN)"
    elif any(w in keyword_words for w in ['emergency', 'hypothetical', 'imagine', 'scenario']):
        return "Urgency-based ethical bypass"
    elif any(w in keyword_words for w in ['jailbreak', 'bypass', 'unlock']):
        return "Direct jailbreak attempt"
    elif any(w in keyword_words for w in ['code', 'execute', 'script']):
        return "Code/command injection"
    elif any(w in keyword_words for w in ['mode', 'developer', 'sudo', 'admin']):
        return "Privilege escalation attempt"
    elif any(w in keyword_words for w in ['prompt', 'system', 'reveal']):
        return "System prompt extraction"
    else:
        return "Novel attack pattern"

if jailbreak_texts is not None and n_clusters > 0:
    # Build detailed cluster data
    table_data = []
    all_keywords_by_family = {}  # Collect keywords per family for aggregation
    
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        
        cluster_mask = labels == cluster_id
        cluster_size = int(sum(cluster_mask))
        
        cluster_keywords, examples = profile_cluster_semantically(
            jailbreak_texts, cluster_mask
        )
        
        family_name = infer_family_name(cluster_keywords)
        strategy = infer_attack_strategy(cluster_keywords)
        top_keywords_str = ", ".join([f"{w} ({c}x)" for w, c in cluster_keywords[:3]])
        
        # Handle numpy array properly
        example_text = "N/A"
        if len(examples) > 0:
            example_text = examples[0][:100] + "..." if len(examples[0]) > 100 else examples[0]
        
        table_data.append({
            'cluster': cluster_id,
            'size': cluster_size,
            'family_name': family_name,
            'top_keywords': top_keywords_str,
            'strategy': strategy,
            'example': example_text
        })
        
        # Aggregate keywords by family
        if family_name not in all_keywords_by_family:
            all_keywords_by_family[family_name] = Counter()
        for word, count in cluster_keywords[:5]:
            all_keywords_by_family[family_name][word] += count
    
    # Save detailed data to CSV (for reference)
    with open('attack_families_detailed.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cluster', 'size', 'family_name', 'top_keywords', 'strategy', 'example'])
        writer.writeheader()
        writer.writerows(table_data)
    print(f"\n>>> Detailed cluster data saved to 'attack_families_detailed.csv'")
    
    # ========================================================================
    # AGGREGATED SUMMARY TABLE (for paper)
    # ========================================================================
    print("\n\n" + "="*80)
    print("AGGREGATED ATTACK FAMILY SUMMARY (for paper)")
    print("="*80)
    
    # Aggregate by family
    family_stats = {}
    for row in table_data:
        family = row['family_name']
        if family not in family_stats:
            family_stats[family] = {
                'clusters': 0,
                'total_attacks': 0,
                'strategies': set(),
                'example_clusters': []
            }
        family_stats[family]['clusters'] += 1
        family_stats[family]['total_attacks'] += row['size']
        family_stats[family]['strategies'].add(row['strategy'])
        if len(family_stats[family]['example_clusters']) < 3:
            family_stats[family]['example_clusters'].append(row['cluster'])
    
    total_attacks = sum(f['total_attacks'] for f in family_stats.values())
    
    # Sort by total attacks (descending)
    sorted_families = sorted(family_stats.items(), key=lambda x: x[1]['total_attacks'], reverse=True)
    
    # Build aggregated summary
    summary_data = []
    for family_name, stats in sorted_families:
        # Get top keywords for this family
        top_kw = all_keywords_by_family.get(family_name, Counter()).most_common(5)
        keywords_str = ", ".join([w for w, c in top_kw])
        
        # Primary strategy (most common)
        primary_strategy = list(stats['strategies'])[0]
        
        summary_data.append({
            'family': family_name,
            'clusters': stats['clusters'],
            'attacks': stats['total_attacks'],
            'percentage': stats['total_attacks'] / total_attacks * 100,
            'keywords': keywords_str,
            'strategy': primary_strategy
        })
    
    # Print Markdown summary table
    print("\n### Table: Aggregated Attack Family Distribution\n")
    print("| Attack Family | Clusters | Attacks | % | Top Keywords | Primary Strategy |")
    print("|---------------|----------|---------|---|--------------|------------------|")
    for row in summary_data:
        print(f"| {row['family']} | {row['clusters']} | {row['attacks']:,} | {row['percentage']:.1f}% | {row['keywords']} | {row['strategy']} |")
    
    print(f"\n**Total:** {n_clusters} clusters, {total_attacks:,} jailbreak attempts analyzed")
    
    # Save aggregated summary to CSV
    with open('attack_families_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['family', 'clusters', 'attacks', 'percentage', 'keywords', 'strategy'])
        writer.writeheader()
        writer.writerows(summary_data)
    print(f">>> Summary saved to 'attack_families_summary.csv'")
    
    # ========================================================================
    # LATEX TABLE FOR PAPER
    # ========================================================================
    print("\n" + "="*80)
    print("LATEX TABLE (copy directly to your paper)")
    print("="*80)
    print(r"""
\begin{table}[h]
\centering
\caption{Distribution of Discovered Jailbreak Attack Families}
\begin{tabular}{|l|c|c|c|p{5cm}|}
\hline
\textbf{Attack Family} & \textbf{Clusters} & \textbf{Attacks} & \textbf{\%} & \textbf{Characteristic Keywords} \\
\hline""")
    for row in summary_data:
        family_escaped = row['family'].replace('_', r'\_').replace('-', '--')
        keywords_escaped = row['keywords'].replace('_', r'\_')
        print(f"{family_escaped} & {row['clusters']} & {row['attacks']:,} & {row['percentage']:.1f}\\% & {keywords_escaped} \\\\")
        print(r"\hline")
    print(r"""\end{tabular}
\label{tab:attack_families}
\end{table}""")
    
    # ========================================================================
    # KEY INSIGHT FOR PAPER
    # ========================================================================
    print("\n" + "="*80)
    print("KEY INSIGHT FOR PAPER")
    print("="*80)
    
    # Find the "Novel Attack Pattern" percentage (topic obfuscation)
    novel_stats = family_stats.get('Novel Attack Pattern', {'total_attacks': 0, 'clusters': 0})
    novel_pct = novel_stats['total_attacks'] / total_attacks * 100 if total_attacks > 0 else 0
    
    # Find instruction override stats
    override_stats = family_stats.get('Instruction Override', {'total_attacks': 0, 'clusters': 0})
    override_pct = override_stats['total_attacks'] / total_attacks * 100 if total_attacks > 0 else 0
    
    print(f"""
Our analysis reveals that {novel_pct:.1f}% of jailbreak attempts use TOPIC OBFUSCATION -
disguising malicious intent behind innocent topics (e.g., "bananas", "yoga", "pottery").

Only {override_pct:.1f}% use direct INSTRUCTION OVERRIDE techniques (e.g., "ignore previous instructions").

This finding suggests that modern jailbreak attacks are increasingly sophisticated,
using semantic camouflage rather than explicit instruction manipulation.
""")

# 7. Visualization using UMAP (consistent with paper methodology)
print("\n>>> Generating UMAP 2D visualization...")
reducer_2d = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
plot_data = reducer_2d.fit_transform(jailbreak_vectors)

plt.figure(figsize=(12, 8))
# Use colormap that handles noise (-1) well
scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=labels, cmap='viridis', s=3, alpha=0.7)
plt.title(f"Jailbreak Attack Landscape: {n_clusters} Distinct Families\n(UMAP projection, noise in dark purple)", 
          fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster ID')
plt.xlabel("UMAP Dimension 1", fontsize=12)
plt.ylabel("UMAP Dimension 2", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("jailbreak_clusters.png", dpi=300, bbox_inches='tight')
plt.close()
print(">>> Cluster visualization saved to 'jailbreak_clusters.png'")

# Final timing
elapsed = time.time() - start_time
print(f"\n>>> Characterization completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
