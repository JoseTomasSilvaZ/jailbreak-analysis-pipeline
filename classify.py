#!/usr/bin/env python3
"""
Jailbreak Classifier - Pass text as argument
Usage: python classify.py "your text here"
"""
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer

if len(sys.argv) < 2:
    print('Usage: python classify.py "your text here"')
    sys.exit(1)

text = " ".join(sys.argv[1:])

print(f"Text: {text}")
print("Loading models...", end=" ", flush=True)

embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
clf = xgb.XGBClassifier()
clf.load_model('jailbreak_detector.json')
print("done.", flush=True)

print("Classifying...", end=" ", flush=True)
embedding = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
pred = clf.predict(embedding)[0]
probs = clf.predict_proba(embedding)[0]
print("done.", flush=True)

label = "⚠️  JAILBREAK" if pred == 1 else "✅ SAFE"
print(f"\nResult: {label}")
print(f"Jailbreak probability: {probs[1]:.1%}")
print(f"Safe probability: {probs[0]:.1%}")

