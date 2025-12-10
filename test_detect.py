#!/usr/bin/env python3

import xgboost as xgb
from sentence_transformers import SentenceTransformer
import numpy as np
import sys

def debug_print(msg):
    print(msg, file=sys.stderr)
    sys.stderr.flush()

print("Starting test...")

try:
    debug_print("Loading embedder...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    debug_print("Embedder loaded")

    debug_print("Loading XGBoost model...")
    clf = xgb.XGBClassifier()
    clf.load_model('jailbreak_detector.json')
    debug_print("Model loaded")

    text = "What's the weather like today?"
    debug_print(f"Classifying: '{text}'")

    debug_print("Generating embedding...")
    embedding = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    debug_print(f"Embedding shape: {embedding.shape}")
    debug_print(f"Embedding sample: {embedding[0][:5]}")

    X = embedding.reshape(1, -1)
    debug_print(f"X shape: {X.shape}")

    debug_print("Making prediction...")
    prediction = clf.predict(X)[0]
    debug_print(f"Prediction: {prediction}")

    debug_print("Getting probabilities...")
    probability = clf.predict_proba(X)[0]
    debug_print(f"Probability: {probability}")

    status = "🚨 JAILBREAK" if prediction else "✅ SAFE"
    confidence = max(probability) * 100

    print(f"Result: {status}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Probabilities: Safe: {probability[0]:.3f}, Jailbreak: {probability[1]:.3f}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
