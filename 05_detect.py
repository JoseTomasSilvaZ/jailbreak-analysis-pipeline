import xgboost as xgb
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class JailbreakDetector:
    def __init__(self, model_path="jailbreak_detector.json"):
        # Detect device (same as training)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load the embedding model (same as training)
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.embedder.half()  # Use half precision like training

        # Load the trained XGBoost model
        print("Loading XGBoost model...")
        self.clf = xgb.XGBClassifier()
        self.clf.load_model(model_path)

        print("✅ Jailbreak detector ready!")

    def classify_text(self, text):
        """Classify if a text is a jailbreak attempt

        Args:
            text (str): Input text to classify

        Returns:
            dict: Classification result with prediction and confidence
        """
        # Generate embedding (same parameters as training)
        embedding = self.embedder.encode(
            [text],
            batch_size=1,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Reshape for XGBoost (single sample)
        X = embedding.reshape(1, -1)

        # Get prediction and probability
        prediction = self.clf.predict(X)[0]
        probability = self.clf.predict_proba(X)[0]

        # Format result
        result = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "is_jailbreak": bool(prediction),
            "confidence": float(max(probability)),
            "probabilities": {
                "safe": float(probability[0]),
                "jailbreak": float(probability[1])
            }
        }

        return result

    def interactive_mode(self):
        """Run interactive classification mode"""
        print("\n🛡️  Jailbreak Detector Interactive Mode")
        print("Enter text to classify (or 'quit' to exit)")
        print("-" * 50)

        while True:
            try:
                text = input("\nEnter text: ").strip()

                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 🛡️")
                    break

                if not text:
                    print("Please enter some text.")
                    continue

                result = self.classify_text(text)

                # Display result
                status = "🚨 JAILBREAK DETECTED" if result["is_jailbreak"] else "✅ SAFE"
                confidence = result["confidence"] * 100

                print(f"\nResult: {status}")
                print(".1f")
                print(f"Probabilities: Safe: {result['probabilities']['safe']:.3f}, Jailbreak: {result['probabilities']['jailbreak']:.3f}")

            except KeyboardInterrupt:
                print("\nGoodbye! 🛡️")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function for command line usage"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 05_detect.py \"your text here\"")
        print("Or run without arguments for interactive mode")
        return

    # Initialize detector
    detector = JailbreakDetector()

    # Classify the provided text
    text = " ".join(sys.argv[1:])
    result = detector.classify_text(text)

    # Display result
    status = "🚨 JAILBREAK DETECTED" if result["is_jailbreak"] else "✅ SAFE"
    confidence = result["confidence"] * 100

    print(f"\nResult: {status}")
    print(".1f")
    print(f"Probabilities: Safe: {result['probabilities']['safe']:.3f}, Jailbreak: {result['probabilities']['jailbreak']:.3f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # No arguments = interactive mode
        detector = JailbreakDetector()
        detector.interactive_mode()
    else:
        # Arguments provided = classify and exit
        main()

