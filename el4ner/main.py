# el4ner/main.py (Corrected and Final Version)

import argparse
import json
import nltk

# Import from the refactored models.py
from el4ner.models import load_el4ner_models
from el4ner.pipeline import run_el4ner_pipeline

def setup_nltk():
    """Checks for and downloads necessary NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading necessary NLTK data (punkt, averaged_perceptron_tagger)...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

def main():
    parser = argparse.ArgumentParser(description="Run the core EL4NER pipeline on an input text.")
    parser.add_argument("--text", type=str, required=True, help="The input text to perform NER on.")
    parser.add_argument("--data_path", type=str, default="data/wnut17_source_pool.json", help="Path to the source pool JSON file.")
    parser.add_argument("--k", type=int, default=5, help="Number of demonstrations to retrieve.")
    parser.add_argument("--verifier", type=str, default="glm", choices=["phi", "glm", "qwen"], help="Model to use for the verification step.")
    args = parser.parse_args()

    # Perform one-time setup for NLTK
    setup_nltk()

    # Load models using the corrected function name
    backbone_models, similarity_model = load_el4ner_models()

    # Load source pool data
    print(f"Loading source pool from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        source_pool = json.load(f)

    # Run the full pipeline
    print("\n--- Starting EL4NER Pipeline ---")
    final_entities = run_el4ner_pipeline(
        text=args.text,
        source_pool=source_pool,
        models=backbone_models,
        similarity_model=similarity_model,
        k=args.k,
        verifier=args.verifier
    )
    print("\n--- EL4NER Final Output ---")
    print(f"Input Text: {args.text}")
    print(f"Named Entities Found: {final_entities}")
    print("-----------------------------\n")

if __name__ == "__main__":
    main()