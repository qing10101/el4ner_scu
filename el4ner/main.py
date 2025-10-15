import argparse
import json
from models import load_all_models
from pipeline import run_el4ner_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the EL4NER pipeline on an input text.")
    parser.add_argument("--text", type=str, required=True, help="The input text to perform NER on.")
    parser.add_argument("--data_path", type=str, default="data/wnut17_source_pool.json", help="Path to the source pool JSON file.")
    parser.add_argument("--k", type=int, default=5, help="Number of demonstrations to retrieve.")
    parser.add_argument("--verifier", type=str, default="glm", choices=["phi", "glm", "qwen"], help="Model to use for the verification step.")
    args = parser.parse_args()

    # Load models
    backbone_models, similarity_model = load_all_models()

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

# ---
#
# ### Part 2: Which Dataset to Use and How
#
# For this implementation, I recommend using the **WNUT17** dataset.
#
# **Why WNUT17?**
# 1.  **Accessibility:** It is publicly and easily available on the Hugging Face Hub. Unlike ACE2005, it doesn't require a license.
# 2.  **Simplicity:** The data format is well-structured and straightforward to parse.
# 3.  **Relevance:** It was one of the key datasets used for evaluation in the original EL4NER paper, so it's a fitting choice.
#
# #### How the Dataset is Used for Extraction and Classification
#
# You use the **same source pool** for both stages. The difference is in *what information you format* for the demonstration prompts.
#
# *   The `prepare_data.py` script creates a single file: `data/wnut17_source_pool.json`. This is your entire reference library.
# *   **For the Span Extraction stage**, the `format_extraction_demos` function in `prompts.py` takes an entry from this file and only uses the `text` and the *keys* of the `entities` dictionary (the spans themselves).
# *   **For the Span Classification stage**, the `format_classification_demos` function uses the `text` and the full `entities` dictionary (both the spans and their types).
#
# This approach is efficient because you only need to load and process one dataset, which then serves as the foundation for all the in-context learning steps.