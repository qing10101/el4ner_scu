# prepare_data.py (Refactored)

import json
import os
from utils import load_and_prepare_dataset

# Add this to prepare_data.py
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data (punkt)...")
    nltk.download('punkt')


def main():
    """
    Creates the source pool for EL4NER by loading and processing the
    WNUT17 training data.
    """
    # Use our new, reusable utility function
    source_pool = load_and_prepare_dataset(split='train')

    if not os.path.exists('data'):
        os.makedirs('data')

    output_path = 'data/wnut17_source_pool.json'
    print(f"Saving source pool to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(source_pool, f, indent=2)

    print(f"Successfully created source pool with {len(source_pool)} entries.")

if __name__ == "__main__":
    main()