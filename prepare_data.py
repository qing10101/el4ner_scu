#### 4. `prepare_data.py` (Utility to download and format the data)
import json
from datasets import load_dataset
import os

def create_source_pool_from_wnut17():
    """
    Downloads the WNUT17 dataset and converts the training split
    into the JSON format required for the EL4NER source pool.
    """
    print("Loading WNUT17 dataset from Hugging Face...")
    # Load the dataset
    dataset = load_dataset("wnut_17")
    train_data = dataset['train']

    # NER tags mapping from IDs to labels
    ner_tags = train_data.features['ner_tags'].feature.names

    source_pool = []

    print("Processing train split to create source pool...")
    for entry in train_data:
        tokens = entry['tokens']
        tags = [ner_tags[tag_id] for tag_id in entry['ner_tags']]

        text = " ".join(tokens)
        entities = {}
        current_entity_tokens = []
        current_entity_tag = None

        for token, tag in zip(tokens, tags):
            if tag.startswith('B-'):
                if current_entity_tokens:
                    entity_name = " ".join(current_entity_tokens)
                    entities[entity_name] = current_entity_tag
                current_entity_tokens = [token]
                current_entity_tag = tag[2:] # Get tag name without 'B-'
            elif tag.startswith('I-') and current_entity_tag == tag[2:]:
                current_entity_tokens.append(token)
            else:
                if current_entity_tokens:
                    entity_name = " ".join(current_entity_tokens)
                    entities[entity_name] = current_entity_tag
                current_entity_tokens = []
                current_entity_tag = None

        # Add the last entity if it exists
        if current_entity_tokens:
            entity_name = " ".join(current_entity_tokens)
            entities[entity_name] = current_entity_tag

        if entities:
            source_pool.append({
                "text": text,
                "entities": entities
            })

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    output_path = 'data/wnut17_source_pool.json'
    print(f"Saving source pool to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(source_pool, f, indent=2)

    print(f"Successfully created source pool with {len(source_pool)} entries.")

if __name__ == "__main__":
    create_source_pool_from_wnut17()