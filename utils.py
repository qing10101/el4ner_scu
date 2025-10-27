# utils.py (Refactored with selective trust_remote_code)

import json
import torch
import gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Centralized Model and Memory Management ---

# A set of model IDs that are known to require custom remote code.
# This makes our loading function much safer and more precise.
MODELS_REQUIRING_REMOTE_CODE = {
    "THUDM/glm-4-9b-chat"
}


def configure_model_and_tokenizer(model, tokenizer):
    """A robust function to correctly configure special tokens for any model."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_model(model_id, quantization_config):
    """Loads a single model and tokenizer with correct configuration."""
    # Selectively apply the trust_remote_code flag
    should_trust = model_id in MODELS_REQUIRING_REMOTE_CODE

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=should_trust)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=should_trust
    )
    model, tokenizer = configure_model_and_tokenizer(model, tokenizer)
    return model, tokenizer


def clear_memory(*args):
    """Clears models and CUDA cache from memory."""
    for model in args:
        if isinstance(model, dict):
            for m_tuple in model.values():
                del m_tuple[0]  # model
                del m_tuple[1]  # tokenizer
        else:
            del model
    gc.collect()
    torch.cuda.empty_cache()


# --- Centralized Data Loading and Formatting ---

def load_and_prepare_dataset(split='train'):
    """
    Loads a specific split of the WNUT17 dataset and formats it.
    Includes trust_remote_code=True to align with best practices.
    """
    print(f"Loading WNUT17 '{split}' split...")
    dataset = load_dataset("wnut_17", split=split, trust_remote_code=True)
    ner_tags = dataset.features['ner_tags'].feature.names

    formatted_data = []
    for entry in dataset:
        tokens = entry['tokens']
        tags = [ner_tags[tag_id] for tag_id in entry['ner_tags']]
        text = " ".join(tokens)

        entities = {}
        current_tokens = []
        current_tag = None
        for token, tag in zip(tokens, tags):
            if tag.startswith('B-'):
                if current_tokens: entities[" ".join(current_tokens)] = current_tag
                current_tokens = [token]
                current_tag = tag[2:]
            elif tag.startswith('I-') and current_tag == tag[2:]:
                current_tokens.append(token)
            else:
                if current_tokens: entities[" ".join(current_tokens)] = current_tag
                current_tokens, current_tag = [], None
        if current_tokens: entities[" ".join(current_tokens)] = current_tag

        if entities:
            formatted_data.append({"text": text, "entities": entities})

    return formatted_data


# --- Centralized Baseline NER Logic ---

SINGLE_LLM_FEW_SHOT_PROMPT = """You are an expert at Named Entity Recognition. Your task is to extract entities from the given text.
The valid entity types are: person, location, organization, product, creative-work, corporation, group.
Respond ONLY with a valid JSON object where keys are the extracted entities and values are their types.

--- EXAMPLES ---
Text: "The new MacBook Pro was unveiled by Tim Cook in Cupertino."
JSON: {{"MacBook Pro": "product", "Tim Cook": "person", "Cupertino": "location"}}

Text: "We are flying with United Airlines to Chicago to watch the new Marvel movie."
JSON: {{"United Airlines": "corporation", "Chicago": "location", "Marvel": "organization"}}
--- END EXAMPLES ---

Text: "{text}"
JSON:"""


def run_single_llm_ner(text, model, tokenizer):
    """Runs a simple few-shot NER task on any given model."""
    prompt = SINGLE_LLM_FEW_SHOT_PROMPT.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        json_part = response.split("JSON:")[1].strip()
        cleaned_json = json_part.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_json) if cleaned_json else {}
    except (IndexError, json.JSONDecodeError):
        return {"error": "Failed to parse JSON response."}