# el4ner/models.py (Final, Stable Version)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
# We can now import this directly from utils again for consistency
from utils import configure_model_and_tokenizer


def load_el4ner_models():
    """
    Loads the models for the core EL4NER pipeline using standard, non-gated models.
    """
    print("Loading models for the EL4NER pipeline...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    backbone_models = {}
    model_ids = {
        "phi": "microsoft/Phi-3-mini-4k-instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",  # <-- REPLACED GLM/Llama3
        "qwen": "Qwen/Qwen2-7B-Instruct"
    }

    for name, model_id in model_ids.items():
        print(f"Loading {name}...")
        # All models are now standard and do not need trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config
        )
        model, tokenizer = configure_model_and_tokenizer(model, tokenizer)
        backbone_models[name] = (model, tokenizer)

    print("Loading similarity model...")
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    print("EL4NER models loaded successfully!")
    return backbone_models, similarity_model