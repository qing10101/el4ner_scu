# el4ner/models.py (Corrected Version)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer


def _configure_model_and_tokenizer(model, tokenizer):
    """
    A robust function to correctly configure special tokens for any model.
    This is the key fix for the GLM-4 slowdown.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Also configure the model's internal config
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def load_el4ner_models():
    """
    Loads ONLY the models required for the core EL4NER pipeline.
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
        "glm": "THUDM/glm-4-9b-chat",
        "qwen": "Qwen/Qwen2-7B-Instruct"
    }

    for name, model_id in model_ids.items():
        print(f"Loading {name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=quantization_config, trust_remote_code=True
        )

        # Apply the fix
        model, tokenizer = _configure_model_and_tokenizer(model, tokenizer)
        backbone_models[name] = (model, tokenizer)

    print("Loading similarity model...")
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    print("EL4NER models loaded successfully!")
    return backbone_models, similarity_model