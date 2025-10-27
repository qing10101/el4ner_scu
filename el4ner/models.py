# el4ner/models.py (Refactored with selective trust_remote_code)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from utils import MODELS_REQUIRING_REMOTE_CODE, configure_model_and_tokenizer


def load_el4ner_models():
    """
    Loads ONLY the models required for the core EL4NER pipeline,
    applying trust_remote_code selectively.
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
        should_trust = model_id in MODELS_REQUIRING_REMOTE_CODE

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=should_trust)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=should_trust
        )

        model, tokenizer = configure_model_and_tokenizer(model, tokenizer)
        backbone_models[name] = (model, tokenizer)

    print("Loading similarity model...")
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    print("EL4NER models loaded successfully!")
    return backbone_models, similarity_model