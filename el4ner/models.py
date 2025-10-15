import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

def load_all_models():
    """
    Loads and configures all the necessary models for the EL4NER pipeline.
    - Backbone sLLMs with 4-bit quantization.
    - Sentence Transformer for semantic similarity.
    """
    print("Loading all models. This might take a while...")

    # Use 4-bit quantization to save memory
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- Load Backbone Models ---
    print("Loading Phi-3...")
    model_id_phi = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer_phi = AutoTokenizer.from_pretrained(model_id_phi)
    model_phi = AutoModelForCausalLM.from_pretrained(
        model_id_phi, device_map="auto", quantization_config=quantization_config, trust_remote_code=True
    )

    print("Loading GLM-4...")
    model_id_glm = "THUDM/glm-4-9b-chat"
    tokenizer_glm = AutoTokenizer.from_pretrained(model_id_glm, trust_remote_code=True)
    model_glm = AutoModelForCausalLM.from_pretrained(
        model_id_glm, device_map="auto", quantization_config=quantization_config, trust_remote_code=True
    )

    print("Loading Qwen2...")
    model_id_qwen = "Qwen/Qwen2-7B-Instruct"
    tokenizer_qwen = AutoTokenizer.from_pretrained(model_id_qwen)
    model_qwen = AutoModelForCausalLM.from_pretrained(
        model_id_qwen, device_map="auto", quantization_config=quantization_config
    )

    backbone_models = {
        "phi": (model_phi, tokenizer_phi),
        "glm": (model_glm, tokenizer_glm),
        "qwen": (model_qwen, tokenizer_qwen)
    }

    # --- Load Sentence Transformer for Similarity ---
    print("Loading similarity model...")
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    print("All models loaded successfully!")
    return backbone_models, similarity_model