# compare_ner_methods.py (VRAM-Efficient Version for RTX A6000)

import json
import textwrap
import torch
import gc
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# We will load models inside functions now, not globally.
from el4ner.pipeline import run_el4ner_pipeline

# --- Define the baseline few-shot NER approach ---

SINGLE_LLM_FEW_SHOT_PROMPT = """You are an expert at Named Entity Recognition. Your task is to extract entities from the given text.
The valid entity types are: person, location, organization, product, creative-work, corporation.
Respond ONLY with a valid JSON object where keys are the extracted entities and values are their types.
The allowed Named Entity categories are Person, Location, Group, Creative work, Corporation, Product.

--- EXAMPLES ---
Text: "The new MacBook Pro was unveiled by Tim Cook in Cupertino."
JSON: {{"MacBook Pro": "product", "Tim Cook": "person", "Cupertino": "location"}}

Text: "We are flying with United Airlines to Chicago to watch the new Marvel movie."
JSON: {{"United Airlines": "corporation", "Chicago": "location", "Marvel": "organization"}}
--- END EXAMPLES ---

Text: "{text}"
JSON:"""


# ADD THIS HELPER FUNCTION AT THE TOP LEVEL
def _configure_model_and_tokenizer(model, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer



def run_single_llm_ner(text, model, tokenizer):
    """Runs a simple few-shot NER task on any given model."""
    prompt = SINGLE_LLM_FEW_SHOT_PROMPT.format(text=text)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        json_part = response.split("JSON:")[1].strip()
        cleaned_json = json_part.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_json)
    except (IndexError, json.JSONDecodeError) as e:
        return {"error": "Failed to parse JSON response.", "details": str(e)}


def print_comparison(text, results):
    """Prints a nicely formatted comparison table."""
    print("\n" + "=" * 80)
    print("ANALYZING TEXT:")
    print(textwrap.fill(text, width=80))
    print("=" * 80)

    print("\n{:<35} | {:<45}".format("NER METHOD", "RESULTS"))
    print("-" * 80)

    for method, result in results.items():
        result_str = json.dumps(result, indent=4)
        print("{:<35} | ".format(method))
        for line in result_str.splitlines():
            print("{:<35} | {}".format("", line))
        print("-" * 80)


def clear_memory(*args):
    """Clears models and CUDA cache from memory."""
    for model in args:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    print("✅ Running VRAM-Efficient Comparison for High-End GPUs (like RTX A6000).")
    print("Models will be loaded and unloaded sequentially to conserve memory.")

    # --- DEFINE SAMPLE TEXTS ---
    sample_texts = [
        "The new iPhone was announced by Apple in California, but Google is launching the Pixel in New York.",
        "After the concert at Madison Square Garden, Taylor Swift was seen leaving with Travis Kelce.",
        "Microsoft is competing with NVIDIA in the AI chip market, with a new datacenter planned for Virginia."
    ]

    # --- MODEL CONFIGS ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- RUN THE COMPARISON LOOP ---
    for text in sample_texts:
        results = {}

        # === 1. Run the full EL4NER pipeline ===
        print(f"\n--- Loading EL4NER Models for: \"{text[:50]}...\" ---")
        backbone_models = {}
        model_ids = {
            "phi": "microsoft/Phi-3-mini-4k-instruct",
            "glm": "THUDM/glm-4-9b-chat",
            "qwen": "Qwen/Qwen2-7B-Instruct"
        }
        for name, model_id in model_ids.items():
            # Only trust remote code for GLM and Qwen
            trust_code = True if name in ["glm", "qwen"] else False
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_code)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                         quantization_config=quantization_config,
                                                         trust_remote_code=trust_code)
            # APPLY THE FIX HERE
            model, tokenizer = _configure_model_and_tokenizer(model, tokenizer)
            backbone_models[name] = (model, tokenizer)

        similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        with open('data/wnut17_source_pool.json', 'r') as f:
            source_pool = json.load(f)

        print("Running EL4NER pipeline...")
        results['EL4NER (Ensemble)'] = run_el4ner_pipeline(text, source_pool, backbone_models, similarity_model, k=5,
                                                           verifier='glm')

        print("Unloading EL4NER models...")
        clear_memory(backbone_models, similarity_model)

        # === 2. Run the powerful standalone LLM (Llama 3.3 70B) ===
        print(f"\n--- Loading Llama 3.3 70B for: \"{text[:50]}...\" ---")
        # Ensure you have requested access on Hugging Face and are logged in via `huggingface-cli login`
        llama_id = "meta-llama/Llama-3.3-70B-Instruct"
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_id)
        llama_model = AutoModelForCausalLM.from_pretrained(llama_id, device_map="auto",
                                                           quantization_config=quantization_config)
        # APPLY THE FIX HERE
        llama_model, llama_tokenizer = _configure_model_and_tokenizer(llama_model, llama_tokenizer)

        print("Running Llama 3.3 70B...")
        results['Powerful LLM (Llama-3.3-70B)'] = run_single_llm_ner(text, llama_model, llama_tokenizer)

        print("Unloading Llama 3.3 70B...")
        clear_memory(llama_model, llama_tokenizer)

        # === 3. Run a single small LLM from the ensemble (Phi-3) ===
        print(f"\n--- Loading Phi-3 for: \"{text[:50]}...\" ---")
        phi_id = "microsoft/Phi-3-mini-4k-instruct"
        phi_tokenizer = AutoTokenizer.from_pretrained(phi_id)
        phi_model = AutoModelForCausalLM.from_pretrained(phi_id, device_map="auto",
                                                         quantization_config=quantization_config,
                                                         trust_remote_code=False)
        # APPLY THE FIX HERE
        phi_model, phi_tokenizer = _configure_model_and_tokenizer(phi_model, phi_tokenizer)

        print("Running Phi-3...")
        results['Single Small LLM (Phi-3)'] = run_single_llm_ner(text, phi_model, phi_tokenizer)

        print("Unloading Phi-3...")
        clear_memory(phi_model, phi_tokenizer)

        # --- Print the final comparison table for this text ---
        print_comparison(text, results)


if __name__ == "__main__":
    main()