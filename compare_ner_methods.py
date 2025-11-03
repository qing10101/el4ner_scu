# compare_ner_methods.py (Final Version with 3-Way Comparison)

import json
import textwrap

import torch
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# --- Import from our central toolbox and the core pipeline ---
from utils import (
    load_model,
    clear_memory,
    run_single_llm_ner
)
from el4ner.pipeline import run_el4ner_pipeline


def print_comparison(text, results):
    """Prints a nicely formatted comparison table for a single text sample."""
    print("\n" + "=" * 80)
    print("ANALYZING TEXT:")
    print(textwrap.fill(text, width=80))
    print("=" * 80)

    print("\n{:<35} | {:<45}".format("NER METHOD", "RESULTS"))
    print("-" * 80)

    # Define the desired order for printing results
    method_order = ["EL4NER (Ensemble)", "Powerful LLM (Qwen3-30B)", "Single Small LLM (Phi-3)"]

    for method in method_order:
        if method in results:
            result = results[method]
            result_str = json.dumps(result, indent=4)
            print("{:<35} | ".format(method))
            for line in result_str.splitlines():
                print("{:<35} | {}".format("", line))
            print("-" * 80)


def main():
    """
    Runs a qualitative, side-by-side comparison of all three NER methods
    on a few sample sentences. This script is intended for clear demos.
    """
    print("✅ Running Full 3-Way Qualitative Comparison.")
    print("Models will be loaded and unloaded sequentially to conserve VRAM. This will take some time.")

    sample_texts = [
        "The new iPhone was announced by Apple in California, but Google is launching the Pixel in New York.",
        "After the concert at Madison Square Garden, Taylor Swift was seen leaving with Travis Kelce.",
        "Microsoft is competing with NVIDIA in the AI chip market, with a new datacenter planned for Virginia."
    ]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the source pool and similarity model once, as they are needed for EL4NER
    print("Loading shared resources (source pool, similarity model)...")
    with open('data/wnut17_source_pool.json', 'r') as f:
        source_pool = json.load(f)
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    for text in sample_texts:
        results = {}

        # --- 1. Run the full EL4NER pipeline ---
        print(f"\n--- Loading EL4NER Ensemble for: \"{text[:50]}...\" ---")
        backbone_models = {}
        # Make sure this list matches the final models in el4ner/models.py
        for name, model_id in {"phi": "microsoft/Phi-3-mini-4k-instruct",
                               "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
                               "qwen": "Qwen/Qwen2-7B-Instruct"}.items():
            backbone_models[name] = load_model(model_id, quantization_config)

        results['EL4NER (Ensemble)'] = run_el4ner_pipeline(
            text=text,
            source_pool=source_pool,
            models=backbone_models,
            similarity_model=similarity_model,
            k=5,
            verifier='mistral'  # Use a non-gated verifier
        )
        clear_memory(backbone_models)

        # --- 2. Run the Powerful Standalone LLM (Qwen3-30B) ---
        print(f"\n--- Loading Qwen3-30B for: \"{text[:50]}...\" ---")
        qwen3_model, qwen3_tokenizer = load_model("Qwen/Qwen3-30B-A3B-Instruct-2507", quantization_config)
        results['Powerful LLM (Qwen3-30B)'] = run_single_llm_ner(text, qwen3_model, qwen3_tokenizer)
        clear_memory(qwen3_model, qwen3_tokenizer)

        # --- 3. Run a Single Small LLM from the Ensemble (Phi-3) ---
        print(f"\n--- Loading Phi-3 for: \"{text[:50]}...\" ---")
        phi_model, phi_tokenizer = load_model("microsoft/Phi-3-mini-4k-instruct", quantization_config)
        results['Single Small LLM (Phi-3)'] = run_single_llm_ner(text, phi_model, phi_tokenizer)
        clear_memory(phi_model, phi_tokenizer)

        # Print the final, formatted table for this sentence
        print_comparison(text, results)

    # Clean up the last shared model
    clear_memory(similarity_model)


if __name__ == "__main__":
    main()