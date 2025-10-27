# compare_ner_methods.py (Refactored and Final Version)

import json
import textwrap

import torch
from transformers import BitsAndBytesConfig

# --- Import from our central toolbox ---
from utils import (
    load_model,
    clear_memory,
    run_single_llm_ner  # Using the baseline NER from utils
)


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


def main():
    print("✅ Running Qualitative Side-by-Side Comparison.")
    print("Models will be loaded and unloaded sequentially to conserve memory.")

    sample_texts = [
        "The new iPhone was announced by Apple in California, but Google is launching the Pixel in New York.",
        "After the concert at Madison Square Garden, Taylor Swift was seen leaving with Travis Kelce.",
        "Microsoft is competing with NVIDIA in the AI chip market, with a new datacenter planned for Virginia."
    ]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    for text in sample_texts:
        results = {}

        # --- Run Llama 3.3 70B ---
        print(f"\n--- Loading Llama 3.3 70B for: \"{text[:50]}...\" ---")
        llama_model, llama_tokenizer = load_model("meta-llama/Llama-3.3-70B-Instruct", quantization_config)
        results['Powerful LLM (Llama-3.3-70B)'] = run_single_llm_ner(text, llama_model, llama_tokenizer)
        clear_memory(llama_model, llama_tokenizer)

        # --- Run Phi-3 ---
        print(f"\n--- Loading Phi-3 for: \"{text[:50]}...\" ---")
        phi_model, phi_tokenizer = load_model("microsoft/Phi-3-mini-4k-instruct", quantization_config)
        results['Single Small LLM (Phi-3)'] = run_single_llm_ner(text, phi_model, phi_tokenizer)
        clear_memory(phi_model, phi_tokenizer)

        print_comparison(text, results)


if __name__ == "__main__":
    main()