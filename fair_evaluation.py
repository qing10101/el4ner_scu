# fair_evaluation.py (Final, Refactored Version)

import json
import random
import argparse
import torch
from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util

# --- Import from our central toolbox ---
from utils import (
    load_model,
    clear_memory,
    load_and_prepare_dataset
)
from el4ner.pipeline import run_el4ner_pipeline


# --- Fair Baseline Implementation ---

def retrieve_simple_demos(text, source_pool, similarity_model, k=5):
    """
    A simple retrieval mechanism for our baseline models.
    Finds the top-k most semantically similar sentences from the source pool.
    """
    text_embedding = similarity_model.encode(text, convert_to_tensor=True)
    # Pre-computing all pool embeddings would be faster for large-scale runs,
    # but this is fine for sample-based evaluation.
    pool_embeddings = similarity_model.encode([item['text'] for item in source_pool], convert_to_tensor=True,
                                              show_progress_bar=False)

    cos_scores = util.pytorch_cos_sim(text_embedding, pool_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(k, len(source_pool)))

    return [source_pool[i] for i in top_results.indices]


def format_baseline_demos(demos):
    demo_str = ""
    for demo in demos:
        demo_str += f"Text: \"{demo['text']}\"\nJSON: {json.dumps(demo['entities'])}\n\n"
    return demo_str


def run_retrieval_augmented_llm_ner(text, demos, model, tokenizer):
    """
    Runs a single LLM with dynamically retrieved demonstrations. THIS IS THE FAIR BASELINE.
    """
    demo_prompt = format_baseline_demos(demos)

    prompt = f"""You are an expert at Named Entity Recognition. Your task is to extract entities from the given text.
The valid entity types are: person, location, organization, product, creative-work, corporation, group.
Respond ONLY with a valid JSON object where keys are the extracted entities and values are their types.

--- EXAMPLES ---
{demo_prompt}--- END EXAMPLES ---

Text: "{text}"
JSON:"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        json_part = response.split("JSON:")[1].strip()
        cleaned_json = json_part.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_json) if cleaned_json else {}
    except (IndexError, json.JSONDecodeError):
        return {"error": "Failed to parse JSON response."}


def convert_to_iob2(text, entities):
    """Converts a text and a dictionary of entities to IOB2 format."""
    tokens = text.split()
    tags = ['O'] * len(tokens)
    if not entities or 'error' in entities:
        return tokens, tags
    for entity_text, entity_type in entities.items():
        entity_tokens = entity_text.split()
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i + len(entity_tokens)] == entity_tokens:
                tags[i] = f'B-{entity_type}'
                for j in range(1, len(entity_tokens)):
                    tags[i + j] = f'I-{entity_type}'
                break
    return tokens, tags


# --- Main Evaluation Function ---

def main(args):
    print("Starting FAIR Quantitative NER Performance Evaluation...")

    test_data = load_and_prepare_dataset(split='test')
    if args.num_samples >= len(test_data):
        print(
            f"Warning: Number of samples ({args.num_samples}) is >= test set size ({len(test_data)}). Using the entire test set.")
        args.num_samples = len(test_data)

    random.seed(42)  # for reproducibility
    sampled_data = random.sample(test_data, args.num_samples)
    print(f"Randomly sampled {len(sampled_data)} examples from the WNUT17 test set.")

    all_true_iob, all_preds_iob = [], {"EL4NER (Ensemble)": [], "Powerful LLM (Qwen3-30B)": [],
                                       "Single Small LLM (Phi-3)": []}

    detailed_results = []

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    with open('data/wnut17_source_pool.json', 'r') as f:
        source_pool = json.load(f)
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    for item in tqdm(sampled_data, desc="Evaluating Samples"):
        text, ground_truth = item['text'], item['entities']
        _, true_tags = convert_to_iob2(text, ground_truth)
        all_true_iob.append(true_tags)
        sample_result = {"text": text, "ground_truth": ground_truth}

        # --- Run EL4NER ---
        backbone_models = {}
        for name, model_id in {"phi": "microsoft/Phi-3-mini-4k-instruct", "glm": "THUDM/glm-4-9b-chat",
                               "qwen": "Qwen/Qwen2-7B-Instruct"}.items():
            backbone_models[name] = load_model(model_id, quantization_config)
        el4ner_preds = run_el4ner_pipeline(text, source_pool, backbone_models, similarity_model, k=5, verifier='glm')
        _, el4ner_tags = convert_to_iob2(text, el4ner_preds)
        all_preds_iob["EL4NER (Ensemble)"].append(el4ner_tags)
        sample_result["el4ner_prediction"] = el4ner_preds
        clear_memory(backbone_models)

        # --- Prepare Demos for Baselines ---
        baseline_demos = retrieve_simple_demos(text, source_pool, similarity_model, k=5)

        # --- Run Qwen3-30B ---
        qwen3_model, qwen3_tokenizer = load_model("Qwen/Qwen3-30B-A3B-Instruct-2507", quantization_config)
        qwen3_preds = run_retrieval_augmented_llm_ner(text, baseline_demos, qwen3_model, qwen3_tokenizer)
        _, qwen3_tags = convert_to_iob2(text, qwen3_preds)
        # Be sure to update the dictionary key for the final report
        all_preds_iob["Powerful LLM (Qwen3-30B)"].append(qwen3_tags)
        sample_result["qwen3_prediction"] = qwen3_preds
        clear_memory(qwen3_model, qwen3_tokenizer)

        # --- Run Phi-3 ---
        phi_model, phi_tokenizer = load_model("microsoft/Phi-3-mini-4k-instruct", quantization_config)
        phi_preds = run_retrieval_augmented_llm_ner(text, baseline_demos, phi_model, phi_tokenizer)
        _, phi_tags = convert_to_iob2(text, phi_preds)
        all_preds_iob["Single Small LLM (Phi-3)"].append(phi_tags)
        sample_result["phi3_prediction"] = phi_preds
        clear_memory(phi_model, phi_tokenizer)

        detailed_results.append(sample_result)

    clear_memory(similarity_model)

    print("\n\n" + "=" * 80)
    print("           FAIR QUANTITATIVE EVALUATION RESULTS (WNUT17 TEST SET)")
    print("=" * 80)
    for method_name, pred_tags in all_preds_iob.items():
        print(f"\n--- Performance Report for: {method_name} ---\n")
        report = classification_report(all_true_iob, pred_tags, mode='strict', scheme=IOB2)
        print(report)
        print("-" * 80)

    with open(args.output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"\n✅ Detailed results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a FAIR quantitative evaluation of NER methods.")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of random samples to draw from the test set.")
    parser.add_argument("--output_file", type=str, default="fair_evaluation_results.json",
                        help="File to save the detailed evaluation results.")
    args = parser.parse_args()
    main(args)