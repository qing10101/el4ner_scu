
import json
import random
import argparse
import torch
import gc
from tqdm import tqdm
from datasets import load_dataset
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Import and reuse components from our existing scripts ---
from el4ner.pipeline import run_el4ner_pipeline


# --- Helper Functions ---

def clear_memory(*args):
    """Clears models and CUDA cache from memory."""
    for model in args:
        if isinstance(model, dict):
            for m in model.values(): del m
        else:
            del model
    gc.collect()
    torch.cuda.empty_cache()


def convert_to_iob2(text, entities):
    """Converts a text and a dictionary of entities to IOB2 format."""
    # A simple whitespace tokenizer; for production, a more robust tokenizer would be better.
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


def load_and_prepare_test_set():
    """Loads the WNUT17 test set and formats it for our evaluation."""
    # (This function is the same as before)
    print("Loading WNUT17 test set...")
    dataset = load_dataset("wnut_17", split="test")
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


# --- Fair Baseline Implementation ---

def retrieve_simple_demos(text, source_pool, similarity_model, k=5):
    """
    A simple retrieval mechanism for our baseline models.
    Finds the top-k most semantically similar sentences from the source pool.
    """
    text_embedding = similarity_model.encode(text, convert_to_tensor=True)
    pool_embeddings = similarity_model.encode([item['text'] for item in source_pool], convert_to_tensor=True)

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
The allowed Named Entity categories are Person, Location, Group, Creative work, Corporation, Product

--- EXAMPLES ---
{demo_prompt}--- END EXAMPLES ---

Text: "{text}"
JSON:"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

    outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        json_part = response.split("JSON:")[1].strip()
        cleaned_json = json_part.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_json) if cleaned_json else {}
    except (IndexError, json.JSONDecodeError):
        return {"error": "Failed to parse JSON response."}


# --- Main Evaluation Function ---

def main(args):
    print("Starting FAIR Quantitative NER Performance Evaluation...")

    test_data = load_and_prepare_test_set()
    if args.num_samples >= len(test_data):
        print(
            f"Warning: Number of samples ({args.num_samples}) is >= test set size ({len(test_data)}). Using the entire test set.")
        args.num_samples = len(test_data)

    random.seed(42)  # for reproducibility
    sampled_data = random.sample(test_data, args.num_samples)
    print(f"Randomly sampled {len(sampled_data)} examples from the WNUT17 test set.")

    all_true_iob, all_preds_iob = [], {"EL4NER (Ensemble)": [], "Powerful LLM (Llama-3.3-70B)": [],
                                       "Single Small LLM (Phi-3)": []}
    detailed_results = []

    quantization_config = torch.nn.Identity()  # Placeholder if not using quantization. Replace with BitsAndBytesConfig if needed.

    # Load the source pool once for all models
    with open('data/wnut_17_train.json', 'r') as f:
        source_pool = json.load(f)

    # Load the similarity model once for all models
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    for item in tqdm(sampled_data, desc="Evaluating Samples"):
        text, ground_truth = item['text'], item['entities']
        _, true_tags = convert_to_iob2(text, ground_truth)
        all_true_iob.append(true_tags)

        sample_result = {"text": text, "ground_truth": ground_truth}

        # --- Run EL4NER ---
        model_ids = {"phi": "microsoft/Phi-3-mini-4k-instruct", "glm": "THUDM/glm-4-9b-chat",
                     "qwen": "Qwen/Qwen2-7B-Instruct"}
        backbone_models = {}
        for name, model_id in model_ids.items():
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                         quantization_config=quantization_config,
                                                         trust_remote_code=True)
            backbone_models[name] = (model, tokenizer)

        el4ner_preds = run_el4ner_pipeline(text, source_pool, backbone_models, similarity_model, k=5, verifier='glm')
        _, el4ner_tags = convert_to_iob2(text, el4ner_preds)
        all_preds_iob["EL4NER (Ensemble)"].append(el4ner_tags)
        sample_result["el4ner_prediction"] = el4ner_preds
        clear_memory(backbone_models)

        # --- Prepare Demos for Baselines ---
        baseline_demos = retrieve_simple_demos(text, source_pool, similarity_model, k=5)

        # --- Run Llama 3.3 70B ---
        llama_id = "meta-llama/Llama-3.3-70B-Instruct"
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_id)
        llama_model = AutoModelForCausalLM.from_pretrained(llama_id, device_map="auto",
                                                           quantization_config=quantization_config)
        llama_preds = run_retrieval_augmented_llm_ner(text, baseline_demos, llama_model, llama_tokenizer)
        _, llama_tags = convert_to_iob2(text, llama_preds)
        all_preds_iob["Powerful LLM (Llama-3.3-70B)"].append(llama_tags)
        sample_result["llama_prediction"] = llama_preds
        clear_memory(llama_model, llama_tokenizer)

        # --- Run Phi-3 ---
        phi_id = "microsoft/Phi-3-mini-4k-instruct"
        phi_tokenizer = AutoTokenizer.from_pretrained(phi_id)
        phi_model = AutoModelForCausalLM.from_pretrained(phi_id, device_map="auto",
                                                         quantization_config=quantization_config,
                                                         trust_remote_code=True)
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