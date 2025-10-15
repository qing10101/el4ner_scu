import torch
import re
import nltk
from sentence_transformers import util
from tqdm import tqdm
from collections import Counter
from prompts import (
    ZERO_SHOT_EXTRACTION_TEMPLATE, FEW_SHOT_EXTRACTION_TEMPLATE,
    FEW_SHOT_CLASSIFICATION_TEMPLATE, VERIFICATION_TEMPLATE,
    format_extraction_demos, format_classification_demos
)

# Initialize NLTK
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


def _run_llm_inference(prompt, model, tokenizer, max_new_tokens=100):
    """A helper function to run inference on a given model."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def _extract_spans_from_response(response, anchor_text="Entities:"):
    try:
        entities_part = response.split(anchor_text)[1].strip()
        cleaned_part = re.sub(r'^\[CLS\]|\[SEP\]$', '', entities_part).strip()
        spans = [span.strip() for span in cleaned_part.split("[SEP]") if span.strip()]
        return list(set(spans))
    except IndexError:
        return []


# --- Stage 1 Functions ---

def extract_spans_zero_shot(text, model, tokenizer):
    prompt = ZERO_SHOT_EXTRACTION_TEMPLATE.format(text=text)
    response = _run_llm_inference(prompt, model, tokenizer)
    return _extract_spans_from_response(response)


def get_pos_weight(span):
    if not span: return 0
    first_word = span.split()[0]
    pos_tag = nltk.pos_tag(nltk.word_tokenize(first_word))
    if not pos_tag: return 0
    tag = pos_tag[0][1]

    if tag in ['NNP', 'NNPS']:
        return 4
    elif tag in ['NN', 'NNS']:
        return 2
    elif tag in ['PRP', 'PRP$']:
        return 1
    else:
        return 0


def calculate_span_similarity(input_spans, candidate_spans, similarity_model):
    if not input_spans or not candidate_spans: return 0.0
    input_embeddings = similarity_model.encode(input_spans, convert_to_tensor=True)
    candidate_embeddings = similarity_model.encode(candidate_spans, convert_to_tensor=True)
    cos_sim_matrix = util.pytorch_cos_sim(input_embeddings, candidate_embeddings)

    weighted_sum = 0
    total_weight = 0
    for i, span in enumerate(input_spans):
        weight = get_pos_weight(span)
        if weight > 0:
            max_sim_for_span = torch.max(cos_sim_matrix[i]).item()
            weighted_sum += weight * max_sim_for_span
            total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def retrieve_demonstrations(text, source_pool, models, similarity_model, k):
    print("Stage 1: Retrieving Demonstrations...")
    input_pre_extracted_spans = set()
    for model, tokenizer in models.values():
        spans = extract_spans_zero_shot(text, model, tokenizer)
        input_pre_extracted_spans.update(spans)

    input_spans_list = list(input_pre_extracted_spans)

    scores = []
    print("  Calculating similarity scores against source pool...")
    for demo in tqdm(source_pool):
        candidate_spans = list(demo['entities'].keys())
        score = calculate_span_similarity(input_spans_list, candidate_spans, similarity_model)
        scores.append((score, demo))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [demo for score, demo in scores[:k]]


# --- Stage 2 Function ---

def extract_spans_few_shot(text, demos, models):
    print("Stage 2: Performing Few-Shot Span Extraction...")
    final_extracted_spans = set()
    demo_prompt = format_extraction_demos(demos)
    for name, (model, tokenizer) in models.items():
        prompt = FEW_SHOT_EXTRACTION_TEMPLATE.format(demonstrations=demo_prompt, text=text)
        response = _run_llm_inference(prompt, model, tokenizer, max_new_tokens=150)
        spans = _extract_spans_from_response(response)
        print(f"  {name} extracted: {spans}")
        final_extracted_spans.update(spans)
    return list(final_extracted_spans)


# --- Stage 3 Functions ---

def classify_spans(text, spans_to_classify, demos, model, tokenizer):
    demo_prompt = format_classification_demos(demos)
    spans_str = "[SEP]".join(spans_to_classify)
    prompt = FEW_SHOT_CLASSIFICATION_TEMPLATE.format(
        demonstrations=demo_prompt, text=text, spans_to_classify=spans_str
    )
    response = _run_llm_inference(prompt, model, tokenizer, max_new_tokens=250)

    try:
        classification_part = response.split("Classification:")[1].strip()
        pairs = classification_part.strip("[CLS]").split("[CLS]")
        results = {}
        for pair in pairs:
            parts = pair.split("[SEP]")
            if len(parts) == 2 and parts[0].strip() in spans_to_classify:
                results[parts[0].strip()] = parts[1].strip()
        return results
    except IndexError:
        return {}


def ensemble_classification_and_vote(text, spans, demos, models):
    print("Stage 3: Classifying Spans and Voting...")
    all_classifications = []
    for name, (model, tokenizer) in models.items():
        classifications = classify_spans(text, spans, demos, model, tokenizer)
        print(f"  {name} classified: {classifications}")
        all_classifications.append(classifications)

    final_results = {}
    for span in spans:
        votes = [res[span] for res in all_classifications if span in res]
        if votes:
            most_common_vote = Counter(votes).most_common(1)[0][0]
            final_results[span] = most_common_vote
    return final_results


# --- Stage 4 Function ---

def verify_and_filter(text, classified_entities, models, verifier_name):
    print("Stage 4: Verifying Entities...")
    verifier_model, verifier_tokenizer = models[verifier_name]
    final_verified_results = {}
    for span, entity_type in classified_entities.items():
        prompt = VERIFICATION_TEMPLATE.format(text=text, span=span, entity_type=entity_type)
        response = _run_llm_inference(prompt, verifier_model, verifier_tokenizer, max_new_tokens=5)
        answer = response.split("Answer:")[1].strip().lower()
        is_valid = "true" in answer
        print(f"  Verifying ('{span}', '{entity_type}') -> {is_valid}")
        if is_valid:
            final_verified_results[span] = entity_type
    return final_verified_results


# --- Main Pipeline Runner ---

def run_el4ner_pipeline(text, source_pool, models, similarity_model, k, verifier):
    # Stage 1
    demonstrations = retrieve_demonstrations(text, source_pool, models, similarity_model, k)
    print(f"\nRetrieved {len(demonstrations)} demonstrations.")

    # Stage 2
    extracted_spans = extract_spans_few_shot(text, demonstrations, models)
    print(f"\nFinal extracted spans after ensemble: {extracted_spans}")

    # Stage 3
    voted_results = ensemble_classification_and_vote(text, extracted_spans, demonstrations, models)
    print(f"\nResults after voting: {voted_results}")

    # Stage 4
    final_results = verify_and_filter(text, voted_results, models, verifier)

    return final_results