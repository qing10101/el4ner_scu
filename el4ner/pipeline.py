# el4ner/pipeline.py (Final, Simplified Version)

import torch
import re
import nltk
from sentence_transformers import util
from tqdm import tqdm
from collections import Counter

# Import prompt templates from the local package
from .prompts import (
    ZERO_SHOT_EXTRACTION_TEMPLATE, FEW_SHOT_EXTRACTION_TEMPLATE,
    FEW_SHOT_CLASSIFICATION_TEMPLATE, VERIFICATION_TEMPLATE,
    format_extraction_demos, format_classification_demos
)

# --- REVERTED TO THE SIMPLE, STANDARD INFERENCE FUNCTION ---
def _run_llm_inference(prompt, model, tokenizer, max_new_tokens=100):
    """
    A standard, simple helper function to run inference on any model.
    No special case logic is needed anymore.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
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

# --- ALL `is_glm_model` PARAMETERS HAVE BEEN REMOVED FROM THE FUNCTIONS BELOW ---

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
    if tag in ['NNP', 'NNPS']: return 4
    elif tag in ['NN', 'NNS']: return 2
    elif tag in ['PRP', 'PRP$']: return 1
    else: return 0

def calculate_span_similarity(input_spans, candidate_spans, similarity_model):
    if not input_spans or not candidate_spans: return 0.0
    input_embeddings = similarity_model.encode(input_spans, convert_to_tensor=True, show_progress_bar=False)
    candidate_embeddings = similarity_model.encode(candidate_spans, convert_to_tensor=True, show_progress_bar=False)
    cos_sim_matrix = util.pytorch_cos_sim(input_embeddings, candidate_embeddings)
    weighted_sum, total_weight = 0, 0
    for i, span in enumerate(input_spans):
        weight = get_pos_weight(span)
        if weight > 0:
            max_sim_for_span = torch.max(cos_sim_matrix[i]).item()
            weighted_sum += weight * max_sim_for_span
            total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else 0.0

def retrieve_demonstrations(text, source_pool, models, similarity_model, k):
    input_pre_extracted_spans = set()
    for name, (model, tokenizer) in models.items():
        spans = extract_spans_zero_shot(text, model, tokenizer)
        input_pre_extracted_spans.update(spans)
    input_spans_list = list(input_pre_extracted_spans)
    scores = []
    for demo in tqdm(source_pool, desc="  Finding Demos", leave=False):
        candidate_spans = list(demo['entities'].keys())
        score = calculate_span_similarity(input_spans_list, candidate_spans, similarity_model)
        scores.append((score, demo))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [demo for score, demo in scores[:k]]

def extract_spans_few_shot(text, demos, models):
    final_extracted_spans = set()
    demo_prompt = format_extraction_demos(demos)
    for name, (model, tokenizer) in models.items():
        prompt = FEW_SHOT_EXTRACTION_TEMPLATE.format(demonstrations=demo_prompt, text=text)
        response = _run_llm_inference(prompt, model, tokenizer, max_new_tokens=150)
        spans = _extract_spans_from_response(response)
        final_extracted_spans.update(spans)
    return list(final_extracted_spans)

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
    all_classifications = []
    for name, (model, tokenizer) in models.items():
        classifications = classify_spans(text, spans, demos, model, tokenizer)
        all_classifications.append(classifications)
    final_results = {}
    for span in spans:
        votes = [res[span] for res in all_classifications if span in res]
        if votes:
            most_common_vote = Counter(votes).most_common(1)[0][0]
            final_results[span] = most_common_vote
    return final_results

def verify_and_filter(text, classified_entities, models, verifier_name):
    verifier_model, verifier_tokenizer = models[verifier_name]
    final_verified_results = {}
    for span, entity_type in classified_entities.items():
        prompt = VERIFICATION_TEMPLATE.format(text=text, span=span, entity_type=entity_type)
        response = _run_llm_inference(prompt, verifier_model, verifier_tokenizer, max_new_tokens=5)
        try:
            answer = response.split("Answer:")[1].strip().lower()
            if "true" in answer:
                final_verified_results[span] = entity_type
        except IndexError:
            continue
    return final_verified_results

def run_el4ner_pipeline(text, source_pool, models, similarity_model, k, verifier):
    demonstrations = retrieve_demonstrations(text, source_pool, models, similarity_model, k)
    extracted_spans = extract_spans_few_shot(text, demonstrations, models)
    if not extracted_spans: return {}
    voted_results = ensemble_classification_and_vote(text, extracted_spans, demonstrations, models)
    if not voted_results: return {}
    final_results = verify_and_filter(text, voted_results, models, verifier)
    return final_results