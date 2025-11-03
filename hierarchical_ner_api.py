"""
Simple API-style wrapper for Hierarchical NER System
Provides easy-to-use functions for common use cases
"""

from hierarchical_ner import HierarchicalNER
from hierarchical_ner_config import (
    HierarchicalNERConfig,
    get_fast_config,
    get_accurate_config,
    get_multilingual_config
)
from typing import List, Dict, Optional, Union
import json


# Global instance (lazy loaded)
_ner_instance = None


def get_ner_instance(
    preset: str = 'balanced',
    uncertainty_threshold: float = 0.3,
    force_reload: bool = False
) -> HierarchicalNER:
    """
    Get or create a global NER instance (singleton pattern).
    
    Args:
        preset: 'fast', 'balanced', or 'accurate'
        uncertainty_threshold: Threshold for large model (0-1)
        force_reload: Force recreation of instance
        
    Returns:
        HierarchicalNER instance
    """
    global _ner_instance
    
    if _ner_instance is None or force_reload:
        # Map presets to thresholds
        preset_thresholds = {
            'fast': 0.5,
            'balanced': 0.3,
            'accurate': 0.2
        }
        
        threshold = preset_thresholds.get(preset, uncertainty_threshold)
        
        _ner_instance = HierarchicalNER(
            uncertainty_threshold=threshold,
            use_gpu=True
        )
    
    return _ner_instance


def extract_entities(
    text: Union[str, List[str]],
    preset: str = 'balanced',
    return_format: str = 'detailed'
) -> Union[List[Dict], Dict[str, List[Dict]]]:
    """
    Extract named entities from text(s).
    
    Args:
        text: Single text string or list of texts
        preset: 'fast', 'balanced', or 'accurate'
        return_format: 'detailed', 'simple', or 'grouped'
        
    Returns:
        Extracted entities in specified format
    """
    ner = get_ner_instance(preset=preset)
    
    # Handle single text
    if isinstance(text, str):
        entities = ner.predict(text, verbose=False)
        return _format_entities(entities, return_format)
    
    # Handle multiple texts
    results = {}
    for i, t in enumerate(text):
        entities = ner.predict(t, verbose=False)
        results[f"text_{i}"] = _format_entities(entities, return_format)
    
    return results


def _format_entities(entities: List[Dict], format_type: str) -> Union[List[Dict], Dict]:
    """Format entities according to specified type."""
    
    if format_type == 'simple':
        # Return just entity text and type
        return [
            {'entity': e['word'], 'type': e['entity_group']}
            for e in entities
        ]
    
    elif format_type == 'grouped':
        # Group entities by type
        grouped = {}
        for e in entities:
            entity_type = e['entity_group']
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append(e['word'])
        return grouped
    
    else:  # detailed
        # Return full information
        return [
            {
                'entity': e['word'],
                'type': e['entity_group'],
                'position': {'start': e['start'], 'end': e['end']},
                'confidence': round(e['score'], 3),
                'votes': e['total_votes'],
                'vote_breakdown': e['vote_details']
            }
            for e in entities
        ]


def extract_entities_by_type(
    text: str,
    entity_types: Optional[List[str]] = None,
    preset: str = 'balanced'
) -> Dict[str, List[str]]:
    """
    Extract entities grouped by type, optionally filtered.
    
    Args:
        text: Input text
        entity_types: List of types to extract (None = all)
        preset: NER preset to use
        
    Returns:
        Dict mapping entity types to lists of entities
    """
    ner = get_ner_instance(preset=preset)
    entities = ner.predict(text, verbose=False)
    
    # Group by type
    grouped = {}
    for e in entities:
        entity_type = e['entity_group']
        
        # Filter by requested types if specified
        if entity_types and entity_type not in entity_types:
            continue
        
        if entity_type not in grouped:
            grouped[entity_type] = []
        grouped[entity_type].append(e['word'])
    
    return grouped


def get_high_confidence_entities(
    text: str,
    confidence_threshold: float = 0.8,
    preset: str = 'balanced'
) -> List[Dict]:
    """
    Extract only high-confidence entities.
    
    Args:
        text: Input text
        confidence_threshold: Minimum confidence (0-1)
        preset: NER preset to use
        
    Returns:
        List of high-confidence entities
    """
    ner = get_ner_instance(preset=preset)
    entities = ner.predict(text, verbose=False)
    
    # Filter by confidence
    high_confidence = [
        {
            'entity': e['word'],
            'type': e['entity_group'],
            'confidence': round(e['score'], 3)
        }
        for e in entities
        if e['score'] >= confidence_threshold
    ]
    
    return high_confidence


def extract_persons(text: str, preset: str = 'balanced') -> List[str]:
    """Extract person names from text."""
    result = extract_entities_by_type(text, entity_types=['PER', 'PERSON'], preset=preset)
    return result.get('PER', []) + result.get('PERSON', [])


def extract_locations(text: str, preset: str = 'balanced') -> List[str]:
    """Extract location names from text."""
    result = extract_entities_by_type(text, entity_types=['LOC', 'LOCATION'], preset=preset)
    return result.get('LOC', []) + result.get('LOCATION', [])


def extract_organizations(text: str, preset: str = 'balanced') -> List[str]:
    """Extract organization names from text."""
    result = extract_entities_by_type(text, entity_types=['ORG', 'ORGANIZATION'], preset=preset)
    return result.get('ORG', []) + result.get('ORGANIZATION', [])


def batch_extract(
    texts: List[str],
    preset: str = 'balanced',
    show_progress: bool = True
) -> List[Dict]:
    """
    Process multiple texts and return results.
    
    Args:
        texts: List of input texts
        preset: NER preset to use
        show_progress: Show progress bar
        
    Returns:
        List of results for each text
    """
    from tqdm import tqdm
    
    ner = get_ner_instance(preset=preset)
    results = []
    
    iterator = tqdm(texts, desc="Processing") if show_progress else texts
    
    for text in iterator:
        entities = ner.predict(text, verbose=False)
        results.append({
            'text': text,
            'entities': _format_entities(entities, 'detailed'),
            'entity_count': len(entities)
        })
    
    return results


def save_results(
    results: Union[List[Dict], Dict],
    output_file: str,
    format: str = 'json'
):
    """
    Save extraction results to file.
    
    Args:
        results: Results to save
        output_file: Output file path
        format: 'json' or 'jsonl'
    """
    if format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    elif format == 'jsonl':
        with open(output_file, 'w', encoding='utf-8') as f:
            if isinstance(results, list):
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps(results, ensure_ascii=False) + '\n')
    
    print(f"✓ Results saved to: {output_file}")


def analyze_uncertainty(text: str) -> Dict:
    """
    Analyze uncertainty for a given text.
    
    Args:
        text: Input text
        
    Returns:
        Dict with uncertainty analysis
    """
    ner = get_ner_instance()
    
    # Get small model predictions
    small_preds, uncertainty = ner._run_small_models(text)
    
    # Determine if large model would be triggered
    would_trigger = uncertainty > ner.uncertainty_threshold
    
    return {
        'text': text,
        'uncertainty_score': round(uncertainty, 3),
        'threshold': ner.uncertainty_threshold,
        'would_trigger_large_model': would_trigger,
        'recommendation': 'Use large model' if would_trigger else 'Small models sufficient',
        'small_model_predictions': {
            name: len(preds)
            for name, preds in small_preds.items()
        }
    }


# ============================================================================
# Example usage functions
# ============================================================================

def example_simple_usage():
    """Example 1: Simple entity extraction."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Usage")
    print("="*70)
    
    text = "Apple Inc. announced new products in California."
    
    # Extract entities
    entities = extract_entities(text, return_format='simple')
    
    print(f"\nText: {text}")
    print(f"\nEntities found:")
    for e in entities:
        print(f"  • {e['entity']} ({e['type']})")


def example_grouped_extraction():
    """Example 2: Grouped by entity type."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Grouped Extraction")
    print("="*70)
    
    text = "Microsoft CEO Satya Nadella spoke in Seattle about Azure and Windows."
    
    # Extract and group
    entities = extract_entities(text, return_format='grouped')
    
    print(f"\nText: {text}")
    print(f"\nEntities by type:")
    for entity_type, entity_list in entities.items():
        print(f"\n{entity_type}:")
        for e in entity_list:
            print(f"  • {e}")


def example_type_specific():
    """Example 3: Extract specific entity types."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Type-Specific Extraction")
    print("="*70)
    
    text = "President Biden met with CEO Tim Cook in Washington to discuss Apple's plans."
    
    print(f"\nText: {text}")
    
    # Extract different types
    persons = extract_persons(text)
    locations = extract_locations(text)
    orgs = extract_organizations(text)
    
    print(f"\nPersons: {persons}")
    print(f"Locations: {locations}")
    print(f"Organizations: {orgs}")


def example_confidence_filtering():
    """Example 4: High-confidence entities only."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Confidence Filtering")
    print("="*70)
    
    text = "Jordan scored 23 points while visiting the Jordan River."
    
    # Get high-confidence entities
    high_conf = get_high_confidence_entities(text, confidence_threshold=0.85)
    
    print(f"\nText: {text}")
    print(f"\nHigh-confidence entities (>0.85):")
    for e in high_conf:
        print(f"  • {e['entity']} ({e['type']}) - {e['confidence']}")


def example_batch_processing():
    """Example 5: Process multiple texts."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Batch Processing")
    print("="*70)
    
    texts = [
        "Google announced AI updates in California.",
        "Amazon is opening new warehouses in Texas.",
        "Tesla's factory in Berlin produces electric vehicles."
    ]
    
    # Batch process
    results = batch_extract(texts, show_progress=True)
    
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text']}")
        print(f"   Entities: {result['entity_count']}")
        for e in result['entities']:
            print(f"     • {e['entity']} ({e['type']})")


def example_uncertainty_analysis():
    """Example 6: Analyze uncertainty."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Uncertainty Analysis")
    print("="*70)
    
    texts = [
        "Apple Inc. is in California.",  # Easy
        "Jordan visited Jordan.",  # Ambiguous
    ]
    
    for text in texts:
        analysis = analyze_uncertainty(text)
        
        print(f"\nText: {text}")
        print(f"Uncertainty: {analysis['uncertainty_score']}")
        print(f"Recommendation: {analysis['recommendation']}")
        print(f"Small models found: {analysis['small_model_predictions']}")


def main():
    """Run all API examples."""
    
    print("\n" + "🚀"*35)
    print("HIERARCHICAL NER - API EXAMPLES")
    print("🚀"*35)
    
    examples = [
        example_simple_usage,
        example_grouped_extraction,
        example_type_specific,
        example_confidence_filtering,
        example_batch_processing,
        example_uncertainty_analysis
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "✅"*35)
    print("ALL API EXAMPLES COMPLETED")
    print("✅"*35)


if __name__ == "__main__":
    main()

