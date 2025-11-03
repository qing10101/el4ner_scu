"""
Integration Example: Combining Hierarchical NER with EL4NER Pipeline

This shows how to use the new hierarchical NER system alongside
the existing EL4NER ensemble learning approach.
"""

import json
from hierarchical_ner import HierarchicalNER
from hierarchical_ner_config import ModelPerformanceTracker
from tabulate import tabulate


def hierarchical_ner_standalone():
    """
    Example 1: Using Hierarchical NER as a standalone system.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Hierarchical NER - Standalone Mode")
    print("=" * 80)
    
    # Initialize
    ner_system = HierarchicalNER(
        uncertainty_threshold=0.3,
        use_gpu=True
    )
    
    # Test texts
    texts = [
        "Apple Inc. CEO Tim Cook announced new products in Cupertino.",
        "Jordan scored the winning goal while visiting Jordan.",
        "Microsoft's headquarters in Washington overlooks the lake."
    ]
    
    results = []
    for i, text in enumerate(texts, 1):
        print(f"\n--- Text {i} ---")
        print(f"Input: {text}")
        
        entities = ner_system.predict(text, verbose=False)
        
        print(f"Found {len(entities)} entities:")
        for entity in entities:
            print(f"  • {entity['word']} ({entity['entity_group']}) - {entity['total_votes']} votes")
        
        results.append({
            'text': text,
            'entities': entities
        })
    
    return results


def hierarchical_ner_with_filtering():
    """
    Example 2: Using Hierarchical NER with confidence filtering.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Hierarchical NER - With Confidence Filtering")
    print("=" * 80)
    
    ner_system = HierarchicalNER(uncertainty_threshold=0.3)
    
    text = "The Amazon rainforest is threatened by Amazon's expansion plans."
    
    print(f"\nInput: {text}")
    print("\nThis text has entity ambiguity:")
    print("  - First 'Amazon' should be LOCATION (rainforest)")
    print("  - Second 'Amazon' should be ORGANIZATION (company)")
    
    entities = ner_system.predict(text, verbose=True)
    
    # Filter by confidence threshold
    high_confidence = [e for e in entities if e['score'] > 0.85]
    medium_confidence = [e for e in entities if 0.7 <= e['score'] <= 0.85]
    low_confidence = [e for e in entities if e['score'] < 0.7]
    
    print("\n" + "=" * 80)
    print("CONFIDENCE-BASED FILTERING")
    print("=" * 80)
    
    print(f"\nHigh Confidence (>0.85): {len(high_confidence)} entities")
    for e in high_confidence:
        print(f"  ✓ {e['word']} ({e['entity_group']}) - Score: {e['score']:.3f}")
    
    print(f"\nMedium Confidence (0.7-0.85): {len(medium_confidence)} entities")
    for e in medium_confidence:
        print(f"  ⚠ {e['word']} ({e['entity_group']}) - Score: {e['score']:.3f}")
    
    print(f"\nLow Confidence (<0.7): {len(low_confidence)} entities")
    for e in low_confidence:
        print(f"  ✗ {e['word']} ({e['entity_group']}) - Score: {e['score']:.3f}")


def hierarchical_ner_batch_analysis():
    """
    Example 3: Batch processing with performance analysis.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Hierarchical NER - Batch Analysis")
    print("=" * 80)
    
    # Load sample data
    test_cases = [
        ("Google's CEO Sundar Pichai spoke at the conference.", "Easy"),
        ("President Biden met with Prime Minister Trudeau in Ottawa.", "Easy"),
        ("Paris Hilton attended the Paris Fashion Week in Paris, France.", "Hard"),
        ("Washington warned against foreign interference from Washington.", "Hard"),
        ("The Nile River flows through Egypt and Sudan in Africa.", "Medium"),
        ("Tesla's factory in Texas produces electric vehicles.", "Easy"),
        ("China's economy impacts global markets in Asia and beyond.", "Medium"),
        ("Jordan Peterson discussed philosophy with Michael Jordan.", "Hard"),
    ]
    
    ner_system = HierarchicalNER(uncertainty_threshold=0.3)
    tracker = ModelPerformanceTracker()
    
    results_table = []
    
    for text, difficulty in test_cases:
        # Get predictions
        small_preds, uncertainty = ner_system._run_small_models(text)
        entities = ner_system.predict(text, verbose=False)
        
        used_large = uncertainty > ner_system.uncertainty_threshold
        
        # Track performance
        tracker.record_prediction(
            uncertainty=uncertainty,
            used_large_model=used_large,
            num_entities=len(entities)
        )
        
        results_table.append({
            'Text': text[:50] + "..." if len(text) > 50 else text,
            'Difficulty': difficulty,
            'Uncertainty': f"{uncertainty:.3f}",
            'Large Model': '✓' if used_large else '✗',
            'Entities': len(entities)
        })
    
    # Print results table
    print("\n" + tabulate(results_table, headers='keys', tablefmt='grid'))
    
    # Print performance summary
    tracker.print_summary()


def compare_with_single_model():
    """
    Example 4: Compare hierarchical approach vs single model.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Comparison - Hierarchical vs Single Model")
    print("=" * 80)
    
    from transformers import pipeline
    
    text = "Microsoft CEO Satya Nadella announced Azure updates in Seattle."
    
    print(f"\nTest Text: {text}\n")
    
    # Method 1: Single small model (BERT)
    print("Method 1: Single Small Model (BERT-base)")
    print("-" * 80)
    try:
        single_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        entities_single = single_model(text)
        print(f"Found {len(entities_single)} entities:")
        for e in entities_single:
            print(f"  • {e['word']} ({e['entity_group']}) - Confidence: {e['score']:.3f}")
    except Exception as ex:
        print(f"Error: {ex}")
    
    # Method 2: Hierarchical NER
    print("\n\nMethod 2: Hierarchical NER (3 small + 1 large with voting)")
    print("-" * 80)
    ner_system = HierarchicalNER(uncertainty_threshold=0.3)
    entities_hierarchical = ner_system.predict(text, verbose=False)
    print(f"Found {len(entities_hierarchical)} entities:")
    for e in entities_hierarchical:
        print(f"  • {e['word']} ({e['entity_group']}) - Confidence: {e['score']:.3f}, Votes: {e['total_votes']}")
    
    # Comparison
    print("\n\nComparison Summary:")
    print("-" * 80)
    print(f"Single Model: {len(entities_single)} entities")
    print(f"Hierarchical: {len(entities_hierarchical)} entities")
    print(f"\nHierarchical Advantages:")
    print("  ✓ Multiple models reduce individual model bias")
    print("  ✓ Weighted voting provides more robust predictions")
    print("  ✓ Uncertainty-based escalation handles difficult cases")
    print("  ✓ Vote counts provide interpretability")


def export_results_to_json():
    """
    Example 5: Export results to JSON for further processing.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Export Results to JSON")
    print("=" * 80)
    
    ner_system = HierarchicalNER(uncertainty_threshold=0.3)
    
    texts = [
        "Apple Inc. announced new products in California.",
        "Amazon is opening warehouses in New York and Texas."
    ]
    
    all_results = []
    
    for text in texts:
        entities = ner_system.predict(text, verbose=False)
        
        result = {
            'text': text,
            'entity_count': len(entities),
            'entities': [
                {
                    'word': e['word'],
                    'type': e['entity_group'],
                    'position': {'start': e['start'], 'end': e['end']},
                    'confidence': e['score'],
                    'votes': e['total_votes'],
                    'vote_breakdown': e['vote_details']
                }
                for e in entities
            ]
        }
        
        all_results.append(result)
    
    # Save to JSON
    output_file = "hierarchical_ner_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results exported to: {output_file}")
    print(f"  Total texts processed: {len(all_results)}")
    print(f"  Total entities found: {sum(r['entity_count'] for r in all_results)}")
    
    # Preview
    print("\nPreview of JSON output:")
    print(json.dumps(all_results[0], indent=2)[:500] + "...")


def main():
    """Run all integration examples."""
    
    print("\n" + "🚀" * 40)
    print("HIERARCHICAL NER - INTEGRATION EXAMPLES")
    print("🚀" * 40)
    
    examples = [
        ("Standalone Usage", hierarchical_ner_standalone),
        ("Confidence Filtering", hierarchical_ner_with_filtering),
        ("Batch Analysis", hierarchical_ner_batch_analysis),
        ("Comparison Study", compare_with_single_model),
        ("JSON Export", export_results_to_json)
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n⚠ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "✅" * 40)
    print("ALL EXAMPLES COMPLETED")
    print("✅" * 40)
    
    print("""
    
    Next Steps:
    -----------
    1. Review the generated 'hierarchical_ner_results.json' file
    2. Adjust uncertainty_threshold based on your needs
    3. Try different model presets (see hierarchical_ner_config.py)
    4. Integrate with your existing pipeline
    
    For more information, see HIERARCHICAL_NER_README.md
    """)


if __name__ == "__main__":
    main()

