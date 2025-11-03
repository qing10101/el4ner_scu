"""
Quick Benchmark Script - Simplified version for rapid testing
Compares Hierarchical NER vs Single Models vs Large Model
"""

import time
import json
from tabulate import tabulate
from hierarchical_ner import HierarchicalNER
from transformers import pipeline


def quick_benchmark(texts: list, use_gpu: bool = True):
    """
    Quick benchmark comparing different approaches.
    
    Args:
        texts: List of input texts
        use_gpu: Whether to use GPU
    """
    
    print("="*70)
    print("QUICK BENCHMARK: Hierarchical NER vs Single Models vs Large Model")
    print("="*70)
    
    device = 0 if use_gpu else -1
    
    # Initialize models
    print("\nLoading models...")
    models = {}
    
    # Load single models
    single_models = {
        'BERT': 'dslim/bert-base-NER',
        'DistilBERT': 'dslim/distilbert-NER',
        'RoBERTa': 'Jean-Baptiste/roberta-large-ner-english'
    }
    
    for name, model_id in single_models.items():
        print(f"  Loading {name}...")
        try:
            models[name] = pipeline(
                "ner",
                model=model_id,
                tokenizer=model_id,
                aggregation_strategy="simple",
                device=device
            )
            print(f"  ✓ {name} loaded")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
    
    # Load large model
    print(f"  Loading XLM-RoBERTa-large...")
    try:
        models['Large'] = pipeline(
            "ner",
            model='xlm-roberta-large-finetuned-conll03-english',
            tokenizer='xlm-roberta-large-finetuned-conll03-english',
            aggregation_strategy="simple",
            device=device
        )
        print(f"  ✓ Large model loaded")
    except Exception as e:
        print(f"  ✗ Failed to load large model: {e}")
    
    # Load hierarchical NER
    print(f"  Loading Hierarchical NER...")
    hierarchical_ner = HierarchicalNER(uncertainty_threshold=0.3, use_gpu=use_gpu)
    print(f"  ✓ Hierarchical NER loaded")
    
    print("\n" + "="*70)
    print("RUNNING BENCHMARKS")
    print("="*70)
    
    # Collect results
    results = {
        'BERT': {'times': [], 'entities': []},
        'DistilBERT': {'times': [], 'entities': []},
        'RoBERTa': {'times': [], 'entities': []},
        'Large': {'times': [], 'entities': []},
        'Hierarchical': {'times': [], 'entities': [], 'large_used': 0}
    }
    
    # Benchmark each text
    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] Processing text {i}...")
        print(f"  Text: {text[:60]}..." if len(text) > 60 else f"  Text: {text}")
        
        # Test single models
        for model_name in ['BERT', 'DistilBERT', 'RoBERTa', 'Large']:
            if model_name in models:
                start = time.time()
                entities = models[model_name](text)
                elapsed = time.time() - start
                
                results[model_name]['times'].append(elapsed)
                results[model_name]['entities'].append(len(entities))
                
                print(f"    {model_name:12s}: {len(entities):2d} entities in {elapsed:.3f}s")
        
        # Test hierarchical
        start = time.time()
        _, uncertainty = hierarchical_ner._run_small_models(text)
        entities = hierarchical_ner.predict(text, verbose=False)
        elapsed = time.time() - start
        used_large = uncertainty > hierarchical_ner.uncertainty_threshold
        
        results['Hierarchical']['times'].append(elapsed)
        results['Hierarchical']['entities'].append(len(entities))
        if used_large:
            results['Hierarchical']['large_used'] += 1
        
        print(f"    {'Hierarchical':12s}: {len(entities):2d} entities in {elapsed:.3f}s" + 
              (f" (used large)" if used_large else ""))
    
    # Calculate statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    import numpy as np
    
    table_data = []
    for approach, data in results.items():
        if data['times']:
            avg_time = np.mean(data['times'])
            avg_entities = np.mean(data['entities'])
            total_time = np.sum(data['times'])
            
            row = {
                'Approach': approach,
                'Avg Time (s)': f"{avg_time:.3f}",
                'Total Time (s)': f"{total_time:.2f}",
                'Avg Entities': f"{avg_entities:.1f}",
                'Speed (text/s)': f"{1/avg_time:.2f}" if avg_time > 0 else "N/A"
            }
            
            if approach == 'Hierarchical':
                row['Large Used'] = f"{data['large_used']}/{len(texts)}"
            
            table_data.append(row)
    
    print("\n" + tabulate(table_data, headers='keys', tablefmt='grid'))
    
    # Comparison with hierarchical
    print("\n" + "="*70)
    print("COMPARISON WITH HIERARCHICAL")
    print("="*70)
    
    hier_time = np.mean(results['Hierarchical']['times'])
    
    comparison_data = []
    for approach, data in results.items():
        if approach != 'Hierarchical' and data['times']:
            other_time = np.mean(data['times'])
            speedup = hier_time / other_time if other_time > 0 else 0
            comparison_data.append({
                'vs Hierarchical': approach,
                'Speed Ratio': f"{speedup:.2f}x",
                'Faster/Slower': 'Faster' if speedup < 1 else 'Slower'
            })
    
    if comparison_data:
        print("\n" + tabulate(comparison_data, headers='keys', tablefmt='grid'))
    
    # Efficiency analysis
    if 'Large' in results and results['Large']['times']:
        large_time = np.mean(results['Large']['times'])
        total_large_time = large_time * len(texts)
        total_hier_time = hier_time * len(texts)
        savings = (total_large_time - total_hier_time) / total_large_time * 100
        
        print("\n" + "="*70)
        print("EFFICIENCY ANALYSIS")
        print("="*70)
        print(f"\nHierarchical vs Always-Large Model:")
        print(f"  Time per text: {hier_time:.3f}s vs {large_time:.3f}s")
        print(f"  Speed improvement: {large_time/hier_time:.2f}x faster" if hier_time > 0 else "N/A")
        print(f"  Time savings: {savings:.1f}%")
        print(f"  Large model used: {results['Hierarchical']['large_used']}/{len(texts)} times ({results['Hierarchical']['large_used']/len(texts)*100:.1f}%)")
    
    # Entity agreement
    print("\n" + "="*70)
    print("ENTITY COUNT COMPARISON")
    print("="*70)
    
    entity_comparison = []
    hier_avg_entities = np.mean(results['Hierarchical']['entities'])
    
    for approach, data in results.items():
        if approach != 'Hierarchical' and data['entities']:
            other_avg = np.mean(data['entities'])
            diff = hier_avg_entities - other_avg
            entity_comparison.append({
                'Approach': approach,
                'Avg Entities': f"{other_avg:.1f}",
                'Diff vs Hierarchical': f"{diff:+.1f}"
            })
    
    entity_comparison.append({
        'Approach': 'Hierarchical',
        'Avg Entities': f"{hier_avg_entities:.1f}",
        'Diff vs Hierarchical': "0.0"
    })
    
    print("\n" + tabulate(entity_comparison, headers='keys', tablefmt='grid'))
    
    return results


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick benchmark for NER systems")
    parser.add_argument('--texts-file', type=str, help='JSON file with texts')
    parser.add_argument('--num-texts', type=int, default=5, help='Number of texts to test')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    # Get test texts
    if args.texts_file:
        with open(args.texts_file, 'r', encoding='utf-8') as f:
            texts = json.load(f)
            if isinstance(texts, dict) and 'texts' in texts:
                texts = texts['texts']
    else:
        # Default test texts
        texts = [
            "Apple Inc. announced new products in California.",
            "Microsoft CEO Satya Nadella spoke in Seattle.",
            "Jordan scored while visiting Jordan River.",
        ]
    
    texts = texts[:args.num_texts]
    
    # Run benchmark
    results = quick_benchmark(texts, use_gpu=not args.cpu)
    
    print("\n" + "✅"*35)
    print("BENCHMARK COMPLETE")
    print("✅"*35)


if __name__ == "__main__":
    main()

