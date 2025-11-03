"""
Comprehensive Benchmark Script for Hierarchical NER System
Compares: Hierarchical NER vs 3 Single Models vs 1 Large Model

Measures:
- Execution time
- Memory usage
- Number of entities found
- Entity agreement/consensus
- Large model usage rate
"""

import time
import json
import sys
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from tabulate import tabulate
import numpy as np
import torch
import gc

# Import our hierarchical NER system
try:
    from hierarchical_ner import HierarchicalNER
    from hierarchical_ner_config import ModelPerformanceTracker
except ImportError:
    print("Error: hierarchical_ner module not found. Please ensure it's in the same directory.")
    sys.exit(1)

# Try to import memory profiling tools
try:
    import psutil
    import os
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Memory profiling will be limited.")
    print("Install with: pip install psutil")


class BenchmarkNER:
    """Benchmark different NER approaches."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Model configurations
        self.small_models_config = {
            'bert': 'dslim/bert-base-NER',
            'distilbert': 'dslim/distilbert-NER',
            'roberta': 'Jean-Baptiste/roberta-large-ner-english'
        }
        
        self.large_model_config = 'xlm-roberta-large-finetuned-conll03-english'
        
        # Initialize models (lazy loading)
        self.models = {}
        self.hierarchical_ner = None
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        return 0.0
    
    def _load_single_model(self, model_name: str, model_id: str):
        """Load a single NER model."""
        from transformers import pipeline
        
        print(f"  Loading {model_name}...")
        try:
            model = pipeline(
                "ner",
                model=model_id,
                tokenizer=model_id,
                aggregation_strategy="simple",
                device=0 if self.use_gpu else -1
            )
            return model
        except Exception as e:
            print(f"  ✗ Failed to load {model_name}: {e}")
            return None
    
    def load_all_models(self):
        """Load all models for comparison."""
        print("\n" + "="*70)
        print("LOADING MODELS FOR BENCHMARKING")
        print("="*70)
        
        initial_memory = self._get_memory_usage()
        
        # Load single models
        print("\nLoading single models...")
        for name, model_id in self.small_models_config.items():
            model = self._load_single_model(name, model_id)
            if model:
                self.models[name] = model
        
        # Load large model
        print(f"\nLoading large model...")
        large_model = self._load_single_model("xlm-roberta-large", self.large_model_config)
        if large_model:
            self.models['large'] = large_model
        
        # Load hierarchical NER
        print(f"\nLoading hierarchical NER system...")
        self.hierarchical_ner = HierarchicalNER(
            uncertainty_threshold=0.3,
            use_gpu=self.use_gpu
        )
        
        final_memory = self._get_memory_usage()
        memory_used = final_memory - initial_memory
        
        print(f"\n✓ All models loaded")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Total models: {len(self.models) + 1}")  # +1 for hierarchical
        
    def predict_single_model(self, text: str, model_name: str) -> Tuple[List[Dict], float]:
        """Run prediction with a single model."""
        if model_name not in self.models:
            return [], 0.0
        
        model = self.models[model_name]
        
        start_time = time.time()
        try:
            entities = model(text)
            elapsed = time.time() - start_time
            
            # Normalize entity format
            normalized = []
            for e in entities:
                normalized.append({
                    'word': e['word'],
                    'entity_group': e['entity_group'],
                    'start': e['start'],
                    'end': e['end'],
                    'score': e['score']
                })
            
            return normalized, elapsed
        except Exception as e:
            print(f"  Error in {model_name}: {e}")
            return [], time.time() - start_time
    
    def predict_hierarchical(self, text: str) -> Tuple[List[Dict], float, bool]:
        """Run prediction with hierarchical NER."""
        if not self.hierarchical_ner:
            return [], 0.0, False
        
        start_time = time.time()
        
        # Get small model predictions to check uncertainty
        small_preds, uncertainty = self.hierarchical_ner._run_small_models(text)
        used_large = uncertainty > self.hierarchical_ner.uncertainty_threshold
        
        # Get final predictions
        entities = self.hierarchical_ner.predict(text, verbose=False)
        
        elapsed = time.time() - start_time
        
        return entities, elapsed, used_large
    
    def benchmark_single_text(self, text: str, text_id: int = 0) -> Dict:
        """Benchmark a single text across all approaches."""
        results = {
            'text_id': text_id,
            'text_length': len(text),
            'text_word_count': len(text.split()),
            'approaches': {}
        }
        
        # Benchmark each single model
        for model_name in ['bert', 'distilbert', 'roberta']:
            if model_name in self.models:
                entities, elapsed = self.predict_single_model(text, model_name)
                results['approaches'][model_name] = {
                    'entities': entities,
                    'entity_count': len(entities),
                    'time': elapsed,
                    'memory': self._get_memory_usage()
                }
        
        # Benchmark large model
        if 'large' in self.models:
            entities, elapsed = self.predict_single_model(text, 'large')
            results['approaches']['large'] = {
                'entities': entities,
                'entity_count': len(entities),
                'time': elapsed,
                'memory': self._get_memory_usage()
            }
        
        # Benchmark hierarchical
        entities, elapsed, used_large = self.predict_hierarchical(text)
        results['approaches']['hierarchical'] = {
            'entities': entities,
            'entity_count': len(entities),
            'time': elapsed,
            'used_large_model': used_large,
            'memory': self._get_memory_usage()
        }
        
        return results
    
    def calculate_entity_overlap(self, entities1: List[Dict], entities2: List[Dict]) -> Dict:
        """Calculate overlap between two entity lists."""
        # Normalize entities for comparison (by position)
        def normalize(e):
            return (e['start'], e['end'], e['entity_group'].lower())
        
        set1 = {normalize(e) for e in entities1}
        set2 = {normalize(e) for e in entities2}
        
        intersection = set1 & set2
        union = set1 | set2
        
        jaccard = len(intersection) / len(union) if union else 0.0
        precision = len(intersection) / len(set1) if set1 else 0.0
        recall = len(intersection) / len(set2) if set2 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'intersection': len(intersection),
            'set1_only': len(set1 - set2),
            'set2_only': len(set2 - set1),
            'jaccard': jaccard,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def analyze_results(self, all_results: List[Dict]) -> Dict:
        """Analyze all benchmark results."""
        analysis = {
            'total_texts': len(all_results),
            'approaches': {},
            'comparisons': {}
        }
        
        approach_names = ['bert', 'distilbert', 'roberta', 'large', 'hierarchical']
        
        # Collect statistics per approach
        for approach in approach_names:
            times = []
            entity_counts = []
            memories = []
            
            for result in all_results:
                if approach in result['approaches']:
                    app_data = result['approaches'][approach]
                    times.append(app_data['time'])
                    entity_counts.append(app_data['entity_count'])
                    if 'memory' in app_data:
                        memories.append(app_data['memory'])
            
            if times:
                analysis['approaches'][approach] = {
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times),
                    'avg_entities': np.mean(entity_counts),
                    'std_entities': np.std(entity_counts),
                    'avg_memory': np.mean(memories) if memories else 0.0
                }
                
                # Special stats for hierarchical
                if approach == 'hierarchical':
                    large_usage_count = sum(
                        1 for r in all_results 
                        if approach in r['approaches'] and r['approaches'][approach].get('used_large_model', False)
                    )
                    analysis['approaches'][approach]['large_model_usage_rate'] = large_usage_count / len(times)
                    analysis['approaches'][approach]['large_model_triggered'] = large_usage_count
        
        # Compare hierarchical vs others
        if 'hierarchical' in analysis['approaches']:
            hierarchical_entities = [
                r['approaches']['hierarchical']['entities']
                for r in all_results
                if 'hierarchical' in r['approaches']
            ]
            
            for compare_to in ['bert', 'distilbert', 'roberta', 'large']:
                if compare_to in analysis['approaches']:
                    compare_entities = [
                        r['approaches'][compare_to]['entities']
                        for r in all_results
                        if compare_to in r['approaches']
                    ]
                    
                    # Calculate average overlap metrics
                    overlaps = []
                    for h_ents, c_ents in zip(hierarchical_entities, compare_entities):
                        overlap = self.calculate_entity_overlap(h_ents, c_ents)
                        overlaps.append(overlap)
                    
                    if overlaps:
                        analysis['comparisons'][f'hierarchical_vs_{compare_to}'] = {
                            'avg_jaccard': np.mean([o['jaccard'] for o in overlaps]),
                            'avg_precision': np.mean([o['precision'] for o in overlaps]),
                            'avg_recall': np.mean([o['recall'] for o in overlaps]),
                            'avg_f1': np.mean([o['f1'] for o in overlaps])
                        }
        
        return analysis
    
    def print_summary(self, analysis: Dict):
        """Print formatted summary of benchmark results."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Performance table
        print("\n" + "-"*80)
        print("PERFORMANCE METRICS")
        print("-"*80)
        
        table_data = []
        for approach, stats in analysis['approaches'].items():
            table_data.append({
                'Approach': approach.upper(),
                'Avg Time (s)': f"{stats['avg_time']:.3f}",
                'Total Time (s)': f"{stats['total_time']:.2f}",
                'Avg Entities': f"{stats['avg_entities']:.1f}",
                'Speed (texts/s)': f"{1/stats['avg_time']:.2f}" if stats['avg_time'] > 0 else "N/A"
            })
            
            if approach == 'hierarchical':
                table_data[-1]['Large Model Usage'] = f"{stats.get('large_model_usage_rate', 0):.1%}"
        
        print(tabulate(table_data, headers='keys', tablefmt='grid'))
        
        # Speed comparison
        print("\n" + "-"*80)
        print("SPEED COMPARISON (vs Hierarchical)")
        print("-"*80)
        
        if 'hierarchical' in analysis['approaches']:
            hier_time = analysis['approaches']['hierarchical']['avg_time']
            
            speed_table = []
            for approach, stats in analysis['approaches'].items():
                if approach != 'hierarchical':
                    speedup = hier_time / stats['avg_time'] if stats['avg_time'] > 0 else 0
                    speed_table.append({
                        'Approach': approach.upper(),
                        'Speed vs Hierarchical': f"{speedup:.2f}x",
                        'Faster' if speedup > 1 else 'Slower': '✓' if abs(speedup - 1) > 0.1 else ''
                    })
            
            print(tabulate(speed_table, headers='keys', tablefmt='grid'))
        
        # Entity agreement comparison
        if analysis.get('comparisons'):
            print("\n" + "-"*80)
            print("ENTITY AGREEMENT (Hierarchical vs Others)")
            print("-"*80)
            
            agreement_table = []
            for comparison, metrics in analysis['comparisons'].items():
                other = comparison.split('_vs_')[1]
                agreement_table.append({
                    'Compared To': other.upper(),
                    'Jaccard Similarity': f"{metrics['avg_jaccard']:.3f}",
                    'Precision': f"{metrics['avg_precision']:.3f}",
                    'Recall': f"{metrics['avg_recall']:.3f}",
                    'F1 Score': f"{metrics['avg_f1']:.3f}"
                })
            
            print(tabulate(agreement_table, headers='keys', tablefmt='grid'))
        
        # Hierarchical specifics
        if 'hierarchical' in analysis['approaches']:
            hier_stats = analysis['approaches']['hierarchical']
            print("\n" + "-"*80)
            print("HIERARCHICAL NER SPECIFICS")
            print("-"*80)
            print(f"Large model triggered: {hier_stats.get('large_model_triggered', 0)} / {analysis['total_texts']} texts")
            print(f"Large model usage rate: {hier_stats.get('large_model_usage_rate', 0):.1%}")
            print(f"Average entities found: {hier_stats['avg_entities']:.1f}")
            print(f"Average processing time: {hier_stats['avg_time']:.3f}s")
            
            # Efficiency metrics
            total_large_time = analysis['approaches'].get('large', {}).get('avg_time', 0) * analysis['total_texts']
            hier_time = hier_stats['avg_time'] * analysis['total_texts']
            if total_large_time > 0:
                efficiency = (total_large_time - hier_time) / total_large_time * 100
                print(f"Time savings vs always-large: {efficiency:.1f}%")
        
        # Memory usage
        print("\n" + "-"*80)
        print("MEMORY USAGE")
        print("-"*80)
        
        memory_table = []
        for approach, stats in analysis['approaches'].items():
            if stats.get('avg_memory', 0) > 0:
                memory_table.append({
                    'Approach': approach.upper(),
                    'Avg Memory (MB)': f"{stats['avg_memory']:.1f}"
                })
        
        if memory_table:
            print(tabulate(memory_table, headers='keys', tablefmt='grid'))
    
    def save_results(self, all_results: List[Dict], analysis: Dict, output_file: str):
        """Save results to JSON file."""
        output = {
            'summary': analysis,
            'detailed_results': all_results,
            'metadata': {
                'device': self.device,
                'total_texts': len(all_results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def run_benchmark(self, test_texts: List[str], output_file: Optional[str] = None):
        """Run complete benchmark."""
        print("\n" + "🚀"*35)
        print("HIERARCHICAL NER BENCHMARK")
        print("🚀"*35)
        
        # Load models
        self.load_all_models()
        
        # Run benchmarks
        print("\n" + "="*70)
        print(f"RUNNING BENCHMARKS ON {len(test_texts)} TEXTS")
        print("="*70)
        
        all_results = []
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n[{i}/{len(test_texts)}] Processing text {i}...")
            print(f"  Length: {len(text)} chars, {len(text.split())} words")
            
            result = self.benchmark_single_text(text, text_id=i)
            all_results.append(result)
            
            # Print quick summary
            hier_entities = result['approaches'].get('hierarchical', {}).get('entity_count', 0)
            hier_time = result['approaches'].get('hierarchical', {}).get('time', 0)
            print(f"  Hierarchical: {hier_entities} entities in {hier_time:.3f}s")
            
            # Clean up GPU cache periodically
            if i % 5 == 0 and self.use_gpu:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Analyze results
        print("\n" + "="*70)
        print("ANALYZING RESULTS...")
        print("="*70)
        
        analysis = self.analyze_results(all_results)
        
        # Print summary
        self.print_summary(analysis)
        
        # Save results
        if output_file:
            self.save_results(all_results, analysis, output_file)
        
        return all_results, analysis


def get_test_texts() -> List[str]:
    """Get diverse test texts for benchmarking."""
    return [
        # Easy - Clear entities
        "Apple Inc. announced new iPhone features at their headquarters in Cupertino, California.",
        
        # Easy - Multiple entity types
        "Microsoft CEO Satya Nadella spoke about Azure cloud services in Seattle, Washington.",
        
        # Medium - Some complexity
        "President Biden met with European leaders in Brussels to discuss NATO expansion and support for Ukraine.",
        
        # Medium - Multiple organizations and locations
        "Amazon is expanding its warehouses in Texas, Florida, and New York while competing with Walmart and Target.",
        
        # Hard - Ambiguous names
        "Jordan scored 23 points while visiting the Jordan River in the Middle East country of Jordan.",
        
        # Hard - Multiple ambiguous entities
        "Washington warned against foreign interference from Washington, D.C. while addressing the nation.",
        
        # Hard - Nested and complex
        "Tesla CEO Elon Musk announced plans to build a new Gigafactory in Berlin, Germany, following successful operations in Shanghai, China.",
        
        # Technical domain
        "Google's DeepMind AI division developed AlphaFold protein folding system using TensorFlow and PyTorch frameworks.",
        
        # Mixed complexity
        "The Nile River flows through Egypt, Sudan, and Ethiopia before reaching the Mediterranean Sea in Northeast Africa.",
        
        # Simple but clear
        "Tim Cook leads Apple in Cupertino."
    ]


def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark Hierarchical NER vs Single Models vs Large Model"
    )
    parser.add_argument(
        '--texts-file',
        type=str,
        help='JSON file with list of texts to benchmark'
    )
    parser.add_argument(
        '--num-texts',
        type=int,
        default=10,
        help='Number of texts to use (if using default test set)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage (even if GPU available)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with fewer texts'
    )
    
    args = parser.parse_args()
    
    # Load test texts
    if args.texts_file:
        with open(args.texts_file, 'r', encoding='utf-8') as f:
            test_texts = json.load(f)
            if isinstance(test_texts, dict) and 'texts' in test_texts:
                test_texts = test_texts['texts']
    else:
        test_texts = get_test_texts()
        if args.quick:
            test_texts = test_texts[:3]
        else:
            test_texts = test_texts[:args.num_texts]
    
    print(f"\nWill benchmark {len(test_texts)} texts")
    
    # Initialize benchmark
    benchmark = BenchmarkNER(use_gpu=not args.cpu)
    
    # Run benchmark
    try:
        all_results, analysis = benchmark.run_benchmark(
            test_texts,
            output_file=args.output
        )
        
        print("\n" + "✅"*35)
        print("BENCHMARK COMPLETE")
        print("✅"*35)
        
        # Final summary
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        
        if 'hierarchical' in analysis['approaches']:
            hier = analysis['approaches']['hierarchical']
            large = analysis['approaches'].get('large', {})
            
            if large:
                speedup = large.get('avg_time', 0) / hier.get('avg_time', 1) if hier.get('avg_time', 0) > 0 else 0
                print(f"\n✓ Hierarchical is {speedup:.2f}x faster than always-large model")
            
            if hier.get('large_model_usage_rate', 0) > 0:
                print(f"✓ Large model used only {hier['large_model_usage_rate']:.1%} of the time")
                print(f"  (Triggered {hier.get('large_model_triggered', 0)} out of {analysis['total_texts']} texts)")
        
        print("\nFor detailed results, see:", args.output)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

