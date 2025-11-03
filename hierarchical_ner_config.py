"""
Configuration and Advanced Features for Hierarchical NER System
Includes different uncertainty metrics and customization options
"""

import torch
import numpy as np
from typing import List, Dict, Callable
from scipy.stats import entropy


class UncertaintyMetrics:
    """Various uncertainty calculation methods for model predictions."""
    
    @staticmethod
    def confidence_based(entities: List[Dict]) -> float:
        """
        Basic uncertainty based on average confidence.
        Lower confidence = higher uncertainty.
        
        Returns:
            Uncertainty score (0-1)
        """
        if not entities:
            return 1.0
        
        scores = [entity['score'] for entity in entities]
        avg_confidence = np.mean(scores)
        return 1.0 - avg_confidence
    
    @staticmethod
    def entropy_based(entities: List[Dict]) -> float:
        """
        Uncertainty based on entropy of confidence scores.
        Higher entropy = more uncertain predictions.
        
        Returns:
            Normalized uncertainty score (0-1)
        """
        if not entities:
            return 1.0
        
        scores = np.array([entity['score'] for entity in entities])
        # Normalize scores to probabilities
        probs = scores / scores.sum() if scores.sum() > 0 else scores
        
        # Calculate entropy
        ent = entropy(probs + 1e-10)  # Add small epsilon to avoid log(0)
        
        # Normalize entropy to [0, 1]
        max_entropy = np.log(len(scores))
        normalized_entropy = ent / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    @staticmethod
    def variance_based(entities: List[Dict]) -> float:
        """
        Uncertainty based on variance of confidence scores.
        High variance = inconsistent predictions = more uncertain.
        
        Returns:
            Uncertainty score (0-1)
        """
        if not entities or len(entities) < 2:
            return 1.0 if not entities else 0.5
        
        scores = [entity['score'] for entity in entities]
        variance = np.var(scores)
        
        # Normalize variance (theoretical max is 0.25 for binary [0,1])
        normalized_variance = min(variance / 0.25, 1.0)
        
        return normalized_variance
    
    @staticmethod
    def combined_metric(entities: List[Dict]) -> float:
        """
        Combined uncertainty metric using multiple factors.
        
        Returns:
            Weighted uncertainty score (0-1)
        """
        if not entities:
            return 1.0
        
        confidence_unc = UncertaintyMetrics.confidence_based(entities)
        entropy_unc = UncertaintyMetrics.entropy_based(entities)
        variance_unc = UncertaintyMetrics.variance_based(entities)
        
        # Weighted combination
        combined = (
            0.5 * confidence_unc +
            0.3 * entropy_unc +
            0.2 * variance_unc
        )
        
        return combined


class HierarchicalNERConfig:
    """Configuration class for the Hierarchical NER system."""
    
    # Pre-configured model sets
    SMALL_MODEL_PRESETS = {
        'fast': {
            'distilbert': 'dslim/distilbert-NER',
            'bert-tiny': 'Davlan/bert-base-multilingual-cased-ner-hrl',
        },
        'balanced': {
            'bert': 'dslim/bert-base-NER',
            'distilbert': 'dslim/distilbert-NER',
            'roberta': 'Jean-Baptiste/roberta-large-ner-english'
        },
        'multilingual': {
            'xlm-roberta': 'xlm-roberta-base-finetuned-conll03-english',
            'bert-multilingual': 'Davlan/bert-base-multilingual-cased-ner-hrl',
            'distilbert': 'dslim/distilbert-NER'
        }
    }
    
    LARGE_MODEL_PRESETS = {
        'default': 'xlm-roberta-large-finetuned-conll03-english',
        'deberta': 'dslim/deberta-large-NER',
        'roberta-large': 'Jean-Baptiste/roberta-large-ner-english'
    }
    
    def __init__(
        self,
        small_models_preset: str = 'balanced',
        large_model_preset: str = 'default',
        uncertainty_threshold: float = 0.3,
        uncertainty_metric: str = 'combined',
        enable_caching: bool = True,
        batch_processing: bool = False
    ):
        """
        Initialize configuration.
        
        Args:
            small_models_preset: Choose from 'fast', 'balanced', 'multilingual'
            large_model_preset: Choose from 'default', 'deberta', 'roberta-large'
            uncertainty_threshold: Threshold for triggering large model (0-1)
            uncertainty_metric: 'confidence', 'entropy', 'variance', or 'combined'
            enable_caching: Cache model predictions
            batch_processing: Process multiple texts efficiently
        """
        self.small_models = self.SMALL_MODEL_PRESETS.get(
            small_models_preset, 
            self.SMALL_MODEL_PRESETS['balanced']
        )
        
        self.large_model = self.LARGE_MODEL_PRESETS.get(
            large_model_preset,
            self.LARGE_MODEL_PRESETS['default']
        )
        
        self.uncertainty_threshold = uncertainty_threshold
        self.enable_caching = enable_caching
        self.batch_processing = batch_processing
        
        # Select uncertainty metric
        metric_map = {
            'confidence': UncertaintyMetrics.confidence_based,
            'entropy': UncertaintyMetrics.entropy_based,
            'variance': UncertaintyMetrics.variance_based,
            'combined': UncertaintyMetrics.combined_metric
        }
        self.uncertainty_metric = metric_map.get(
            uncertainty_metric,
            UncertaintyMetrics.combined_metric
        )
    
    def get_voting_weights(self) -> Dict[str, float]:
        """
        Get voting weights for models.
        
        Returns:
            Dict mapping model type to vote weight
        """
        return {
            'small': 1.0,
            'large': 2.0
        }
    
    def summary(self) -> str:
        """Return a summary of the configuration."""
        summary = []
        summary.append("=" * 70)
        summary.append("HIERARCHICAL NER CONFIGURATION")
        summary.append("=" * 70)
        summary.append(f"\nSmall Models ({len(self.small_models)}):")
        for name, model_id in self.small_models.items():
            summary.append(f"  - {name}: {model_id}")
        summary.append(f"\nLarge Model:")
        summary.append(f"  - {self.large_model}")
        summary.append(f"\nSettings:")
        summary.append(f"  - Uncertainty Threshold: {self.uncertainty_threshold}")
        summary.append(f"  - Uncertainty Metric: {self.uncertainty_metric.__name__}")
        summary.append(f"  - Caching: {'Enabled' if self.enable_caching else 'Disabled'}")
        summary.append(f"  - Batch Processing: {'Enabled' if self.batch_processing else 'Disabled'}")
        summary.append("\nVoting Weights:")
        for model_type, weight in self.get_voting_weights().items():
            summary.append(f"  - {model_type} model: {weight} vote(s)")
        summary.append("=" * 70)
        return "\n".join(summary)


class ModelPerformanceTracker:
    """Track and analyze model performance over multiple predictions."""
    
    def __init__(self):
        self.stats = {
            'total_predictions': 0,
            'large_model_triggered': 0,
            'small_model_only': 0,
            'uncertainties': [],
            'entity_counts': [],
            'processing_times': []
        }
    
    def record_prediction(
        self, 
        uncertainty: float, 
        used_large_model: bool,
        num_entities: int,
        processing_time: float = 0.0
    ):
        """Record statistics from a prediction."""
        self.stats['total_predictions'] += 1
        self.stats['uncertainties'].append(uncertainty)
        self.stats['entity_counts'].append(num_entities)
        
        if used_large_model:
            self.stats['large_model_triggered'] += 1
        else:
            self.stats['small_model_only'] += 1
        
        if processing_time > 0:
            self.stats['processing_times'].append(processing_time)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if self.stats['total_predictions'] == 0:
            return {"message": "No predictions recorded yet"}
        
        summary = {
            'total_predictions': self.stats['total_predictions'],
            'large_model_usage_rate': self.stats['large_model_triggered'] / self.stats['total_predictions'],
            'avg_uncertainty': np.mean(self.stats['uncertainties']),
            'avg_entities_found': np.mean(self.stats['entity_counts']),
        }
        
        if self.stats['processing_times']:
            summary['avg_processing_time'] = np.mean(self.stats['processing_times'])
        
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        
        if 'message' in summary:
            print(summary['message'])
            return
        
        print(f"\nTotal Predictions: {summary['total_predictions']}")
        print(f"Large Model Usage: {summary['large_model_usage_rate']:.1%}")
        print(f"  - Triggered: {self.stats['large_model_triggered']} times")
        print(f"  - Small only: {self.stats['small_model_only']} times")
        print(f"\nAverage Uncertainty: {summary['avg_uncertainty']:.3f}")
        print(f"Average Entities Found: {summary['avg_entities_found']:.1f}")
        
        if 'avg_processing_time' in summary:
            print(f"Average Processing Time: {summary['avg_processing_time']:.2f}s")
        
        print("=" * 70)


# Example usage configurations
def get_fast_config():
    """Configuration optimized for speed."""
    return HierarchicalNERConfig(
        small_models_preset='fast',
        large_model_preset='default',
        uncertainty_threshold=0.5,  # Higher threshold = less likely to use large model
        uncertainty_metric='confidence',
        enable_caching=True
    )


def get_accurate_config():
    """Configuration optimized for accuracy."""
    return HierarchicalNERConfig(
        small_models_preset='balanced',
        large_model_preset='deberta',
        uncertainty_threshold=0.2,  # Lower threshold = more likely to use large model
        uncertainty_metric='combined',
        enable_caching=True
    )


def get_multilingual_config():
    """Configuration for multilingual texts."""
    return HierarchicalNERConfig(
        small_models_preset='multilingual',
        large_model_preset='default',
        uncertainty_threshold=0.3,
        uncertainty_metric='combined',
        enable_caching=True
    )


if __name__ == "__main__":
    # Demo different configurations
    configs = [
        ("Fast Config", get_fast_config()),
        ("Accurate Config", get_accurate_config()),
        ("Multilingual Config", get_multilingual_config())
    ]
    
    for name, config in configs:
        print(f"\n{'='*70}")
        print(f"{name}")
        print(config.summary())

