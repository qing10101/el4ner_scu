"""
Hierarchical NER System with Uncertainty-Based Model Selection
Uses 3 small models (BERT, DistilBERT, RoBERTa) with voting mechanism.
If uncertainty is high, consults a larger model with 2x voting weight.
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HierarchicalNER:
    """
    A hierarchical NER system that uses small models first and escalates
    to a larger model when uncertainty is detected.
    """
    
    def __init__(
        self, 
        uncertainty_threshold: float = 0.3,
        use_gpu: bool = True
    ):
        """
        Initialize the hierarchical NER system.
        
        Args:
            uncertainty_threshold: Threshold for uncertainty (0-1). Higher means more uncertain.
            use_gpu: Whether to use GPU if available.
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        
        # Model configurations
        self.small_models_config = {
            'bert': 'dslim/bert-base-NER',
            'distilbert': 'dslim/distilbert-NER',
            'roberta': 'Jean-Baptiste/roberta-large-ner-english'
        }
        
        self.large_model_config = 'xlm-roberta-large-finetuned-conll03-english'
        
        # Initialize models
        self.small_models = {}
        self.large_model = None
        self.large_model_loaded = False
        
        print("Initializing Hierarchical NER System...")
        self._load_small_models()
        
    def _load_small_models(self):
        """Load all small models into memory."""
        print("\nLoading small models...")
        for name, model_id in self.small_models_config.items():
            try:
                print(f"  Loading {name}...")
                self.small_models[name] = pipeline(
                    "ner",
                    model=model_id,
                    tokenizer=model_id,
                    aggregation_strategy="simple",
                    device=self.device
                )
                print(f"  ✓ {name} loaded successfully")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
        
        print(f"\n✓ Loaded {len(self.small_models)}/3 small models")
    
    def _load_large_model(self):
        """Lazy loading of the large model (only when needed)."""
        if not self.large_model_loaded:
            print("\n⚠ High uncertainty detected! Loading large model...")
            try:
                self.large_model = pipeline(
                    "ner",
                    model=self.large_model_config,
                    tokenizer=self.large_model_config,
                    aggregation_strategy="simple",
                    device=self.device
                )
                self.large_model_loaded = True
                print("✓ Large model loaded successfully\n")
            except Exception as e:
                print(f"✗ Failed to load large model: {e}\n")
                self.large_model_loaded = False
    
    def _unload_large_model(self):
        """Unload large model to free memory."""
        if self.large_model_loaded:
            del self.large_model
            self.large_model = None
            self.large_model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ Large model unloaded to free memory")
    
    def _calculate_uncertainty(self, entities: List[Dict]) -> float:
        """
        Calculate uncertainty score based on prediction confidence.
        
        Args:
            entities: List of entity predictions with scores
            
        Returns:
            Uncertainty score (0-1, higher means more uncertain)
        """
        if not entities:
            return 1.0  # Maximum uncertainty if no entities found
        
        # Calculate uncertainty as 1 - average_confidence
        scores = [entity['score'] for entity in entities]
        avg_confidence = np.mean(scores)
        uncertainty = 1.0 - avg_confidence
        
        # Also consider variance in predictions as additional uncertainty
        if len(scores) > 1:
            score_variance = np.var(scores)
            uncertainty = (uncertainty + score_variance) / 2
        
        return uncertainty
    
    def _normalize_entity(self, entity: Dict) -> Tuple[str, str, int, int]:
        """
        Normalize entity representation for comparison.
        
        Returns:
            Tuple of (text, entity_type, start, end)
        """
        return (
            entity['word'].strip().lower(),
            entity['entity_group'],
            entity['start'],
            entity['end']
        )
    
    def _run_small_models(self, text: str) -> Tuple[Dict[str, List[Dict]], float]:
        """
        Run all small models and calculate aggregate uncertainty.
        
        Args:
            text: Input text for NER
            
        Returns:
            Tuple of (predictions dict, max uncertainty)
        """
        predictions = {}
        uncertainties = []
        
        print("\n" + "="*70)
        print("STAGE 1: Running Small Models")
        print("="*70)
        
        for name, model in self.small_models.items():
            try:
                entities = model(text)
                predictions[name] = entities
                uncertainty = self._calculate_uncertainty(entities)
                uncertainties.append(uncertainty)
                
                print(f"\n{name.upper()}:")
                print(f"  Found {len(entities)} entities")
                print(f"  Uncertainty: {uncertainty:.3f}")
                if entities:
                    print(f"  Avg confidence: {np.mean([e['score'] for e in entities]):.3f}")
            except Exception as e:
                print(f"  ✗ Error running {name}: {e}")
                uncertainties.append(1.0)
        
        max_uncertainty = max(uncertainties) if uncertainties else 1.0
        return predictions, max_uncertainty
    
    def _run_large_model(self, text: str) -> List[Dict]:
        """
        Run the large model.
        
        Args:
            text: Input text for NER
            
        Returns:
            List of entity predictions
        """
        if not self.large_model_loaded:
            self._load_large_model()
        
        if self.large_model:
            print("\n" + "="*70)
            print("STAGE 2: Running Large Model (counts as 2 votes)")
            print("="*70)
            try:
                entities = self.large_model(text)
                print(f"  Found {len(entities)} entities")
                if entities:
                    print(f"  Avg confidence: {np.mean([e['score'] for e in entities]):.3f}")
                return entities
            except Exception as e:
                print(f"  ✗ Error running large model: {e}")
                return []
        return []
    
    def _weighted_voting(
        self, 
        small_predictions: Dict[str, List[Dict]], 
        large_predictions: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Perform weighted voting across all models.
        Small models: 1 vote each
        Large model: 2 votes
        
        Args:
            small_predictions: Dict of predictions from small models
            large_predictions: Predictions from large model (if used)
            
        Returns:
            Final voted entities
        """
        print("\n" + "="*70)
        print("STAGE 3: Weighted Voting")
        print("="*70)
        
        # Collect all unique entities by position
        entity_votes = defaultdict(list)
        
        # Add small model votes (weight = 1)
        for model_name, entities in small_predictions.items():
            for entity in entities:
                key = (entity['start'], entity['end'])
                entity_votes[key].append({
                    'word': entity['word'],
                    'entity_group': entity['entity_group'],
                    'score': entity['score'],
                    'weight': 1,
                    'source': model_name
                })
        
        # Add large model votes (weight = 2)
        if large_predictions:
            for entity in large_predictions:
                key = (entity['start'], entity['end'])
                entity_votes[key].append({
                    'word': entity['word'],
                    'entity_group': entity['entity_group'],
                    'score': entity['score'],
                    'weight': 2,
                    'source': 'large_model'
                })
        
        # Perform weighted voting
        final_entities = []
        
        for (start, end), votes in entity_votes.items():
            # Count weighted votes for each entity type
            type_votes = defaultdict(float)
            type_scores = defaultdict(list)
            type_words = defaultdict(list)
            
            for vote in votes:
                entity_type = vote['entity_group']
                weight = vote['weight']
                type_votes[entity_type] += weight
                type_scores[entity_type].append(vote['score'])
                type_words[entity_type].append(vote['word'])
            
            # Select entity type with most weighted votes
            winner_type = max(type_votes.items(), key=lambda x: x[1])[0]
            winner_votes = type_votes[winner_type]
            
            # Calculate average score and select most common word form
            avg_score = np.mean(type_scores[winner_type])
            most_common_word = Counter(type_words[winner_type]).most_common(1)[0][0]
            
            final_entities.append({
                'word': most_common_word,
                'entity_group': winner_type,
                'start': start,
                'end': end,
                'score': avg_score,
                'total_votes': winner_votes,
                'vote_details': dict(type_votes)
            })
        
        # Sort by start position
        final_entities.sort(key=lambda x: x['start'])
        
        # Print voting summary
        print(f"\nVoting Summary:")
        print(f"  Total unique entity positions: {len(entity_votes)}")
        print(f"  Final entities after voting: {len(final_entities)}")
        
        if large_predictions:
            print(f"  Large model participated: YES (2x weight)")
        else:
            print(f"  Large model participated: NO (uncertainty below threshold)")
        
        return final_entities
    
    def predict(self, text: str, verbose: bool = True) -> List[Dict]:
        """
        Main prediction method with hierarchical model selection.
        
        Args:
            text: Input text for NER
            verbose: Whether to print detailed information
            
        Returns:
            List of final entity predictions
        """
        if not verbose:
            # Temporarily suppress prints
            import sys
            import os
            sys.stdout = open(os.devnull, 'w')
        
        try:
            # Stage 1: Run small models
            small_predictions, uncertainty = self._run_small_models(text)
            
            # Stage 2: Decide whether to use large model
            large_predictions = None
            if uncertainty > self.uncertainty_threshold:
                large_predictions = self._run_large_model(text)
            else:
                print(f"\n✓ Uncertainty ({uncertainty:.3f}) below threshold ({self.uncertainty_threshold})")
                print("  Skipping large model (using small models only)")
            
            # Stage 3: Weighted voting
            final_entities = self._weighted_voting(small_predictions, large_predictions)
            
            return final_entities
            
        finally:
            if not verbose:
                sys.stdout = sys.__stdout__
    
    def print_results(self, entities: List[Dict]):
        """Pretty print the NER results."""
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        
        if not entities:
            print("No entities found.")
            return
        
        print(f"\nFound {len(entities)} entities:\n")
        
        for i, entity in enumerate(entities, 1):
            print(f"{i}. '{entity['word']}'")
            print(f"   Type: {entity['entity_group']}")
            print(f"   Position: {entity['start']}-{entity['end']}")
            print(f"   Confidence: {entity['score']:.3f}")
            print(f"   Total Votes: {entity['total_votes']}")
            print(f"   Vote Breakdown: {entity['vote_details']}")
            print()


def main():
    """Example usage of the Hierarchical NER system."""
    
    # Example texts with varying difficulty
    test_texts = [
        # Easy case - clear entities
        "Apple Inc. is planning to open a new store in New York City next month.",
        
        # Medium case - some ambiguity
        "The President met with Angela Merkel in Berlin to discuss climate change.",
        
        # Hard case - more ambiguous
        "Jordan scored 23 points while visiting the Jordan River in the Middle East."
    ]
    
    # Initialize system
    ner_system = HierarchicalNER(
        uncertainty_threshold=0.3,
        use_gpu=True
    )
    
    # Process each text
    for i, text in enumerate(test_texts, 1):
        print("\n" + "🔷"*35)
        print(f"EXAMPLE {i}")
        print("🔷"*35)
        print(f"\nInput: {text}")
        
        entities = ner_system.predict(text, verbose=True)
        ner_system.print_results(entities)
        
        print("\n" + "-"*70 + "\n")
    
    # Clean up
    ner_system._unload_large_model()


if __name__ == "__main__":
    main()

