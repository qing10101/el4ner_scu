# Hierarchical NER - Quick Start Guide

Get up and running with the Hierarchical NER system in 5 minutes!

## 🚀 Installation

### Option 1: Using pip (Recommended)

```bash
# Install dependencies
pip install -r requirements_hierarchical.txt
```

### Option 2: Manual installation

```bash
pip install torch transformers scipy numpy tabulate tqdm
```

## 🎯 Basic Usage

### 1. Simple Entity Extraction

```python
from hierarchical_ner_api import extract_entities

text = "Apple Inc. announced new products in California."
entities = extract_entities(text)

for entity in entities:
    print(f"{entity['entity']} - {entity['type']}")
```

**Output:**
```
Apple Inc. - ORGANIZATION
California - LOCATION
```

### 2. Extract Specific Types

```python
from hierarchical_ner_api import extract_persons, extract_locations, extract_organizations

text = "Tim Cook leads Apple in Cupertino."

persons = extract_persons(text)
locations = extract_locations(text)
orgs = extract_organizations(text)

print(f"Persons: {persons}")
print(f"Locations: {locations}")
print(f"Organizations: {orgs}")
```

**Output:**
```
Persons: ['Tim Cook']
Locations: ['Cupertino']
Organizations: ['Apple']
```

### 3. High-Confidence Entities Only

```python
from hierarchical_ner_api import get_high_confidence_entities

text = "Microsoft announced Azure updates in Seattle."
entities = get_high_confidence_entities(text, confidence_threshold=0.85)

for e in entities:
    print(f"{e['entity']} ({e['type']}) - Confidence: {e['confidence']}")
```

### 4. Batch Processing

```python
from hierarchical_ner_api import batch_extract

texts = [
    "Google launched new AI features.",
    "Amazon opened a warehouse in Texas.",
    "Tesla produces cars in Germany."
]

results = batch_extract(texts)

for result in results:
    print(f"Text: {result['text']}")
    print(f"Found {result['entity_count']} entities")
```

## ⚙️ Configuration Presets

### Fast Mode (Speed Priority)

```python
from hierarchical_ner_api import extract_entities

entities = extract_entities(text, preset='fast')
```

- Uses lighter models
- Higher uncertainty threshold (0.5)
- Large model rarely triggered (~10%)
- **Best for:** Real-time applications, high-volume processing

### Balanced Mode (Default)

```python
entities = extract_entities(text, preset='balanced')
```

- Standard BERT models
- Medium threshold (0.3)
- Large model triggered ~30%
- **Best for:** General purpose NER

### Accurate Mode (Accuracy Priority)

```python
entities = extract_entities(text, preset='accurate')
```

- Best available models
- Lower threshold (0.2)
- Large model triggered ~50%
- **Best for:** Research, critical applications

## 📊 Output Formats

### Simple Format

```python
entities = extract_entities(text, return_format='simple')
# Returns: [{'entity': 'Apple', 'type': 'ORG'}, ...]
```

### Grouped Format

```python
entities = extract_entities(text, return_format='grouped')
# Returns: {'ORG': ['Apple', 'Microsoft'], 'LOC': ['California']}
```

### Detailed Format (Default)

```python
entities = extract_entities(text, return_format='detailed')
# Returns full details including confidence, votes, positions
```

## 🔍 Advanced Features

### Analyze Uncertainty

```python
from hierarchical_ner_api import analyze_uncertainty

analysis = analyze_uncertainty("Jordan visited Jordan.")

print(f"Uncertainty: {analysis['uncertainty_score']}")
print(f"Would trigger large model: {analysis['would_trigger_large_model']}")
print(f"Recommendation: {analysis['recommendation']}")
```

### Custom Configuration

```python
from hierarchical_ner import HierarchicalNER

ner = HierarchicalNER(
    uncertainty_threshold=0.25,  # Adjust threshold
    use_gpu=True
)

entities = ner.predict(text)
```

### Save Results to File

```python
from hierarchical_ner_api import batch_extract, save_results

results = batch_extract(texts)
save_results(results, 'output.json', format='json')
```

## 💡 Common Use Cases

### News Article Processing

```python
from hierarchical_ner_api import extract_entities

article = """
President Biden met with European leaders in Brussels 
to discuss NATO expansion and support for Ukraine.
"""

entities = extract_entities(article, preset='accurate')
```

### Social Media Monitoring

```python
from hierarchical_ner_api import extract_entities

tweets = [
    "Just met @elonmusk at Tesla HQ in Austin!",
    "Amazon Prime Day deals are live in the US and UK."
]

results = batch_extract(tweets, preset='fast')
```

### Document Classification

```python
from hierarchical_ner_api import extract_entities_by_type

document = "Contract between Acme Corp and Smith Consulting for services in New York."

entities_by_type = extract_entities_by_type(document)

# Check if it's a contract (has ORG and LOC entities)
has_orgs = 'ORG' in entities_by_type or 'ORGANIZATION' in entities_by_type
has_locs = 'LOC' in entities_by_type or 'LOCATION' in entities_by_type

if has_orgs and has_locs:
    print("This appears to be a business contract")
```

## 🐛 Troubleshooting

### Out of Memory Error

```python
# Solution 1: Use CPU instead of GPU
ner = HierarchicalNER(use_gpu=False)

# Solution 2: Use fast preset
entities = extract_entities(text, preset='fast')
```

### Slow Performance

```python
# Solution 1: Increase threshold (use large model less)
ner = HierarchicalNER(uncertainty_threshold=0.5)

# Solution 2: Use fast preset
entities = extract_entities(text, preset='fast')
```

### Low Accuracy

```python
# Solution 1: Decrease threshold (use large model more)
ner = HierarchicalNER(uncertainty_threshold=0.2)

# Solution 2: Use accurate preset
entities = extract_entities(text, preset='accurate')
```

## 📚 Next Steps

1. **Read the full documentation**: See `HIERARCHICAL_NER_README.md`
2. **Try the examples**: Run `python hierarchical_ner_api.py`
3. **Run comparisons**: Execute `python compare_hierarchical_ner.py`
4. **Explore integration**: Check `integration_example.py`

## 🎓 Key Concepts

### What is Uncertainty?

Uncertainty measures how confident the small models are:
- **Low uncertainty** (0.0-0.3): Models agree, predictions are confident
- **Medium uncertainty** (0.3-0.5): Some disagreement or low confidence
- **High uncertainty** (0.5-1.0): Major disagreement or very low confidence

### What is Weighted Voting?

Each model gets votes based on its size:
- **Small models** (BERT, DistilBERT, RoBERTa): 1 vote each
- **Large model** (XLM-RoBERTa-large): 2 votes

Example: If BERT says "person", DistilBERT says "person", RoBERTa says "location", 
and Large model says "person", then:
- Person: 4 votes (BERT:1 + DistilBERT:1 + Large:2)
- Location: 1 vote (RoBERTa:1)
- **Winner: Person**

### When Does Large Model Help?

The large model is most useful for:
1. Ambiguous names (Jordan = person or place?)
2. Multiple entities with same text (Apple the company vs apple the fruit)
3. Complex or technical text
4. When small models disagree

## 💻 Complete Working Example

```python
#!/usr/bin/env python3
"""Complete working example of Hierarchical NER."""

from hierarchical_ner_api import (
    extract_entities,
    extract_persons,
    get_high_confidence_entities,
    analyze_uncertainty
)

def main():
    # Sample text
    text = "Microsoft CEO Satya Nadella announced Azure AI updates in Seattle."
    
    # 1. Extract all entities
    print("All entities:")
    entities = extract_entities(text, preset='balanced')
    for e in entities:
        print(f"  • {e['entity']} ({e['type']}) - Confidence: {e['confidence']}")
    
    # 2. Extract only persons
    print("\nPersons:")
    persons = extract_persons(text)
    print(f"  {persons}")
    
    # 3. High-confidence only
    print("\nHigh-confidence entities:")
    high_conf = get_high_confidence_entities(text, confidence_threshold=0.9)
    for e in high_conf:
        print(f"  • {e['entity']} - {e['confidence']}")
    
    # 4. Analyze uncertainty
    print("\nUncertainty analysis:")
    analysis = analyze_uncertainty(text)
    print(f"  Uncertainty: {analysis['uncertainty_score']}")
    print(f"  Recommendation: {analysis['recommendation']}")

if __name__ == "__main__":
    main()
```

Save this as `my_ner_script.py` and run:

```bash
python my_ner_script.py
```

## 🎉 You're Ready!

Start extracting entities with the Hierarchical NER system. Happy coding! 🚀

For questions or issues, see the full documentation in `HIERARCHICAL_NER_README.md`.

