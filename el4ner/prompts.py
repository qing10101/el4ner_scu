# --- Prompt Templates for each stage of the EL4NER pipeline ---

ZERO_SHOT_EXTRACTION_TEMPLATE = """You are a Named Entity Recognition expert. Your task is to list all potential named entities from the sentence.
- Separate each entity with '[SEP]'.
- Do not include explanations.
- Start with '[CLS]' and end with '[SEP]'.

Sentence: "{text}"
Entities:"""

FEW_SHOT_EXTRACTION_TEMPLATE = """You are a Named Entity Recognition expert. Your task is to list all potential named entities from the sentence, following the examples.
- Separate each entity with '[SEP]'.
- Do not include explanations.
- Start with '[CLS]' and end with '[SEP]'.

{demonstrations}Sentence: "{text}"
Entities:"""

FEW_SHOT_CLASSIFICATION_TEMPLATE = """You are a Named Entity Recognition expert. Your task is to classify the given potential entities from the provided sentence based on the examples.
- The output format must be: [CLS]entity1[SEP]type1[CLS]entity2[SEP]type2...
- Do not introduce new entities.
- Classify only the provided entities.

{demonstrations}Sentence: "{text}"
Spans to classify: {spans_to_classify}
Classification:"""

VERIFICATION_TEMPLATE = """You are a highly skilled linguist. Evaluate if the entity "{span}" in the sentence corresponds to the type "{entity_type}".
Respond strictly with "true" or "false", without any explanation.

Sentence: "{text}"
Is "{span}" a "{entity_type}"?
Answer:"""

def format_extraction_demos(demos):
    demo_str = ""
    for demo in demos:
        entities = list(demo['entities'].keys())
        formatted_spans = "[SEP]".join(entities)
        demo_str += f"Sentence: \"{demo['text']}\"\nEntities: [CLS]{formatted_spans}[SEP]\n\n"
    return demo_str

def format_classification_demos(demos):
    demo_str = ""
    for demo in demos:
        input_spans = list(demo['entities'].keys())
        output_pairs = "".join([f"[CLS]{span}[SEP]{type}" for span, type in demo['entities'].items()])
        demo_str += f"Sentence: \"{demo['text']}\"\nSpans to classify: {'[SEP]'.join(input_spans)}\nClassification: {output_pairs}\n\n"
    return demo_str