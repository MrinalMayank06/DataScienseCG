# Scenario: Corporate Expansion Analysis
# A business intelligence team analyzes press releases to extract useful
# information like company names, people, locations, and numbers.
# Named Entity Recognition (NER) helps convert raw news text into
# structured business insights.

# Step 1: Import required libraries
import spacy
from collections import defaultdict

# Step 2: Load spaCy English NLP model
# This model already contains tokenizer, POS tagger, parser, and NER
nlp = spacy.load("en_core_web_sm")

# Step 3: Press release text that needs analysis
text = """
Apple Inc. plans a new office in Hyderabad, India. Tim Cook announced this in March 2023.
The site will create 5,000 jobs and focus on innovative technologies in supply chain analytics.
"""

# Step 4: Process the text using spaCy pipeline
# This automatically performs tokenization and entity detection
doc = nlp(text)

# Step 5: Print all named entities detected in the text
# ent.text  -> actual entity
# ent.label_ -> entity category like ORG, PERSON, GPE etc.
print("Named Entities (text -> label):")

for ent in doc.ents:
    print(f"{ent.text:25} -> {ent.label_}")

# Step 6: Create a dictionary to group entities by their label
entities_by_type = defaultdict(list)

# Step 7: Store entities under their respective category
for ent in doc.ents:
    entities_by_type[ent.label_].append(ent.text)

# Step 8: Display grouped entities
# This gives a clean structured summary of important information
print("\nEntities grouped by label:")

for label, values in entities_by_type.items():
    print(f"{label}: {sorted(set(values))}")