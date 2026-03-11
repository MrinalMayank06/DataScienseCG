# Scenario: Sports Event Analysis
# A sports analytics team wants to extract useful information from
# match reports such as team names, player names, stadiums, locations,
# dates, and scores. Named Entity Recognition (NER) helps convert
# raw sports text into structured insights.

# Step 1: Import required libraries
import spacy
from collections import defaultdict

# Step 2: Load spaCy English NLP model
# This model can detect entities like PERSON, ORG, GPE, DATE, CARDINAL, etc.
nlp = spacy.load("en_core_web_sm")

# Step 3: Match report text
text = """
Manchester United defeated Real Madrid 3-2 at Wembley Stadium, London.
Cristiano Ronaldo scored twice, while Marcus Rashford netted the winning goal in July 2024.
"""

# Step 4: Process the text using spaCy pipeline
# spaCy will tokenize the text and detect named entities
doc = nlp(text)

# Step 5: Print all detected named entities with their labels
print("Named Entities (text -> label):")

for ent in doc.ents:
    print(f"{ent.text:25} -> {ent.label_}")

# Step 6: Create a dictionary to group entities by category
entities_by_type = defaultdict(list)

# Step 7: Store each entity under its label
for ent in doc.ents:
    entities_by_type[ent.label_].append(ent.text)

# Step 8: Print grouped entities
print("\nEntities grouped by label:")

for label, values in entities_by_type.items():
    print(f"{label}: {sorted(set(values))}")