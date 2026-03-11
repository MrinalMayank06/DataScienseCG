# Scenario: Research Assistant for Academic Papers
# Imagine you’re building a tool for university researchers who analyze large collections of academic papers.
# - A researcher uploads a sentence:
# “Deep learning models are powerful”
# - Your system uses the BERT tokenizer to break the sentence into smaller units (tokens).




# Research Assistant for Academic Papers
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "Deep learning models are powerful"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Input Text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)