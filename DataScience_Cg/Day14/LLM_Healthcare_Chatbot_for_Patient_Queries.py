# Healthcare Chatbot for Patient Queries
# 🏥 Scenario: Healthcare Chatbot for Patient Queries
# A hospital is developing a chatbot to help patients ask questions about their health.
# - A patient types:
# “Deep learning models are powerful” (or in practice, something like “I have chest pain and shortness of breath”).
# - The chatbot uses the BERT tokenizer to break the sentence into tokens:



from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

patient_query = "I have chest pain and shortness of breath"

encoded = tokenizer(
    patient_query,
    padding="max_length",
    truncation=True,
    max_length=20,
    return_tensors="pt"
)

tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

print("Patient Query:", patient_query)
print("Tokens:", tokens)
print("Token IDs:", encoded["input_ids"][0].tolist())
print("Attention Mask:", encoded["attention_mask"][0].tolist())