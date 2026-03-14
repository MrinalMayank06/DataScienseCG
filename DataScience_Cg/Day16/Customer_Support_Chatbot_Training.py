# Scenario: Customer Support Chatbot Training
# Imagine you’re building a customer support chatbot for an e-commerce company.
# The chatbot needs to understand different types of sentences so it can respond appropriately.
# You feed it these sentences:
# - “Machine learning is a subset of AI”
# - “Deep learning uses neural networks”
# - “Bananas are yellow fruits”
# - “Artificial intelligence powers chatbots”




from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Bananas are yellow fruits",
    "Artificial intelligence powers chatbots"
]

embeddings = model.encode(sentences)

print(embeddings.shape)