# Scenario: Customer Support Chatbot Training
# Imagine you’re building a customer support chatbot for an e-commerce company.
# The chatbot needs to understand different types of sentences so it can respond appropriately.
# You feed it these sentences:
# - “Machine learning is a subset of AI”
# - “Deep learning uses neural networks”
# - “Bananas are yellow fruits”
# - “Artificial intelligence powers chatbots”

# Goal:
# Convert sentences into embeddings so the chatbot can understand semantic meaning
# and retrieve relevant answers later.

from sentence_transformers import SentenceTransformer

# List of embedding models (different LLM-based embedding architectures)
models = {
    "MiniLM": "all-MiniLM-L6-v2",
    "MPNet": "all-mpnet-base-v2",
    "BGE": "BAAI/bge-base-en-v1.5"
}

sentences = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Bananas are yellow fruits",
    "Artificial intelligence powers chatbots"
]

for name, model_name in models.items():
    
    print("\n")
    print(f"Running Model: {name}")
    
    model = SentenceTransformer(model_name)
    
    embeddings = model.encode(sentences)
    
    print("Embedding Shape:", embeddings.shape)
    print("Sample Embedding Values:", embeddings[0][:10])