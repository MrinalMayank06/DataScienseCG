# Scenario: Student Course Recommendations
# Imagine you’re running an online learning platform.
# Students type queries like “beginner-friendly Python tutorials”,
# but your catalog has courses titled “Introduction to Programming with Python”.
# Traditional keyword search might miss the match, but embeddings + vector search
# (Chroma locally or Pinecone in the cloud) can connect them semantically.

# 🧩 Teaching Exercise Flow
# - Documents (Courses)
#   - "Python is a high-level programming language"
#   - "Machine learning models need training data"
#   - "Dogs are loyal and friendly animals"
#   - "Cats are independent and curious pets"
#
# - Embedding Model
#   - Converts each course description into a vector (numbers capturing meaning).
#
# - Indexing
#   - Chroma (local): Great for classroom demos, no API key needed.
#
# - Query
#   - Student asks: “What animals make good companions?”
#
# - Retrieval
#   - Embedding of query is compared against course embeddings.
#
# - Results
#   - Top matches:
#     - Cats are independent and curious pets
#     - Dogs are loyal and friendly animals

# OPTION A: Chroma (local, no API key needed)

 

import chromadb
from sentence_transformers import SentenceTransformer

# Step 1: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Create Chroma client and collection
client = chromadb.Client()   # For persistent storage, use chromadb.PersistentClient(path="./db")
collection = client.create_collection("docs")

# Step 3: Sample documents
documents = [
    "Python is a high-level programming language",
    "Machine learning models need training data",
    "Dogs are loyal and friendly animals",
    "Cats are independent and curious pets",
]

# Step 4: Convert documents into embeddings
embeddings = model.encode(documents).tolist()

# Step 5: Add documents + embeddings into Chroma collection
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc{i}" for i in range(len(documents))],
    metadatas=[{"source": "demo", "idx": i} for i in range(len(documents))]
)

# Step 6: Semantic query
query = "What animals make good companions?"
q_emb = model.encode([query]).tolist()

# Step 7: Search top matching documents
results = collection.query(
    query_embeddings=q_emb,
    n_results=3
)

# Step 8: Print retrieved results
print(f"Query: {query}\n")
print("Top Matches:")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"[{dist:.3f}] {doc}")