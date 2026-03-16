# Scenario: AI Research Assistant for a Corporate Innovation Team
# Imagine you’re part of a corporate innovation lab that constantly reviews new AI research papers to stay ahead of
# trends. The team struggles with long PDFs full of technical jargon, and they want a quick way to ask natural questions
# about the papers instead of reading them cover to cover.
# How the RAG Chatbot Fits In
# - Input Source: The team uploads a research paper (e.g., ai_research.pdf).
# - Chunking: The chatbot splits the paper into manageable sections so no detail is lost.
# - Embeddings + Vector DB: Each section is converted into embeddings and stored in Chroma, making the paper searchable by meaning rather than keywords.
# - Retriever: When someone asks, “What does this paper say about reinforcement learning?”, the retriever pulls the most relevant chunks.
# - LLM Response: The Hugging Face model (Flan-T5) generates a concise, human-readable answer using those chunks as context.
# - Chat Loop: The team can keep asking questions interactively, like a research assistant that knows the paper inside out.

# ==========================================================
# SIMPLE RAG CHATBOT (NO LANGCHAIN) — FULLY ANNOTATED
# ==========================================================

# ----------------------------------------------------------
# STEP 0 — Install Required Libraries
# ----------------------------------------------------------
# chromadb → vector database
# sentence-transformers → embedding model
# pypdf → reading PDF files
# transformers → running the LLM




# ----------------------------------------------------------
# STEP 1 — Import Libraries
# ----------------------------------------------------------

import os

# Library for reading PDF documents
from pypdf import PdfReader

# Embedding model
from sentence_transformers import SentenceTransformer

# Vector database
import chromadb

# HuggingFace model pipeline
from transformers import pipeline


# ----------------------------------------------------------
# STEP 2 — Load the PDF Document
# ----------------------------------------------------------
# This document acts as the knowledge source for RAG

print("Loading PDF document...")

reader = PdfReader(r"D:\CG\datascience\DeepLearningCG\Day17\AI_Research_Assistant_for_a_Corporate_Innovation_Team\ai_research.pdf")

text = ""

# Extract text from every page
for page in reader.pages:
    text += page.extract_text()

print("Document Loaded")
print("Total Characters:", len(text))

# Preview some text
print("\nPreview:\n")
print(text[:500])


# ----------------------------------------------------------
# STEP 3 — Chunk the Document
# ----------------------------------------------------------
# LLMs work better with smaller text segments
# so we split the document into chunks

print("\nSplitting document into chunks...")

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks

    chunk_size = max characters per chunk
    overlap = shared characters between chunks
    """

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = chunk_text(text)

print("Total Chunks Created:", len(chunks))

print("\nExample Chunk:\n")
print(chunks[0])


# ----------------------------------------------------------
# STEP 4 — Create Embeddings
# ----------------------------------------------------------
# Convert each chunk into a numerical vector
# These vectors allow semantic similarity search

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ----------------------------------------------------------
# STEP 5 — Create Vector Database
# ----------------------------------------------------------
# ChromaDB stores embeddings and documents

client = chromadb.Client()

# Delete collection if it exists
try:
    client.delete_collection("pdf_collection")
    print("Old collection deleted")
except:
    pass


collection = client.create_collection("pdf_collection")

print("New vector collection created")


# ----------------------------------------------------------
# STEP 6 — Store Chunks in Vector DB
# ----------------------------------------------------------

print("\nCreating embeddings and storing in ChromaDB...")

for i, chunk in enumerate(chunks):

    # Create embedding vector
    embedding = embedding_model.encode(chunk).tolist()

    # Store in vector database
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[str(i)]
    )

print("All chunks stored successfully")


# ----------------------------------------------------------
# STEP 7 — Retriever Function
# ----------------------------------------------------------
# Converts user question → embedding
# Finds most similar document chunks

def retrieve(query, k=3):

    # Convert query to embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]


# ----------------------------------------------------------
# STEP 8 — Load the LLM
# ----------------------------------------------------------
# We use Flan-T5 for answer generation

print("\nLoading LLM...")

qa_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

print("LLM loaded successfully")


# ----------------------------------------------------------
# STEP 9 — Question Answering Function
# ----------------------------------------------------------
# RAG Workflow:
#
# Question
#   ↓
# Retrieve Relevant Chunks
#   ↓
# Send Context + Question to LLM
#   ↓
# Generate Answer

def answer_question(query):

    # Retrieve relevant context
    context_docs = retrieve(query)

    context = " ".join(context_docs)

    prompt = f"""
You are an AI assistant.

Answer the question using ONLY the context below.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{query}

Answer:
"""

    response = qa_pipeline(
        prompt,
        max_length=256,
        temperature=0.5
    )

    return response[0]["generated_text"]


# ----------------------------------------------------------
# STEP 10 — Chat Loop
# ----------------------------------------------------------
# Interactive question-answering interface

print("\n==============================")
print("RAG Chatbot Ready")
print("Type 'exit' to stop")
print("==============================\n")

while True:

    question = input("Ask a question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    answer = answer_question(question)

    print("\nAnswer:\n", answer)
    print("\n" + "-"*60 + "\n")