# Scenario: University Library Assistant
# A large university library has thousands of digitized textbooks, research papers, and course notes.
# Students often struggle to find specific explanations or summaries when preparing for exams.
# This project builds a simple RAG chatbot without LangChain.
# It reads a textbook PDF, splits it into chunks, creates embeddings,
# stores them in ChromaDB, retrieves relevant sections, and uses Flan-T5
# to answer student questions in clear and simple language.

# Install required libraries before running:
# pip install chromadb sentence-transformers pypdf transformers torch

import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


PDF_PATH = r"D:\CG\datascience\DeepLearningCG\Day17\Introduction_to_Data_Science.pdf"
CHROMA_PATH = "./library_rag_db"
COLLECTION_NAME = "university_library_collection"


def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            full_text += f"\n\n[Page {page_number}]\n{page_text}"

    return full_text


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)
    return client, collection


def store_chunks(collection, chunks, embedding_model):
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"]
        )


def retrieve(query, collection, embedding_model, k=3):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]


def build_prompt(context, query):
    prompt = f"""
You are a university library study assistant.

Answer the student's question using ONLY the context below.
If the answer is not present, say: Not found in document.
Explain in simple, clear, student-friendly language.
Keep the answer concise and accurate.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt.strip()


def generate_answer(tokenizer, model, prompt, max_new_tokens=180):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def answer_question(query, collection, embedding_model, tokenizer, model):
    context_docs = retrieve(query, collection, embedding_model, k=3)
    context = "\n\n".join(context_docs)

    prompt = build_prompt(context, query)
    answer = generate_answer(tokenizer, model, prompt)

    return answer, context_docs


def main():
    if not os.path.exists(PDF_PATH):
        print("PDF file not found. Check PDF_PATH.")
        return

    print("Loading PDF document...")
    text = load_pdf_text(PDF_PATH)

    print("Document loaded")
    print("Total characters:", len(text))

    print("\nSplitting document into chunks...")
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    print("Total chunks created:", len(chunks))

    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded")

    print("\nCreating vector database...")
    _, collection = create_collection()
    print("New vector collection created")

    print("\nCreating embeddings and storing in ChromaDB...")
    store_chunks(collection, chunks, embedding_model)
    print("All chunks stored successfully")

    print("\nLoading LLM...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    print("LLM loaded successfully")

    print("\n==============================")
    print("University Library RAG Chatbot Ready")
    print("Type 'exit' to stop")
    print("==============================\n")

    print("Example Questions:")
    print("1. What is Data Science?")
    print("2. Explain the difference between supervised and unsupervised learning.")
    print("3. What are the steps in the Data Science workflow?")
    print("4. What tools are used in data science?")
    print("5. Summarize the main concepts of the document.\n")

    while True:
        try:
            question = input("Ask a question: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a valid question.\n")
            continue

        answer, context_docs = answer_question(
            question,
            collection,
            embedding_model,
            tokenizer,
            model
        )

        print("\nTop Retrieved Chunks:\n")
        for i, doc in enumerate(context_docs, start=1):
            print(f"Chunk {i}:")
            print(doc[:400])
            print("-" * 60)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()