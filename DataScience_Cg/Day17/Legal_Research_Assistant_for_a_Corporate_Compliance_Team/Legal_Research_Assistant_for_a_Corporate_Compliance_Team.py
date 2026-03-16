 # Scenario: Legal Research Assistant for a Corporate Compliance Team
# Context
# A corporate compliance department constantly reviews lengthy legal documents, regulatory filings, and policy updates. These documents are dense, full of
# legal terminology, and often hundreds of pages long. The team struggles to quickly extract relevant clauses or understand implications without spending hours reading.
# How the RAG Chatbot Fits In
# - Input Source: The team uploads a legal document (e.g., data_privacy_regulation.pdf).
# - Chunking: The chatbot splits the document into sections (clauses, articles, sub-sections) so no detail is overlooked.
# - Embeddings + Vector DB: Each section is converted into embeddings and stored in Chroma, enabling semantic search rather than keyword-only lookup.
# - Retriever: When someone asks, “What does this regulation say about cross-border data transfers?”, the retriever surfaces the most relevant clauses.
# - LLM Response: A Hugging Face model (e.g., Flan-T5) generates a concise, plain-language summary of those clauses, stripping away heavy legal jargon.
# - Chat Loop: The compliance team can continue asking questions interactively, like “Does this regulation conflict with GDPR?” or “What penalties are mentioned
#  for non-compliance?”.
# Outcome
# The chatbot acts as a legal research assistant, helping the compliance team quickly interpret complex documents, identify risks, and prepare summaries for executives
#  without needing to manually parse every page.

# explanation
# Scenario: Legal Research Assistant for a Corporate Compliance Team
# This project builds a simple RAG chatbot without LangChain.
# It reads a legal PDF, splits it into meaningful sections,
# creates embeddings, stores them in ChromaDB, retrieves relevant parts,
# and uses a Hugging Face model to answer questions in plain English.

# Install required libraries before running:
# pip install chromadb sentence-transformers pypdf transformers torch

# Scenario: Legal Research Assistant for a Corporate Compliance Team
# This project builds a simple RAG chatbot without LangChain.
# It reads a legal PDF, splits it into meaningful sections,
# creates embeddings, stores them in ChromaDB, retrieves relevant parts,
# and uses a Hugging Face model to answer questions in plain English.

# Install required libraries before running:
# pip install chromadb sentence-transformers pypdf transformers torch

import os
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


PDF_PATH = r"D:\CG\datascience\DeepLearningCG\Day17\Legal_Research_Assistant_for_a_Corporate_Compliance_Team\data_privacy_regulation.pdf"
CHROMA_PATH = "./legal_rag_db"
COLLECTION_NAME = "legal_compliance_collection"


def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            full_text += f"\n\n[Page {page_number}]\n{page_text}"

    return full_text


def clean_text(text):
    text = text.replace("\u25a0", "-")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_by_articles(text):
    pattern = r"(Article\s+\d+\s*:\s*.*?)(?=(Article\s+\d+\s*:)|$)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)

    article_chunks = []
    for match in matches:
        article_text = match[0].strip()
        if article_text:
            article_chunks.append(article_text)

    if article_chunks:
        return article_chunks

    return [text]


def split_long_chunk(text, chunk_size=700, overlap=120):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def prepare_chunks(text):
    article_chunks = split_by_articles(text)

    final_chunks = []
    metadata_list = []

    for article_index, article_text in enumerate(article_chunks):
        article_match = re.search(r"(Article\s+\d+)", article_text, flags=re.IGNORECASE)
        article_name = article_match.group(1) if article_match else f"Section_{article_index + 1}"

        subchunks = split_long_chunk(article_text, chunk_size=700, overlap=120)

        for sub_index, subchunk in enumerate(subchunks):
            final_chunks.append(subchunk)
            metadata_list.append(
                {
                    "article": article_name,
                    "chunk_number": sub_index + 1,
                    "source": os.path.basename(PDF_PATH)
                }
            )

    return final_chunks, metadata_list


def create_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)
    return client, collection


def store_chunks(collection, chunks, metadata_list, embedding_model):
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[metadata_list[i]]
        )


def retrieve_relevant_chunks(collection, embedding_model, query, k=3):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    distances = results.get("distances", [[]])
    if distances and len(distances[0]) > 0:
        distances = distances[0]
    else:
        distances = [None] * len(documents)

    return documents, metadatas, ids, distances


def build_prompt(context, query):
    prompt = f"""
You are a legal research assistant for a corporate compliance team.

Answer the user's question using ONLY the context below.

Rules:
1. Do not invent anything.
2. If the answer is not clearly present, say: Not found in document.
3. Explain in plain English.
4. Mention duties, restrictions, deadlines, safeguards, or penalties if they appear.
5. Keep the answer concise and accurate.

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


def answer_question(collection, embedding_model, tokenizer, model, query, k=3):
    docs, metas, ids, distances = retrieve_relevant_chunks(collection, embedding_model, query, k=k)

    context = "\n\n".join(docs)
    prompt = build_prompt(context, query)

    answer = generate_answer(tokenizer, model, prompt)

    return answer, docs, metas, ids, distances


def show_retrieved_context(docs, metas, distances):
    print("\nTop Retrieved Clauses / Sections:\n")

    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        print(f"Result {i}")
        print("Article:", meta.get("article", "Unknown"))
        print("Chunk Number:", meta.get("chunk_number", "Unknown"))
        print("Source:", meta.get("source", "Unknown"))
        print("Distance:", distance if distance is not None else "N/A")
        print("Text Preview:")
        print(doc[:500])
        print("-" * 80)


def main():
    if not os.path.exists(PDF_PATH):
        print("PDF file not found. Check PDF_PATH.")
        return

    print("Loading PDF...")
    raw_text = load_pdf_text(PDF_PATH)

    print("Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("Preparing chunks...")
    chunks, metadata_list = prepare_chunks(cleaned_text)

    print("Total chunks created:", len(chunks))

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Creating ChromaDB collection...")
    _, collection = create_collection()

    print("Storing chunks in vector database...")
    store_chunks(collection, chunks, metadata_list, embedding_model)

    print("Loading LLM...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    print("\n==============================================")
    print("Enhanced Legal RAG Chatbot is Ready")
    print("Type 'exit' to stop")
    print("==============================================\n")

    while True:
        try:
            query = input("Ask a legal/compliance question: ").strip()
        except KeyboardInterrupt:
            print("\nSession ended")
            break

        if query.lower() == "exit":
            print("Session ended")
            break

        if not query:
            print("Please enter a question.\n")
            continue

        answer, docs, metas, ids, distances = answer_question(
            collection=collection,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            model=model,
            query=query,
            k=3
        )

        show_retrieved_context(docs, metas, distances)

        print("\nFinal Answer:\n")
        print(answer)
        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()


# prompts
#     What does this regulation say about cross-border data transfers?

# What obligations does the regulation impose on companies that collect personal data?

# Are there any penalties mentioned for non-compliance with this regulation?

# Does the document mention any deadlines for reporting data breaches?

# What safeguards are required when processing sensitive personal data?

# Does the regulation require user consent before collecting personal data?

# What responsibilities do companies have regarding data security?

# Does the regulation specify how long personal data can be stored?

# Are there any restrictions on sharing personal data with third parties?

# Does the regulation mention any conflict or alignment with GDPR?

# What rights do individuals have over their personal data according to this document?

# What procedures must companies follow in case of a data breach?

# Are organizations required to appoint a data protection officer?

# Does the regulation define what qualifies as personal data?

# What compliance steps must organizations follow to avoid penalties?