# pip install chromadb sentence-transformers pypdf transformers torch gradio

import os
import re
import torch
import gradio as gr
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


PDF_PATH = r"D:\CG\datascience\DeepLearningCG\Day17\Legal_Research_Assistant_for_a_Corporate_Compliance_Team\data_privacy_regulation.pdf"

CHROMA_PATH = "./legal_rag_db"
COLLECTION_NAME = "legal_compliance_collection"


def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=700, overlap=120):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def create_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    return collection


def store_chunks(collection, chunks, embedding_model):
    for i, chunk in enumerate(chunks):

        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding]
        )


def retrieve(collection, embedding_model, query, k=3):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]


def build_prompt(context, query):

    prompt = f"""
You are a legal research assistant.

Answer using ONLY the context.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt


def generate_answer(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


def answer_question(query):

    docs = retrieve(collection, embedding_model, query)

    context = " ".join(docs)

    prompt = build_prompt(context, query)

    answer = generate_answer(prompt)

    return answer


print("Loading PDF...")

text = load_pdf_text(PDF_PATH)

text = clean_text(text)

chunks = chunk_text(text)

print("Chunks:", len(chunks))


print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


print("Creating vector DB...")

collection = create_collection()

store_chunks(collection, chunks, embedding_model)


print("Loading LLM...")

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-base"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
)


print("System ready")


def gradio_chat(question):
    return answer_question(question)


demo = gr.Interface(
    fn=gradio_chat,
    inputs=gr.Textbox(label="Ask legal question"),
    outputs=gr.Textbox(label="Answer"),
    title="Legal Research Assistant"
)

demo.launch()