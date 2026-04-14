import os, requests
from dotenv import load_dotenv

load_dotenv()

def chatbot(student_id, query):
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview"

    headers = {
        "Content-Type": "application/json",
        "api-key": key
    }

    prompt = f"Student ID: {student_id}\nQuery: {query}\nAnswer properly."

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    res = requests.post(url, headers=headers, json=data)
    return f"[Student ID: {student_id}] " + res.json()["choices"][0]["message"]["content"]
