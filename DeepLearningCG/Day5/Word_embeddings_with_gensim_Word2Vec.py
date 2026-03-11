# Word embeddings with gensim Word2Vec

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from numpy import dot
from numpy.linalg import norm

nltk.download("punkt", quiet=True)

corpus = """
Manufacturing relies on predictive maintenance and supply chain optimization.
Data engineers build pipelines, while analysts monitor KPIs and anomalies.
Robotics and IoT sensors stream telemetry to cloud databases for real-time insights.
Quality control uses computer vision to detect defects on the shop floor.
"""

print("Starting tokenization...")

sentences = [word_tokenize(s.lower()) for s in sent_tokenize(corpus)]

print("\nTokenized Sentences:")
for s in sentences:
    print(s)

print("\nTraining Word2Vec model...")

model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=1,
    workers=1,
    sg=1
)

print("\nModel trained successfully.")

for target in ["maintenance", "supply", "quality", "telemetry"]:
    print(f"\nTop similar to '{target}':")
    try:
        for w, score in model.wv.most_similar(target, topn=5):
            print(f"{w:15} -> {score:.3f}")
    except KeyError:
        print("Word not in vocabulary.")

def cosine(u, v):
    return dot(u, v) / (norm(u) * norm(v))

pairs = [
    ("maintenance", "telemetry"),
    ("quality", "defects"),
    ("supply", "optimization")
]

print("\nCosine similarities:")
for a, b in pairs:
    try:
        sim = cosine(model.wv[a], model.wv[b])
        print(f"{a:12} ~ {b:12} -> {sim:.3f}")
    except KeyError:
        print(f"Missing word: {a} or {b}")

input("\nexit")