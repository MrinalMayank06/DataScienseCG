# Scenario: Healthcare Research Insights
# A healthcare research team wants to study how medical terms are related
# inside reports and research papers. Word2Vec is used to learn word
# embeddings so similar healthcare concepts like diagnosis, treatment,
# symptoms, and medication can be compared numerically.

from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from numpy import dot
from numpy.linalg import norm

corpus = """
Doctors analyze symptoms to make an accurate diagnosis for each patient.
Treatment plans often include medication, therapy, and regular monitoring.
Hospitals store patient records to improve clinical decision support systems.
Researchers study disease patterns, symptoms, and treatments in medical reports.
Medication safety and patient care are important in healthcare operations.
Clinical teams use diagnostic tools to support treatment decisions.
"""

sentences = [word_tokenize(sentence.lower()) for sentence in sent_tokenize(corpus)]

model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=1,
    workers=1,
    sg=1
)

print("Tokenized Sentences:\n")
for sentence in sentences:
    print(sentence)

print("\nTop similar words:\n")
for target in ["diagnosis", "treatment", "symptoms", "patient", "medication"]:
    print(f"Top similar to '{target}':")
    for word, score in model.wv.most_similar(target, topn=5):
        print(f"{word:15} -> {score:.3f}")
    print()

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

pairs = [
    ("diagnosis", "symptoms"),
    ("treatment", "medication"),
    ("patient", "care"),
    ("clinical", "diagnostic")
]

print("Cosine similarities:\n")
for word1, word2 in pairs:
    similarity = cosine_similarity(model.wv[word1], model.wv[word2])
    print(f"{word1:12} ~ {word2:12} -> {similarity:.3f}")