# Code Demo (Trigram counts)
from collections import defaultdict

corpus = "the cat sat on the mat the cat lay on the rug the dog barked loudly"
tokens = corpus.split()

# Build trigram counts
trigram_counts = defaultdict(lambda: defaultdict(int))
for i in range(len(tokens)-2):
    context = (tokens[i], tokens[i+1])
    next_word = tokens[i+2]
    trigram_counts[context][next_word] += 1

# Predict next word after "the cat"
context = ("the", "cat")
print("Next word predictions for context:", context)
for word, count in trigram_counts[context].items():
    print(f"{word} -> {count}")