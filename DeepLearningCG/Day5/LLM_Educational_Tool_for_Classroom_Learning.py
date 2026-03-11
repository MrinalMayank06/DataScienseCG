# Scenario: Educational Tool for Classroom Learning
# A high school teacher is introducing students to AI‑powered writing assistants.
# - The teacher wants to show how different models produce different styles of text.
# - They use the same prompt:
# “Artificial Intelligence will transform”
# - The system generates two outputs:
# - GPT‑2 Output
# More elaborate continuation, e.g.:
# “Artificial Intelligence will transform the way societies function, influencing education, healthcare, and the future of human creativity.”
# → Great for showing students how AI can generate rich, detailed ideas.
# - DistilGPT‑2 Output
# Shorter, faster continuation, e.g.:
# “Artificial Intelligence will transform our daily lives.”
# → Useful for demonstrating concise summaries.


#  Educational Tool for Classroom Learning
# A teacher demonstrates how different language models generate text
# using the same prompt to show variation in style and detail.
# The prompt "Artificial Intelligence will transform" is given to
# both GPT-2 and DistilGPT-2 so students can observe how the larger
# model tends to produce richer text while the distilled model
# produces faster and more concise continuations.

from transformers import pipeline

gpt2_generator = pipeline("text-generation", model="gpt2")
distil_generator = pipeline("text-generation", model="distilgpt2")

prompt = "Artificial Intelligence will transform"

gpt2_text = gpt2_generator(
    prompt,
    max_length=60,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)

distil_text = distil_generator(
    prompt,
    max_length=40,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)

print("Prompt:", prompt)

print("\nGPT-2 Output:\n")
print(gpt2_text[0]["generated_text"])

print("\nDistilGPT-2 Output:\n")
print(distil_text[0]["generated_text"])