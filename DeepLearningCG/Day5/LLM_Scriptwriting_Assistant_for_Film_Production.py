# Scenario: Scriptwriting Assistant for Film Production
# A film studio experiments with AI to assist screenwriters in brainstorming
# narrative dialogue and speculative plot directions. The writer provides the
# prompt "The future of Artificial Intelligence". The system generates two
# stylistic continuations by adjusting the temperature parameter. A lower
# temperature produces structured and documentary-style narration, while a
# higher temperature encourages imaginative storytelling suitable for creative
# scripts or science-fiction dialogue.

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The future of Artificial Intelligence"

formal_version = generator(
    prompt,
    max_length=60,
    temperature=0.2,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)

creative_version = generator(
    prompt,
    max_length=60,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    num_return_sequences=1
)

print("Prompt:", prompt)
print("\nDocumentary Style Output:\n")
print(formal_version[0]["generated_text"])
print("\nCreative Sci-Fi Style Output:\n")
print(creative_version[0]["generated_text"])