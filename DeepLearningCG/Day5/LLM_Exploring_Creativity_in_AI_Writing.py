# Scenario: Exploring Creativity in AI Writing
# A university professor is running a workshop on creative writing with AI.
# - The professor gives students the prompt:
# “The future of Artificial Intelligence”
# - The system uses GPT‑2 to generate continuations of the sentence.
# To demonstrate how temperature affects creativity:
# - With temperature = 0.2 (low randomness), the output is more predictable and conservative, e.g.:
# “The future of Artificial Intelligence will bring advancements in healthcare, education, and business.”

#Explanation : of the scenario > usage
# Scenario: Exploring Creativity in AI Writing
# In a university AI-assisted creative writing workshop, students explore how
# generative language models transform a simple prompt into different narrative
# styles. The professor provides the prompt "The future of Artificial Intelligence"
# and demonstrates that adjusting the temperature parameter influences how
# conservative or imaginative the generated continuation becomes.

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The future of Artificial Intelligence"

low_temperature_output = generator(
    prompt,
    max_length=60,
    temperature=0.2,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)

high_temperature_output = generator(
    prompt,
    max_length=60,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    num_return_sequences=1
)

print("Prompt:", prompt)
print("\nLow Temperature Generation:\n")
print(low_temperature_output[0]["generated_text"])
print("\nHigh Temperature Generation:\n")
print(high_temperature_output[0]["generated_text"])