# Scenario: AI Model Evaluation in a Startup
# A startup is evaluating two generative models to power a marketing content assistant.
# The same prompt "Artificial Intelligence will transform" is given to both GPT-2
# and DistilGPT-2 to observe differences in expressiveness and response speed.
# GPT-2 typically generates richer and more detailed descriptions,
# while DistilGPT-2 is optimized for faster inference with lighter computation.

from transformers import pipeline

gpt2_generator = pipeline("text-generation", model="gpt2")
distilgpt2_generator = pipeline("text-generation", model="distilgpt2")

prompt = "Artificial Intelligence will transform"

gpt2_output = gpt2_generator(
    prompt,
    max_length=60,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)

distil_output = distilgpt2_generator(
    prompt,
    max_length=60,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)

print("Prompt:", prompt)

print("\nGPT-2 Generated Text:\n")
print(gpt2_output[0]["generated_text"])

print("\nDistilGPT-2 Generated Text:\n")
print(distil_output[0]["generated_text"])