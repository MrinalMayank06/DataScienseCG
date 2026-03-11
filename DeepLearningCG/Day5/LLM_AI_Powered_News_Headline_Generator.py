# Scenario: AI‑Powered News Headline Generator
# A digital media company wants to help journalists brainstorm headlines and opening lines for articles.
# - The system asks the journalist to enter a prompt (e.g., “Enter a prompt: The future of Artificial Intelligence”).
# - The GPT‑2 model then generates a continuation of the text, up to 50 tokens.
# - Example output:
# “The future of Artificial Intelligence will reshape industries, redefine creativity, and challenge
#  our understanding of human potential.”


#  AI-Powered News Headline Generator
# A digital media platform uses a generative language model to assist journalists
# in brainstorming headlines and opening lines for articles. The journalist enters
# a prompt such as "The future of Artificial Intelligence". The system then uses
# GPT-2 to expand the idea into a possible headline or introduction, generating
# up to 50 tokens of text that continue the journalist's prompt.

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = input("Enter a prompt: ")

headline = generator(
    prompt,
    max_length=50,
    temperature=0.6,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)

print("\nGenerated Headline / Opening Line:\n")
print(headline[0]["generated_text"])