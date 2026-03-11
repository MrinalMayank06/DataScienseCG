#  Scenario: AI‑Powered Email Assistant
# You’re building an email drafting assistant for a corporate team.
# - When a user starts typing:
# “Artificial Intelligence will…”
# - The assistant uses GPT‑2 to auto‑complete the sentence with a coherent continuation.
# - Example output:
# “Artificial Intelligence will transform the way businesses interact with customers,
# enabling faster decisions and personalized experiences.”

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

print(generator("Artificial Intelligence will", max_length=30))