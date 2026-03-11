#  Scenario: Game Storyline Generator
# You’re designing a fantasy role‑playing game and need quick ideas for quests and dialogue.
# - The developer types a prompt:
# “The hero enters the ancient forest and discovers…”
# - The GPT‑2 pipeline continues the story:
# “The hero enters the ancient forest and discovers a hidden village where the people whisper of
#  a cursed artifact buried beneath the old ruins.” :


# Game Storyline Generator
from transformers import pipeline
import random

generator = pipeline("text-generation", model="gpt2")

seed_prompts = [
"The hero enters the ancient forest and discovers",
"Deep beneath the ruined castle the warrior finds",
"At the edge of the forgotten kingdom a traveler hears",
"Inside the glowing cave the young mage uncovers",
"Beyond the silent mountains a mysterious voice whispers"
]

prompt = random.choice(seed_prompts)

story = generator(
prompt,
max_length=80,
num_return_sequences=1,
temperature=0.9,
top_k=50,
top_p=0.95,
do_sample=True
)

print(story[0]["generated_text"])

