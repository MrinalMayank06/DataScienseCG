from transformers import pipeline, AutoTokenizer

# Task 1 – LLM Interaction Setup


print("\nLoading AI Model... Please wait\n")

generator = pipeline("text-generation", model="gpt2")

print("Model Loaded Successfully!\n")

# Sample Research Article
article = """
Artificial Intelligence (AI) has become one of the most transformative technologies of the 21st century. 
It is widely used in industries such as healthcare, finance, education, and manufacturing. 
Machine learning, a subset of AI, allows computers to learn from data and improve performance without explicit programming.

Recent advances in deep learning have significantly improved capabilities in image recognition, natural language processing, 
and autonomous systems. For example, AI-powered diagnostic tools can detect diseases from medical images with high accuracy.

Despite its benefits, AI also presents challenges such as data privacy concerns, ethical issues, and job displacement. 
Researchers are working to create responsible AI systems that are transparent, fair, and accountable.

In the future, AI is expected to assist humans in solving complex global problems such as climate change, 
healthcare accessibility, and smart city development.
"""

print("TASK 1: ARTICLE SUMMARY")

prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"

response = generator(prompt, max_length=200)

print(response[0]['generated_text'])

# Task 2 – Prompt Engineering Experiments

print(" TASK 2: ZERO-SHOT PROMPT ")

zero_shot_prompt = f"""
Summarize the following article in 5 bullet points.

Article:
{article}
"""

response = generator(zero_shot_prompt, max_length=200)

print(response[0]['generated_text'])


print("TASK 2: FEW-SHOT PROMPT ")

few_shot_prompt = f"""
Example 1
Article: Climate change is affecting global temperatures and weather patterns.
Summary:
• Global temperatures are rising
• Climate change affects weather patterns
• Reducing emissions is necessary

Example 2
Article: Online education allows students to learn remotely using digital platforms.
Summary:
• Online learning enables remote education
• Digital platforms increase accessibility
• Technology improves learning flexibility

Now summarize the following article.

Article:
{article}

Summary:
"""

response = generator(few_shot_prompt, max_length=250)

print(response[0]['generated_text'])


print("TASK 2: CHAIN OF THOUGHT PROMPT ")

cot_prompt = f"""
Analyze the article step by step.

Step 1: Identify the main topic
Step 2: Extract key ideas
Step 3: Generate a concise summary

Article:
{article}

Answer:
"""

response = generator(cot_prompt, max_length=250)

print(response[0]['generated_text'])

# Task 3 – Prompt Optimization


print(" TASK 3: OPTIMIZED PROMPT")

optimized_prompt = f"""
You are an academic research assistant.

Analyze the following article and produce:

1. Executive Summary
2. Three Key Insights
3. One Actionable Takeaway

Use a professional tone.

Article:
{article}

Response Format:

Executive Summary:
Key Insights:
Actionable Takeaway:
"""

response = generator(optimized_prompt, max_length=250)

print(response[0]['generated_text'])

# Test optimized prompt on second article

article2 = """
Blockchain technology is a decentralized digital ledger used to record transactions securely.
It removes the need for intermediaries and ensures transparency and trust in digital systems.
Blockchain is widely used in cryptocurrencies, supply chain management, and digital identity verification.
"""

print("TASK 3: TEST ON SECOND ARTICLE ")

test_prompt = optimized_prompt.replace(article, article2)

response = generator(test_prompt, max_length=200)

print(response[0]['generated_text'])

# Task 4 – Tokenization Experiment


print(" TASK 4: TOKENIZATION EXPERIMENT ")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

paragraph = """
Artificial Intelligence enables machines to perform tasks that normally require human intelligence.
"""

tokens = tokenizer.tokenize(paragraph)
token_ids = tokenizer.encode(paragraph)

print("Tokens:\n")
print(tokens)

print("\nToken IDs:\n")
print(token_ids)

print("\nTotal Number of Tokens:", len(token_ids))

# Task 5 – Mini AI Research Assistant Tool

print("TASK 5: MINI AI RESEARCH ASSISTANT")

def research_assistant(text):

    prompt = f"""
    You are an AI research assistant.

    From the following article generate:

    1. Short Summary
    2. Three Key Insights
    3. One Actionable Recommendation

    Article:
    {text}

    Response:
    """

    result = generator(prompt, max_length=250)

    return result[0]['generated_text']


# Accept user input
user_article = input("\nPaste a research article here:\n")

output = research_assistant(user_article)

print("AI ASSISTANT OUTPUT ")
print(output)
