# Scenario-Based Assessment
# Case Study: AI Assistant for a University Helpdesk
# A university wants to build an AI chatbot to answer student queries
# related to admissions, fees, hostel facilities, course schedules,
# and exam results. The goal is to reduce response time, lower cost,
# and provide more consistent answers using Generative AI and LLMs.

# UNIVERSITY AI HELPDESK CHATBOT
# Demonstrates:
# Generative AI
# Large Language Models
# Tokenization
# Transformer Inference
# Prompt Engineering

# STEP 3: Define University Knowledge Base
# # ------------------------------------------
# knowledge_base = """
# University Helpdesk Information

# Hostel Admission Requirements:
# - Admission confirmation letter
# - Government ID proof
# - Two passport size photographs

# Course Registration:
# - Registration happens online through the university portal
# - Students must register before the semester deadline

# Fee Payment:
# - Fees can be paid through the student dashboard
# - Online payment methods include debit card, credit card, and net banking

# Examination Schedule:
# - Semester exams usually start in December and May

# Library Access:
# - Students must carry their university ID card
# """
 

from transformers import pipeline

chatbot = pipeline("text-generation", model="gpt2")

knowledge_base = """
University Helpdesk Information

Hostel Admission Requirements:
- Admission confirmation letter
- Government ID proof
- Two passport size photographs

Course Registration:
- Registration happens online through the university portal
- Students must register before the semester deadline

Fee Payment:
- Fees can be paid through the student dashboard
- Online payment methods include debit card, credit card, and net banking

Examination Schedule:
- Semester exams usually start in December and May

Library Access:
- Students must carry their university ID card
"""

student_query = "What documents are required for hostel admission?"

prompt = f"""
You are a helpful university helpdesk assistant.
Use the university knowledge base below to answer the student clearly,
politely, and in short simple language. Only use the given information.

Knowledge Base:
{knowledge_base}

Student Question: {student_query}

Helpdesk Answer:
"""

response = chatbot(
    prompt,
    max_length=180,
    temperature=0.4,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)

generated_text = response[0]["generated_text"]
final_answer = generated_text.split("Helpdesk Answer:")[-1].strip()

print("Student Query:\n")
print(student_query)

print("\nKnowledge Base:\n")
print(knowledge_base)

print("\nGenerated Helpdesk Answer:\n")
print(final_answer)