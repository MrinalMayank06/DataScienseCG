# Scenario: Healthcare Chatbot – Patient Symptom Matching
# Imagine you’re designing a healthcare chatbot that helps patients find
# possible conditions based on their symptoms.
#
# Both patients and conditions are represented as vectors of features:
# [fever, cough, fatigue, headache]
#
# Patient’s symptom vector:
# [0.2, 0.8, -0.3, 0.5]
# → mild fever, strong cough, no fatigue, moderate headache
#
# Condition’s symptom profile vector:
# [0.1, 0.9, -0.2, 0.4]
# → mild fever, strong cough, little fatigue, moderate headache
#
# Goal:
# Use cosine similarity to measure how closely the patient’s symptoms
# align with the condition’s profile.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Patient symptom vector
patient_vector = np.array([[0.2, 0.8, -0.3, 0.5]])

# Condition symptom profile vector
condition_vector = np.array([[0.1, 0.9, -0.2, 0.4]])

# Compute cosine similarity
similarity = cosine_similarity(patient_vector, condition_vector)

print("Patient Vector:", patient_vector)
print("Condition Vector:", condition_vector)
print("Cosine Similarity:", similarity[0][0])