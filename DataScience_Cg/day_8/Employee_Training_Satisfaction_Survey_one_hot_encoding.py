# Scenario: Employee Training & Satisfaction Survey
# A company conducts a survey to understand how employee education level and job satisfaction affect performance.
# They collect data such as:
# - Education: High School, Bachelor, Master, PhD
# - Satisfaction: Poor, Average, Good, Excellent
# Since these categories have a natural order (e.g., PhD is higher than Bachelor, Excellent is better
#  than Good), the company decides to use Ordinal Encoding to convert them into numbers that respect 
#  this ranking.
# They define custom ordering:
# - Education → High School (0), Bachelor (1), Master (2), PhD (3)
# - Satisfaction → Poor (0), Average (1), Good (2), Excellent (3)
# They also compare this with a manual dictionary mapping to ensure consistency.

# Questions for Learners
# Part A: Why is it important to use Ordinal Encoding instead of simple Label Encoding for ordered 
# categories like education level?
# Part B: If the company encoded “PhD = 0” and “High School = 3,” what problem might arise in 
# interpreting the model?
# Part C: How does Ordinal Encoding differ from One-Hot Encoding in representing categorical data?
# Part D (Applied): Suppose the company adds a new satisfaction level “Outstanding.” How should they 
# update their encoding scheme to keep the order meaningful?







import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Step 1: Sample survey data
data = pd.DataFrame({
    'Education': ['Bachelor', 'PhD', 'Master', 'High School'],
    'Satisfaction': ['Good', 'Excellent', 'Poor', 'Average']
})

# Step 2: Define custom ordering
education_order = [['High School', 'Bachelor', 'Master', 'PhD']]
satisfaction_order = [['Poor', 'Average', 'Good', 'Excellent']]

# Step 3: Use OrdinalEncoder for custom ordering
edu_encoder = OrdinalEncoder(categories=education_order)
sat_encoder = OrdinalEncoder(categories=satisfaction_order)

data['Education_Encoded'] = edu_encoder.fit_transform(data[['Education']])
data['Satisfaction_Encoded'] = sat_encoder.fit_transform(data[['Satisfaction']])

# Step 4: Alternative manual mapping with dictionaries
education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
satisfaction_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}

data['Education_Manual'] = data['Education'].map(education_map)
data['Satisfaction_Manual'] = data['Satisfaction'].map(satisfaction_map)

# Step 5: Display final encoded DataFrame
print("\nEncoded Survey DataFrame:")
print(data)



