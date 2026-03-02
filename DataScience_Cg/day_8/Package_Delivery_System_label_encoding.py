# Scenario: Package Delivery System
# A logistics company wants to build a machine learning model to optimize package delivery.
# They collect data about each package, including:
# - Size: Small, Medium, Large
# - Priority: Low, Medium, High
# Since machine learning models work with numbers (not text labels), the company decides to use Label
#  Encoding to convert these categories into numeric values.
# They apply Scikit-learn’s LabelEncoder to both features:
# - Size → converted into numeric codes (e.g., Small = 2, Medium = 1, Large = 0)
# - Priority → converted into numeric codes (e.g., Low = 1, Medium = 2, High = 0)
# They also check the mapping and decode the numbers back to the original labels to ensure correctness.

# Questions for Learners
# Part A: Why does the company need to convert categorical values like Small, Medium, Large into numbers 
# before training a model?
# Part B: If the encoded values are 0, 1, 2, does the order (e.g., Large = 0, Medium = 1, Small = 2) 
# matter for all models? Why or why not?
# Part C: How can the company decode the numeric values back into the original labels after prediction?
# Part D (Applied): Suppose the company adds a new category “Extra Large.” How would LabelEncoder handle 
# this, and what challenge might arise if the model was already trained?


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Sample Data
df = pd.DataFrame({
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small'],
    'Priority': ['Low', 'High', 'Medium', 'Low', 'High']
})

# Step 2: Create Encoders
le_size = LabelEncoder()
le_priority = LabelEncoder()

# Encode columns
df['Size_Encoded'] = le_size.fit_transform(df['Size'])
df['Priority_Encoded'] = le_priority.fit_transform(df['Priority'])

# Step 3: View the encoding mapping
print("Size mapping:", dict(zip(le_size.classes_,
                               le_size.transform(le_size.classes_))))

print("Priority mapping:", dict(zip(le_priority.classes_,
                                    le_priority.transform(le_priority.classes_))))

# Step 4: Inverse transform (decode back)
decoded_size = le_size.inverse_transform([0, 1, 2])
print("Decoded Size:", decoded_size)

# Step 5: Final Encoded Dataset
print("\nFinal Encoded Dataset:\n")
print(df[['Size_Encoded', 'Priority_Encoded']])