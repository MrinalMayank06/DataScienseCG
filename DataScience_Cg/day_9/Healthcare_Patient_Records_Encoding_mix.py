#  Scenario: Healthcare Patient Records Encoding
# Imagine you are working as a data scientist in a hospital. The hospital wants to build a machine
#  learning model to predict patient recovery time based on demographic and treatment details.
#   The dataset contains categorical variables (like Treatment Type and Hospital Wing) that must be converted into numeric form before modeling.
# Business Context
# - Treatment Type: Different medical procedures (Surgery, Therapy, Medication).
# - Hospital Wing: Location where the patient is admitted (East, West, North, South).
# - Recovery Days: Numeric values representing how many days the patient took to recover.
# The challenge:
# Machine learning models cannot directly interpret text categories, so you need to encode categorical
#  features into numbers.


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample dataset
data = pd.DataFrame({
    'Treatment_Type': ['Surgery', 'Therapy', 'Medication', 'Surgery'],
    'Hospital_Wing': ['East', 'West', 'North', 'South'],
    'Recovery_Days': [10, 20, 7, 15]
})

 
le = LabelEncoder()
data['Treatment_encoded'] = le.fit_transform(data['Treatment_Type'])

 
ohe = OneHotEncoder(sparse_output=False)
wing_encoded = ohe.fit_transform(data[['Hospital_Wing']])

wing_df = pd.DataFrame(
    wing_encoded,
    columns=ohe.get_feature_names_out(['Hospital_Wing'])
)

 
data = pd.concat([data, wing_df], axis=1)

print(data)