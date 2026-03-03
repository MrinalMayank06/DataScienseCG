# 📖 Scenario: Retail Sales Data Encoding
# Imagine you are working as a data analyst for a retail company. The company wants
# to build a machine learning model to predict sales performance based on product type
# and region. However, the dataset contains categorical variables (like Product and Region)
# that machine learning algorithms cannot directly process — they need numerical inputs.


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample dataset
data = pd.DataFrame({
    'Product': ['Shoes', 'Shirt', 'Shoes', 'Watch'],
    'Region': ['North', 'South', 'East', 'West'],
    'Sales': [200, 150, 300, 400]
})

# Label Encoding
le = LabelEncoder()
data['Product_encoded'] = le.fit_transform(data['Product'])

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False)
region_encoded = ohe.fit_transform(data[['Region']])

region_df = pd.DataFrame(
    region_encoded,
    columns=ohe.get_feature_names_out(['Region'])
)

# Combine with original data
data = pd.concat([data, region_df], axis=1)

print(data)