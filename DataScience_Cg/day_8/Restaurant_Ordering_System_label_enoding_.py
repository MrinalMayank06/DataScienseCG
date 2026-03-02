import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample Data
df = pd.DataFrame({
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small'],
    'Priority': ['Low', 'High', 'Medium', 'Low', 'High']
})

# Create LabelEncoders
le_size = LabelEncoder()
le_priority = LabelEncoder()

# Encode columns
df['Size_Encoded'] = le_size.fit_transform(df['Size'])
df['Priority_Encoded'] = le_priority.fit_transform(df['Priority'])

# Mapping in readable list/dict
size_mapping = {label: int(code) for label, code in zip(le_size.classes_, le_size.transform(le_size.classes_))}
priority_mapping = {label: int(code) for label, code in zip(le_priority.classes_, le_priority.transform(le_priority.classes_))}

# Print mapping as list
print("Size Mapping:", size_mapping)
print("Priority Mapping:", priority_mapping)

# Inverse transform (decoded)
decoded_size = le_size.inverse_transform([0, 1, 2])
decoded_priority = le_priority.inverse_transform([0, 1, 2])
print("\nDecoded Size:", list(decoded_size))
print("Decoded Priority:", list(decoded_priority))

# Final Encoded Dataset (table view)
print("\nFinal Encoded Dataset:\n")
print(df[['Size_Encoded', 'Priority_Encoded']].to_string(index=False))