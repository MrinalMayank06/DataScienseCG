# Scenario: Patient Health Monitoring
# A hospital wants to analyze patient records to understand how disease severity and recovery 
# satisfaction affect treatment outcomes.
# They collect data such as:
# - Disease Severity: Mild, Moderate, Severe, Critical
# - Recovery Satisfaction: Poor, Average, Good, Excellent
# Since these categories have a natural order (e.g., Critical is worse than Mild, Excellent is 
# better than Poor), the hospital uses Ordinal Encoding to convert them into numbers that respect this
#  ranking.
# They define custom ordering:
# - Disease Severity → Mild (0), Moderate (1), Severe (2), Critical (3)
# - Recovery Satisfaction → Poor (0), Average (1), Good (2), Excellent (3)
# They also compare this with a manual dictionary mapping to ensure consistency.


import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Step 1: Sample patient data
data = pd.DataFrame({
    'Disease_Severity': ['Mild', 'Severe', 'Moderate', 'Critical', 'Mild'],
    'Recovery_Satisfaction': ['Good', 'Excellent', 'Poor', 'Average', 'Good']
})

# Step 2: Define custom ordering
severity_order = [['Mild', 'Moderate', 'Severe', 'Critical']]
satisfaction_order = [['Poor', 'Average', 'Good', 'Excellent']]

# Step 3: Use OrdinalEncoder for custom ordering
sev_encoder = OrdinalEncoder(categories=severity_order)
rec_encoder = OrdinalEncoder(categories=satisfaction_order)

data['Severity_Encoded'] = sev_encoder.fit_transform(data[['Disease_Severity']])
data['Satisfaction_Encoded'] = rec_encoder.fit_transform(data[['Recovery_Satisfaction']])

# Step 4: Manual mapping for verification
severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2, 'Critical': 3}
satisfaction_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}

data['Severity_Manual'] = data['Disease_Severity'].map(severity_map)
data['Satisfaction_Manual'] = data['Recovery_Satisfaction'].map(satisfaction_map)

# Step 5: Display final encoded DataFrame
print("\nEncoded Patient DataFrame:")
print(data)