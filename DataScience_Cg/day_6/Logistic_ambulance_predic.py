#step1 import libs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#database
df = pd.read_csv(r"C:\Users\krish\Downloads\BMI_dataset - Sheet1.csv")

print(df.head())
print(df.columns)

X = df[['BMI']]
y = df['Diabetes']

 

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=50, stratify=y)

 
model = LogisticRegression()
model.fit(X_train,y_train)

 
y_pred = model.predict(X_test)

 
print("accuracy",accuracy_score(y_test,y_pred))

 
plt.scatter(X['BMI'], y, color="black", label="Data points")
x_range = np.linspace(15,55,200).reshape(-1,1)
y_prob = model.predict_proba(x_range)[:,1]
plt.plot(x_range,y_prob,color="red",label ="BMI")
plt.xlabel("BMI")
plt.ylabel("Probability of Diabetes")
plt.legend()
plt.show()



