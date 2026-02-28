#step1 import libs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#database

X = np.array([[18],[20],[22],[25],[28],[30],[35],[40],[45],[50]])
y = np.array([1,1,0,0,1,1,1,0,0,1])

#split into the train and test set 

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#train logistic reg model
model = LogisticRegression()
model.fit(X_train,y_train)

#make prediction
y_pred = model.predict(X_test)

#accuracy 
print("accuracy",accuracy_score(y_test,y_pred))

#visualize the data 
plt.scatter(X,y,color="blue",label = "Data points")
x_range = np.linspace(15,55,200).reshape(-1,1)
y_prob = model.predict_proba(x_range)[:,1]
plt.plot(x_range,y_prob,color="red",label ="Logistic curve")
plt.xlabel("Driver age")
plt.ylabel("prob. of filling chain")
plt.legend()
plt.show()
