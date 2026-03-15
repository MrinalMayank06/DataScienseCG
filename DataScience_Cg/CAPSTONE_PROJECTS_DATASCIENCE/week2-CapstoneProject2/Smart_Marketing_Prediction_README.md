# Smart Marketing Prediction System

This project demonstrates how machine learning can be used to improve
marketing decision making in an e-commerce environment.

The system predicts whether a customer browsing an online store is
likely to purchase a product. By identifying potential buyers, companies
can optimize marketing campaigns, target high-value customers, and
reduce unnecessary advertising costs.

The project implements a complete machine learning pipeline including
preprocessing, feature encoding, scaling, model training, evaluation,
and prediction.

------------------------------------------------------------------------

# Business Scenario

An e-commerce company named ShopEasy receives thousands of visitors
every day. Many users browse products but leave without purchasing
anything.

The marketing team currently spends a large budget on advertisements,
promotions, and discount campaigns without knowing which customers are
most likely to convert.

This project aims to build a predictive system that estimates purchase
probability based on user behavior.

If the model predicts a high probability of purchase, the system can
recommend targeted offers and promotions.

If the probability is low, marketing resources can be allocated
elsewhere.

------------------------------------------------------------------------

# Technologies Used

Python

Pandas

Scikit-learn

Matplotlib

Machine Learning Pipeline

Random Forest Classifier

------------------------------------------------------------------------

# Dataset

The dataset contains customer browsing behavior information including:

Age

Gender

Device type

Traffic source

Time spent on website

Pages visited

Ad clicks

Previous purchases

The target variable indicates whether the customer completed a purchase.

------------------------------------------------------------------------

# Machine Learning Workflow

Load dataset

Clean and preprocess data

Encode categorical variables

Split data into training and testing sets

Train machine learning model

Perform hyperparameter tuning

Evaluate model performance

Generate predictions

------------------------------------------------------------------------

# How to Run the Project

Step 1: Install required libraries

pip install pandas scikit-learn matplotlib

Step 2: Place dataset file in the project folder

DatasetCapstoneProject3.xlsx

Step 3: Run the script

python Smart_Marketing_Prediction_System_Capstone_Project_3.py

The program will automatically perform data preprocessing, train the
model, and display evaluation results.

------------------------------------------------------------------------

# Model Evaluation

The project evaluates the model using:

Accuracy score

Classification report

Confusion matrix visualization

Feature importance visualization

------------------------------------------------------------------------

# Learning Outcomes

This project demonstrates:

End-to-end machine learning pipeline design

Feature engineering and encoding

Model evaluation techniques

Hyperparameter tuning using GridSearchCV

Predictive analytics for business decision making
