# Smart Traffic Sign Recognition System

This project demonstrates the basic principles of neural networks by
implementing a perceptron model that classifies traffic signs.

The system is designed to recognize whether a traffic sign represents a
STOP sign or not based on extracted features.

The goal is to understand how neural networks learn decision boundaries
through weight updates during training.

------------------------------------------------------------------------

# Project Scenario

A city plans to build a smart traffic monitoring system using cameras
installed at road intersections.

These cameras must detect traffic signs automatically in order to
support autonomous vehicles and traffic management systems.

Before implementing a full deep learning model, engineers first
implement a simple perceptron model to understand the fundamentals of
neural networks.

------------------------------------------------------------------------

# Technologies Used

Python

NumPy

Pandas

Matplotlib

Perceptron Neural Network

------------------------------------------------------------------------

# Dataset

The dataset contains numerical features extracted from traffic sign
images such as:

Color intensity

Shape characteristics

Text presence

Edge count

The target variable indicates whether the sign represents a STOP sign.

------------------------------------------------------------------------

# Neural Network Workflow

Load dataset

Separate features and labels

Initialize weights and bias

Train perceptron using gradient updates

Generate predictions

Evaluate accuracy

Visualize predictions

------------------------------------------------------------------------

# Running the Project

Step 1: Install required libraries

pip install numpy pandas matplotlib

Step 2: Place dataset file in the project folder

DatasetCapstoneProject4.csv

Step 3: Run the script

python Smart_Traffic_Sign_Recognition_System_Capstone_Project_4.py

The program will train the perceptron model and display prediction
results.

------------------------------------------------------------------------

# Model Training

The perceptron learns by adjusting weights using a learning rate and
error feedback.

For each training sample:

Prediction is generated using the step activation function.

Error is calculated.

Weights and bias are updated accordingly.

This iterative process allows the model to learn classification
patterns.

------------------------------------------------------------------------

# Learning Outcomes

This project helps understand:

Fundamentals of neural networks

Perceptron training mechanism

Linear classification models

Visualization of prediction performance
