# Smart Waste Classification System

This project demonstrates how deep learning can be used to automatically
classify waste images into different categories. The objective of the
system is to help build intelligent waste segregation tools that can
support smart city initiatives.

The program trains an image classification model that can recognize
different types of waste and assign them to the correct category.

The project also demonstrates the use of data preprocessing,
convolutional neural networks, transfer learning, and model evaluation
techniques.



# Problem Statement

Cities generate massive amounts of waste every day. Sorting waste
manually is inefficient and error-prone.

An automated system that can identify waste types from images can help
improve recycling efficiency and reduce environmental impact.

The goal of this project is to develop a machine learning model that
classifies waste images into the following categories:

Recyclable\
Organic\
Non-Recyclable

This is a multi-class image classification problem.



# Dataset

The dataset used in this project is available from Kaggle.

Garbage Classification Dataset

Images are organized into folders based on their category. Each folder
represents a class label.

Example structure:

dataset/

train/

recyclable/

organic/

non_recyclable/

validation/

recyclable/

organic/

non_recyclable/



# Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Seaborn

Scikit-learn



# Project File

Main script:

Smart_Waste_Classification_System.py

The script contains the entire pipeline including data loading,
preprocessing, training, evaluation, and visualization.



# System Workflow

The system follows a typical deep learning pipeline.

Dataset collection\
Data preprocessing\
CNN model training\
Model evaluation\
Prediction testing\
Transfer learning comparison

Each step is implemented sequentially in the script.



# Step-by-Step Setup Guide

This section explains how someone can run the project from scratch.

Step 1 -- Install Python

Ensure Python 3.9 or newer is installed on your system.

Check installation using:

python --version



Step 2 -- Install Required Libraries

Open a terminal and install the required dependencies.

pip install tensorflow numpy matplotlib seaborn scikit-learn

These libraries are used for model training, visualization, and
evaluation.



Step 3 -- Download the Dataset

Download the dataset from Kaggle.

Garbage Classification Dataset

Extract the dataset and store it in a folder on your system.

Example:

C:/datasets/garbage-classification/



Step 4 -- Update Dataset Path

Inside the script you will find the dataset path variable.

Update the path so that it points to the dataset location on your
computer.

Example:

dataset_path = "C:/datasets/garbage-classification"



Step 5 -- Run the Script

Navigate to the project folder in the terminal and run the program.

python Smart_Waste_Classification_System.py

The program will start loading the dataset and training the model.



# Data Preprocessing

Before training the model, the images are preprocessed.

Images are resized to 224 x 224 pixels.

Pixel values are normalized to improve training stability.

Data augmentation techniques such as rotation, flipping, zooming, and
brightness adjustment are applied to increase dataset diversity.



# CNN Model

A custom convolutional neural network is used.

The model consists of multiple layers including:

Convolution layers for feature extraction

Pooling layers for dimensionality reduction

Dense layers for classification

Dropout layers to reduce overfitting

The final layer uses a softmax activation function to produce class
probabilities.



# Training Visualization

During training the model records performance metrics.

The program generates graphs for:

Training Accuracy vs Validation Accuracy

Training Loss vs Validation Loss

These plots help analyze how the model learns over time.



# Model Evaluation

After training the model, predictions are generated for validation data.

A confusion matrix is created to visualize prediction performance across
classes.

The overall accuracy score is also calculated.



# Sample Prediction

The program randomly selects an image from the dataset and predicts its
class.

The selected image is displayed along with the predicted label.

This allows visual verification of model predictions.



# Transfer Learning

In addition to the custom CNN, the project also demonstrates transfer
learning.

A pretrained MobileNetV2 model is used as the base network.

The base layers are frozen and new classification layers are added.

This approach often improves accuracy because the pretrained model
already understands general image features.



# Model Comparison

Finally the project compares the performance of:

Custom CNN model

Transfer Learning model

This comparison helps understand the benefits of using pretrained
architectures.



# Possible Improvements

The project can be expanded further in several ways.

Training with more epochs

Using larger datasets

Adding additional waste categories

Deploying the model as a web application

Integrating the system with IoT-based smart bins



# Learning Outcomes

This project helps understand:

Image preprocessing techniques

CNN architecture design

Model training and evaluation

Transfer learning concepts

Visualization of model performance
