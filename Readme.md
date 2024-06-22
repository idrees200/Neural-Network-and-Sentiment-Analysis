# Neural Network and Sentiment Analysis

## Overview

This repository contains a Python script that demonstrates the implementation of machine learning and deep learning techniques using popular libraries such as TensorFlow, Keras, Scikit-learn, and NLTK. The script covers tasks including data preprocessing, neural network modeling for binary classification (churn prediction), and an optional custom implementation for sentiment analysis using a basic neural network architecture.

## Features

### 1. Imports and Data Loading

- **Libraries:** Imports necessary Python libraries including pandas, numpy, TensorFlow, Keras, Scikit-learn, and NLTK.
- **Data Loading:** Loads two datasets from CSV files (`Churn_Modelling.csv` for churn prediction data and `reddit-comments-2015-08.csv` for sentiment analysis data) using pandas.

### 2. Data Preprocessing

- **Categorical Encoding:** Utilizes custom functions and `LabelEncoder` from Scikit-learn to encode categorical variables (`Gender` and `Geography`).
- **Train-Test Split:** Splits the data into training and testing sets using `train_test_split` from Scikit-learn.
- **Feature Scaling:** Standardizes numerical features using `StandardScaler` from Scikit-learn.

### 3. Neural Network Modeling (Churn Prediction)

- **Model Architecture:** Builds a sequential neural network model using Keras with layers including Dense for binary classification.
- **Model Compilation:** Compiles the model with Adam optimizer and binary cross-entropy loss function.
- **Model Training:** Trains the neural network on the training data with specified batch size and number of epochs.
- **Model Evaluation:** Evaluates the model performance on the test set using confusion matrix and accuracy metrics.
- **Prediction:** Makes predictions for a new customer using the trained model.

### 4. Convolutional Neural Network (Optional - Sentiment Analysis)

- **Image Data Handling:** Uses `ImageDataGenerator` from Keras to preprocess and augment image data for a binary classification task (e.g., cat vs. dog).
- **CNN Architecture:** Constructs a Convolutional Neural Network (CNN) using Keras with Conv2D, MaxPooling2D, and Dense layers.
- **Model Compilation and Training:** Compiles and trains the CNN model on training data, evaluates on test data, and makes predictions on new images.

### 5. Custom Sentiment Analysis (Optional)

- **Text Preprocessing:** Tokenizes and preprocesses text data (Reddit comments) using NLTK for sentiment analysis.
- **Custom Neural Network:** Implements a basic neural network for sentiment analysis with forward propagation and loss calculation.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn

## Usage

1. Clone the repository:

