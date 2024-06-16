
# NLP Emotion Classification Project

Welcome to the NLP Emotion Classification project! This project focuses on classifying text data into six distinct emotion categories using various machine learning models. Below you'll find detailed information on how to set up the project, use the provided CLI tool, and understand the project's overall functionality and results.  To check out the code for training look at sample.ipynb

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Models Used](#models-used)
5. [Results](#results)
6. [CLI Tool](#cli-tool)
    - [Description](#description)
    - [Functionality](#functionality)
    - [Sample Inputs and Outputs](#sample-inputs-and-outputs)
    - [Usage Instructions](#usage-instructions)
7. [Setup and Installation](#setup-and-installation)

## Project Overview

This project aims to classify text into emotions such as sadness, joy, love, anger, fear, and surprise using machine learning techniques. The dataset is sourced from Kaggle and consists of text data labeled with these six emotion categories.

## Dataset

- **Source**: [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data)
- **Size**: 416,808 entries
- **Features**: Text data and corresponding emotion labels

### Class Labels and Emotions

| Label | Emotion   |
|-------|-----------|
| 0     | Sadness   |
| 1     | Joy       |
| 2     | Love      |
| 3     | Anger     |
| 4     | Fear      |
| 5     | Surprise  |

## Preprocessing

The preprocessing steps include:

- **Text Cleaning**: Removing unnecessary characters, converting to lowercase, removing punctuations and URLs.
- **Tokenization**: Splitting text into words or tokens for analysis.
- **Vectorization**: Converting text data into numerical representations using Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF).

## Models Used

The project implements several models to predict emotions:

1. **Naive Bayes**:
   - Accuracy: 74%
   - Simple probabilistic model

2. **Logistic Regression**:
   - Accuracy: 89.44%
   - Statistical model for binary outcomes

3. **XGBoost**:
   - Accuracy: 89.39%
   - Optimized gradient boosting algorithm

4. **Custom Neural Network**:
   - Accuracy: 88.66%
   - Multi-layer perceptron with Leaky ReLU and Sigmoid activation functions

## Results

### Model Performance

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Naive Bayes          | 0.74     | 0.79      | 0.74   | 0.69     |
| Logistic Regression  | 0.8944   | 0.89      | 0.89   | 0.89     |
| XGBoost              | 0.8939   | 0.90      | 0.89   | 0.90     |
| Neural Network       | 0.8866   | 0.89      | 0.89   | 0.89     |

### Visual Results

- [ROC Curve Comparison](docs/comparison.png)
- [Accuracy Bar Chart Comparison](docs/comparison_bar.png)

## CLI Tool

The CLI (Command-Line Interface) tool allows users to input a sentence and get an emotion prediction using trained machine learning models.

### Description

The tool interacts with the following models to predict emotions:
- **XGBoost**
- **Logistic Regression**
- **Naive Bayes**

### Functionality

1. **Text Preprocessing**: Cleans input text.
2. **Vectorization**: Converts text into numerical format.
3. **Prediction**: Uses models to predict emotions.
4. **User Interaction**: Accepts user input and returns predicted emotions.

### Sample Inputs and Outputs

- **Input**: "I am so happy and surprised you did this!"
  - **XGBoost**: Joy
  - **Logistic Regression**: Joy
  - **Naive Bayes**: Surprise

- **Input**: "I am already feeling frantic."
  - **XGBoost**: Fear
  - **Logistic Regression**: Fear
  - **Naive Bayes**: Fear

### Usage Instructions

1. Run the script using Python.
2. Enter a sentence to get emotion predictions.
3. The tool displays predictions from each model.
4. Type `exit` to quit the tool.

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/emotion-classification.git
    cd emotion-classification
    ```

2. **Install dependencies**:
    ```bash
    pip install (packages required)
    ```

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data) 

4. **Run the CLI tool**:
    ```bash
    python cliapp.py
    ```
5. **To train the model**: main file is sample.ipynb


## Contact

For any questions or suggestions, feel free to contact me at `parth22352@iiitd.ac.in`.

---
