import streamlit as st
import pandas as pd
import joblib
import re
import string

# Load pre-trained models and vectorizer
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
xgboost_model = joblib.load('xgboost_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Emotion classes
classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 50px;
        color: #ff6347;
        text-align: center;
        font-family: 'Comic Sans MS', 'Comic Sans', cursive;
    }
    .subheader {
        font-size: 30px;
        color: #32cd32;
        text-align: center;
        font-family: 'Comic Sans MS', 'Comic Sans', cursive;
    }
    .prediction {
        font-size: 20px;
        color: #1e90ff;
        text-align: center;
        font-family: 'Comic Sans MS', 'Comic Sans', cursive;
    }
    .image-caption {
        font-size: 18px;
        color: #ff4500;
        text-align: center;
        font-family: 'Comic Sans MS', 'Comic Sans', cursive;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Emotion Classifier</div>', unsafe_allow_html=True)

# Input text
sentence = st.text_input('Enter a sentence:')

def preprocess(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ", text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def predict_emotion(model, text):
    preprocessed_text = preprocess(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    pred = model.predict(vectorized_text)
    return pred[0]

if sentence:
    # Predictions
    nb_prediction = predict_emotion(naive_bayes_model, sentence)
    lr_prediction = predict_emotion(logistic_regression_model, sentence)
    xgb_prediction = predict_emotion(xgboost_model, sentence)

    # Display predictions
    st.markdown('<div class="subheader">Predictions</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction">Naive Bayes Prediction: {classes[nb_prediction]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction">Logistic Regression Prediction: {classes[lr_prediction]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction">XGBoost Prediction: {classes[xgb_prediction]}</div>', unsafe_allow_html=True)

# Display images of model performance
st.markdown('<div class="subheader">Model Performance Comparison</div>', unsafe_allow_html=True)
st.image("pics/comparison_bar.png", caption="Comparison of Model Accuracies", use_column_width=True)
st.markdown('<div class="image-caption">Comparison of Model Accuracies</div>', unsafe_allow_html=True)

st.markdown('<div class="subheader">Confusion Matrices</div>', unsafe_allow_html=True)
st.image("pics/cmnb.png", caption="Confusion Matrix: Naive Bayes", use_column_width=True)
st.markdown('<div class="image-caption">Confusion Matrix: Naive Bayes</div>', unsafe_allow_html=True)
st.image("pics/outputLOGREG.png", caption="Confusion Matrix: Logistic Regression", use_column_width=True)
st.markdown('<div class="image-caption">Confusion Matrix: Logistic Regression</div>', unsafe_allow_html=True)
st.image("pics/confusionmatrixneuralnet.png", caption="Confusion Matrix: Neural Net", use_column_width=True)
st.markdown('<div class="image-caption">Confusion Matrix: Neural Net</div>', unsafe_allow_html=True)
st.image("pics/xgbconfusionmatrix.png", caption="Confusion Matrix: XGBoost", use_column_width=True)
st.markdown('<div class="image-caption">Confusion Matrix: XGBoost</div>', unsafe_allow_html=True)

st.markdown('<div class="subheader">Data Class Distributions & Word Clouds</div>', unsafe_allow_html=True)
st.image("pics/output.png", caption="Class Distribution 1", use_column_width=True)
st.markdown('<div class="image-caption">Class Distribution 1</div>', unsafe_allow_html=True)
st.image("pics/output2.png", caption="Class Distribution 2", use_column_width=True)
st.markdown('<div class="image-caption">Class Distribution 2</div>', unsafe_allow_html=True)
st.image("pics/output3.png", caption="WordCloud before preprocessing", use_column_width=True)
st.markdown('<div class="image-caption">Class Distribution 3</div>', unsafe_allow_html=True)
st.image("pics/output4.png", caption="WordCloud after preprocessing", use_column_width=True)
st.markdown('<div class="image-caption">Class Distribution 4</div>', unsafe_allow_html=True)

st.markdown('<div class="subheader">ROC Curves</div>', unsafe_allow_html=True)
st.image("pics/rocnb.png", caption="ROC Curve: Naive Bayes", use_column_width=True)
st.markdown('<div class="image-caption">ROC Curve: Naive Bayes</div>', unsafe_allow_html=True)
st.image("pics/roclogreg.png", caption="ROC Curve: Logistic Regression", use_column_width=True)
st.markdown('<div class="image-caption">ROC Curve: Logistic Regression</div>', unsafe_allow_html=True)
st.image("pics/xgbroc.png", caption="ROC Curve: XGBoost", use_column_width=True)
st.markdown('<div class="image-caption">ROC Curve: XGBoost</div>', unsafe_allow_html=True)
