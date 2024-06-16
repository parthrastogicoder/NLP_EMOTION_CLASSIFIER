import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string

xgb_model = joblib.load('xgboost_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
logistic_model = joblib.load('logistic_regression_model.pkl')
naive_bayes_model = joblib.load('naive_bayes_model.pkl')

def f(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
def predict_emotion(sentence):
    processed_sentence = f(sentence)
    vectorized_sentence = vectorizer.transform([processed_sentence])
    predicted_label = xgb_model.predict(vectorized_sentence)
    label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    predicted_emotion = label_mapping.get(predicted_label[0], 'Unknown')
    
    return predicted_emotion

def predict_emotion1(sentence):
    processed_sentence = f(sentence)
    vectorized_sentence = vectorizer.transform([processed_sentence])
    predicted_label = logistic_model.predict(vectorized_sentence)
    label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    predicted_emotion = label_mapping.get(predicted_label[0], 'Unknown')
    return predicted_emotion

def predict_emotion2(sentence):
    processed_sentence = f(sentence)
    vectorized_sentence = vectorizer.transform([processed_sentence])
    predicted_label = naive_bayes_model.predict(vectorized_sentence)
    label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    predicted_emotion = label_mapping.get(predicted_label[0], 'Unknown')
    return predicted_emotion
if __name__ == "__main__":
    print("Welcome to Emotion Predictor CLI")
    print("Enter 'exit' to quit.")
    
    while True:
        sentence = input("\nEnter a sentence: ")
        if sentence.lower() == 'exit':
            print("Exiting...")
            break
        predicted_emotion = predict_emotion(sentence)
        print(f"Predicted Emotion xgb: {predicted_emotion}")
        predicted_emotion = predict_emotion1(sentence)
        print(f"Predicted Emotion logistic regression: {predicted_emotion}")
        predicted_emotion = predict_emotion2(sentence)
        print(f"Predicted Emotion naive bayes: {predicted_emotion}")
        # here are some sample inputs
        # i think that after i had spent some time investigating the surroundings and things i started to feel more curiouhe dishes
        # wow i am so happy and surprised you did this 
        # i am already feeling frantic
