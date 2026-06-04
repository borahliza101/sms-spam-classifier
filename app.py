from flask import Flask, render_template, request, jsonify
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load saved model
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()

    tokens = word_tokenize(text)

    filtered_tokens = []

    for word in tokens:
        if word not in stop_words:
            filtered_tokens.append(word)

    stemmed_tokens = []

    for word in filtered_tokens:
        stemmed_tokens.append(ps.stem(word))

    return " ".join(stemmed_tokens)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    message = data["message"]

    transformed_message = transform_text(message)

    vector = vectorizer.transform([transformed_message])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    return jsonify({
        "label": "spam" if prediction == 1 else "ham",
        "probability": float(probability[1])
    })


import os

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )