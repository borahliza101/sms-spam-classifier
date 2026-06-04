import pandas as pd
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score


# -------------------------
# Text Preprocessing
# -------------------------

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):

    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    filtered_tokens = []

    for word in tokens:
        if word not in stop_words:
            filtered_tokens.append(word)

    # Stemming
    stemmed_tokens = []

    for word in filtered_tokens:
        stemmed_tokens.append(ps.stem(word))

    return " ".join(stemmed_tokens)


# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("data/spam.csv", encoding='latin-1')

# Remove extra columns
df = df.iloc[:, :2]

# Rename columns
df.rename(columns={
    'v1': 'label',
    'v2': 'message'
}, inplace=True)


# -------------------------
# Preprocess Messages
# -------------------------

df['transformed_message'] = df['message'].apply(transform_text)


# -------------------------
# Encode Labels
# -------------------------

encoder = LabelEncoder()

y = encoder.fit_transform(df['label'])


# -------------------------
# TF-IDF
# -------------------------

tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_message'])


# -------------------------
# Train Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------
# Train Model
# -------------------------

model = MultinomialNB()

model.fit(X_train, y_train)


# -------------------------
# Evaluation
# -------------------------

y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))


# -------------------------
# Save Model
# -------------------------

joblib.dump(tfidf, "models/vectorizer.pkl")
joblib.dump(model, "models/model.pkl")

print("Model Saved Successfully!")