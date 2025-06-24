import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK stopwords
nltk.download("stopwords")

# Load dataset
print("Loading dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=["label", "message"])
print(f"Dataset loaded with {len(df)} entries.\n")

# Preprocess text
def clean_text(msg):
    msg = msg.lower()
    msg = "".join([char for char in msg if char not in string.punctuation])
    words = msg.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

print("Cleaning messages...")
df["cleaned"] = df["message"].apply(clean_text)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"].map({"ham": 0, "spam": 1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict custom input
def predict_spam(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)
    return "Spam" if result[0] == 1 else "Not Spam"

# Test custom messages
print("\n--- Test your own messages ---")
while True:
    user_input = input("\nEnter a message (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    prediction = predict_spam(user_input)
    print("Prediction:", prediction)
