import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("spam.csv", encoding="latin-1")
# print(df.shape)
# print(df.head())
# print(df.columns)

# Keep only useful columns and rename them
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Check spam vs ham distribution
# print(df["label"].value_counts())
# print(df.head())


# Convert labels to numbers
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
predictions = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=["Ham", "Spam"]))
