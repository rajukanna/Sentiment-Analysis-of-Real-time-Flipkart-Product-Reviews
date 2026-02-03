import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

from utils import clean_text

print("Loading dataset...")
df = pd.read_csv("data/data.csv")


required_columns = ["Reviewer Rating", "Review Text"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")


print("Creating sentiment labels...")
df["sentiment"] = df["Reviewer Rating"].apply(lambda x: 1 if x >= 4 else 0)


print("Cleaning review text...")
df["cleaned_review"] = df["Review Text"].apply(clean_text)


print("Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]


print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


print("Evaluating model...")
y_pred = model.predict(X_test)

print("\nF1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


print("Saving model and vectorizer...")
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nTraining completed successfully!")
print("Files saved:")
print("   - model/sentiment_model.pkl")
print("   - model/tfidf_vectorizer.pkl")
