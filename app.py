from flask import Flask, request, render_template_string
import joblib
from utils import clean_text

app = Flask(__name__)

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Flipkart Sentiment Analysis</title>
    <style>
        body {
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: Arial, sans-serif;
            background-color: #f5f6fa;
        }
        .container {
            width: 420px;
            padding: 30px;
            background: #ffffff;
            border-radius: 10px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.08);
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #dcdde1;
            resize: none;
        }
        button {
            margin-top: 15px;
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 6px;
            background-color: #40739e;
            color: white;
            font-size: 15px;
        }
        .positive { color: #2ecc71; }
        .negative { color: #e84118; }
    </style>
</head>
<body>

<div class="container">
    <h2>Flipkart Review Sentiment</h2>

    <form method="POST">
        <textarea name="review" placeholder="Enter product review..." required></textarea>
        <button type="submit">Analyze</button>
    </form>

    {% if sentiment %}
        <h3 class="{{ sentiment_class }}">{{ sentiment }}</h3>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    sentiment_class = None

    if request.method == "POST":
        review = request.form["review"]

        cleaned_review = clean_text(review)

        vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            sentiment = "Positive "
            sentiment_class = "positive"
        else:
            sentiment = "Negative "
            sentiment_class = "negative"

    return render_template_string(
        HTML_TEMPLATE,
        sentiment=sentiment,
        sentiment_class=sentiment_class
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
