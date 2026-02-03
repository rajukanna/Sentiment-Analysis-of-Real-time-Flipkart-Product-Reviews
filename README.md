# Flipkart Review Sentiment Analysis

An end-to-end **NLP and Machine Learning project** that analyzes customer sentiment from Flipkart product reviews.  
The project covers data preprocessing, feature extraction, model training, evaluation, and deployment of a **Flask web application** on **AWS EC2** for real-time sentiment prediction.

---

## 📌 Project Overview

Customer reviews play a critical role in online purchasing decisions.  
This project classifies Flipkart product reviews as **Positive** or **Negative** using Natural Language Processing (NLP) techniques and Machine Learning.

### Key Objectives
- Analyze customer sentiment from Flipkart reviews
- Identify pain points from negative reviews
- Build a complete ML pipeline
- Deploy a real-time sentiment analysis web app

---

## 🗂 Dataset

- Source: Provided Flipkart review dataset (scraped beforehand)
- File: `data/data.csv`
- Records: ~8,500 customer reviews
- Important Columns:
  - Rating
  - Review Text
  - Review Title
  - Reviewer Name
  - Review Date
  - Up Votes / Down Votes

---

## 🏗 Project Structure
```bash
flipkart_sentiment_analysis/
│
├── data/
│ └── data.csv
│
├── model/
│ ├── sentiment_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── app.py
├── train_model.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK  
- **NLP Technique:** TF-IDF  
- **ML Algorithm:** Logistic Regression  
- **Web Framework:** Flask  
- **Deployment:** AWS EC2  

---

## 🔄 Workflow

1. Load and explore the dataset
2. Clean and preprocess review text
3. Convert text to numerical features using TF-IDF
4. Train a sentiment classification model
5. Evaluate model using F1-score
6. Save trained model and vectorizer
7. Build Flask web application
8. Deploy application on AWS EC2

---

## 🧹 Text Preprocessing

- Lowercasing
- Removal of special characters and punctuation
- Stopword removal using NLTK
- Same preprocessing used for **training and inference**

---

## 🧠 Model Training

- **Algorithm:** Logistic Regression
- **Feature Extraction:** TF-IDF (max 5000 features)
- **Evaluation Metric:** F1-Score
- **Labeling Logic:**
  - Rating ≥ 4 → Positive
  - Rating ≤ 3 → Negative

----

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/rajukanna/flipkart_sentiment_analysis.git
cd flipkart_sentiment_analysis
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model
```bash
python train_model.py
```

### 5️⃣ Run Flask App
```bash
python app.py
```


Open browser:
```bash
http://127.0.0.1:5000
```
🎯 Conclusion

This project demonstrates a complete machine learning lifecycle — from raw data to a deployed web application.
It follows industry best practices for NLP preprocessing, model training, deployment, and testing.

## 👥 Contributors

- **Innomatics Research Labs** – Project Lead & Mentor

- **Raju** – Intern  
- Open to community contributions 🚀  

Feel free to fork this repository, raise issues, or submit pull requests.

⭐ If you like this project, give it a star on GitHub!

