# Phishing Email Detection using Machine Learning in Python

This project uses machine learning to detect phishing (spam) emails based on their content. It converts email text into numerical features using TF-IDF and classifies messages as either **Safe (Ham)** or **Phishing (Spam)** using trained models. The results, along with confidence scores and timestamps, are stored in a local database.

---

## Features

- Train and evaluate **Multinomial Naive Bayes** and **Logistic Regression** models
- Preprocess raw text using **TF-IDF Vectorizer**
- Use a **Command-Line Interface (CLI)** to test any email
- **Log predictions** (with confidence and timestamp) in a local **SQLite database**
- Visualize dataset distribution and model performance

---

## Project Structure

| File / Folder         | Description                                        |
|-----------------------|----------------------------------------------------|
| `main_training.py`    | Trains the ML model and generates evaluation graphs |
| `cli_predict.py`      | CLI script that takes email input and predicts spam/ham |
| `phishing_model.pkl`  | Saved model file (Logistic Regression)             |
| `vectorizer.pkl`      | Saved TF-IDF vectorizer                            |
| `results.db`          | SQLite database storing email predictions          |
| `README.md`           | Project guide and usage documentation              |
| `requirements.txt`    | List of required libraries                         |

---

## Requirements
- Python 3.x+
- Pandas, scikit-learn, matpotlib, seaborn, sqlite3, argparse, joblib are the required libraries.

---

## How to Use
- Train the model - Run in terminal - 'python p_email.py'
- Check new email - Run in terminal - 'python cli_pi.py --text "Click here to verify your account"

---

## output
- Email: Click here to verify your account
- Prediction: safe (ham)
- Confidence: 0.67
- Logged to results.db 

