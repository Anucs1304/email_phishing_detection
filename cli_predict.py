import argparse
import joblib
import sqlite3
import datetime

# Load the trained model and vectorizer
model = joblib.load('phishing_model.pkl')         # Make sure this file exists
vectorizer = joblib.load('vectorizer.pkl')        # Same for this one

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Phishing Email Detector")
parser.add_argument('--text', type=str, required=True, help="Email text to analyze")
args = parser.parse_args()

# Transform and predict
X_input = vectorizer.transform([args.text])
proba = model.predict_proba(X_input)[0]         # Get probabilities
prediction = model.predict(X_input)[0]          # Get label
confidence = max(proba)                         # Highest probability = model confidence

# Show result
result = "Phishing (Spam)" if prediction == 1 else "Safe (Ham)"
print(f"\nEmail: {args.text}")
print("Prediction:", result)
print(f"Confidence: {confidence:.2f}")

# Log to SQLite database with confidence
conn = sqlite3.connect('results.db')
c = conn.cursor()

# Updated schema with 'confidence'
c.execute('''CREATE TABLE IF NOT EXISTS predictions (
    text TEXT,
    prediction INTEGER,
    confidence REAL,
    timestamp TEXT
)''')

# Log the current prediction
c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?)",
          (args.text, int(prediction), float(confidence), str(datetime.datetime.now())))

conn.commit()
conn.close()

print("Logged to results.db")

