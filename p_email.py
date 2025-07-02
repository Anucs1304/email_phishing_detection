import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv(
    r"C:\Users\anucs\OneDrive\Desktop\email phishing detector\SMSSpamCollection\spam.csv",
    encoding='latin1',
    usecols=[0, 1],  # Only read the first two columns
    names=["label", "text"],
    skiprows=1       # Skip the first row if it's a header
)

df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 and 1
df.dropna(inplace=True)

print("Dataset Preview: ")
print(df.head())
print(df['text'].apply(type).value_counts())
print(df.columns)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=df['label'])
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)
nb_pred = nb_model.predict(X_test_vectorized)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vectorized, y_train)
lr_pred = lr_model.predict(X_test_vectorized)

# Evaluate both
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

y_pred = lr_model.predict(X_test_vectorized)


joblib.dump(lr_model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

sns.countplot(x='label', data=df)
plt.title('Distribution of Ham (0) and Spam (1)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Try a sample prediction
sample = "Congratulations! You’ve won a free prize"
sample_vect = vectorizer.transform([sample])
sample_pred = lr_model.predict(sample_vect)[0]

print("\n🔍 Sample Email Test:")
print(f"Input: {sample}")
print("Prediction:", "Phishing (spam)" if sample_pred == 1 else "Safe (ham)")

# Pause to keep terminal open
input("\nPress Enter to exit...")