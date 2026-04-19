import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv(r"E:\PRACTICE ON VS CODE\Machine Learning\Language detection major project\LANG.csv")

# Show language distribution
plt.figure(figsize=(10, 5))
sns.countplot(y='Language', data=df, order=df['Language'].value_counts().index)
plt.title("Language Distribution")
plt.xlabel("Count")
plt.ylabel("Language")
plt.tight_layout()
plt.show()

# Split the data into features and labels
X = df['Text']
y = df['Language']

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔹 Plot Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 🔹 Bar chart of F1-scores per language
f1_scores = {label: report[label]['f1-score'] for label in model.classes_ if label in report}
plt.figure(figsize=(10, 5))
sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
plt.title("F1 Score per Language")
plt.ylabel("F1 Score")
plt.xlabel("Language")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Example prediction
# User input for prediction
user_input = input("\nEnter a sentence to detect its language: ")
sample_vec = vectorizer.transform([user_input])
prediction = model.predict(sample_vec)
print("Predicted Language:", prediction[0])