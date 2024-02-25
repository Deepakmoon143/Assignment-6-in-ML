import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


data = pd.read_csv(r"C:\Users\deepu\OneDrive\Desktop\DATA SCIENCE\MACHINE LEARNING\Assignment-6\Question_1\spam.csv", encoding='ISO-8859-1')

data.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

data["label"] = data["label"].map({'ham': 0, 'spam': 1})

data["message"] = data["message"].str.lower()  
data["message"] = data["message"].str.replace("[^\w\s]", "")  

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(data["message"])


X_train, X_test, y_train, y_test = train_test_split(features, data["label"], test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

conf_matrix = confusion_matrix(y_test, predictions)
plt.matshow(conf_matrix)
plt.title("Confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()
