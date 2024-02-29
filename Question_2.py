import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import string


data = pd.read_csv(r"C:\Users\deepu\OneDrive\Desktop\DATA SCIENCE\MACHINE LEARNING\Assignment-6\Question_2\train.csv") 

print("Data exploration:")
print(data.describe())

data = data[data['target'] > 0]

X_train, X_test, y_train, y_test = train_test_split(data['excerpt'], data['target'], test_size=0.2, random_state=42)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=preprocess_text)),
    ('model', Ridge())
])

parameters = {
    'tfidf__max_features': [1000, 2000, 3000],
    'model__alpha': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)

train_preds = grid_search.predict(X_train)
test_preds = grid_search.predict(X_test)

train_rmse = mean_squared_error(y_train, train_preds, squared=False)
test_rmse = mean_squared_error(y_test, test_preds, squared=False)

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

best_model = grid_search.best_estimator_.named_steps['model']
feature_names = grid_search.best_estimator_.named_steps['tfidf'].get_feature_names_out()
