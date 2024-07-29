import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle
import numpy as np
import re

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Load dataset
df = pd.read_csv(r"static\data2.csv")

# Drop 'Class' column if it exists
if 'Class' in df.columns:
    df.drop(columns=['Class'], inplace=True)

# Extract text data and labels
text_data = df['tweet'].tolist()
labels = df['class'].tolist()

# Preprocessing and TF-IDF Vectorization
preprocessor = FunctionTransformer(lambda x: [preprocess_text(text) for text in x])
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('tfidf', tfidf_vectorizer)
])

# Fit-transform the pipeline on the text data
tfidf_matrix = pipeline.fit_transform(text_data)

# Save TF-IDF vectorizer to pickle file
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(tfidf_matrix, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize models
rf_classifier = RandomForestClassifier(random_state=42)
lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
xgb_classifier = XGBClassifier(random_state=42)
svm_classifier = SVC(random_state=42)
dt_classifier = DecisionTreeClassifier(random_state=42)

# Function to train and evaluate model
def train_evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f'Validation Accuracy ({model_name}): {val_accuracy:.4f}')
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f'Test Accuracy ({model_name}): {test_accuracy:.4f}')
    return test_accuracy

# Train and evaluate each model
rf_accuracy = train_evaluate_model(rf_classifier, 'Random Forest', X_train, y_train, X_val, y_val, X_test, y_test)
lr_accuracy = train_evaluate_model(lr_classifier, 'Logistic Regression', X_train, y_train, X_val, y_val, X_test, y_test)
xgb_accuracy = train_evaluate_model(xgb_classifier, 'XGBoost', X_train, y_train, X_val, y_val, X_test, y_test)
svm_accuracy = train_evaluate_model(svm_classifier, 'SVM', X_train, y_train, X_val, y_val, X_test, y_test)
dt_accuracy = train_evaluate_model(dt_classifier, 'Decision Tree', X_train, y_train, X_val, y_val, X_test, y_test)

# Save the best model (XGBoost)
best_model = xgb_classifier
joblib.dump(best_model, "ensemble_classifier.joblib")
