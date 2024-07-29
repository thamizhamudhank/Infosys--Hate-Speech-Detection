#!/usr/bin/env python
# coding: utf-8

# ## Hate Speech Detection - Milestone 2

# In[1]:


#Stuthi Shrisha


# ### Preprocessing Unbalanced Dataset and TfIdf Vectorization Phase 2

# In[2]:


#Importing necessary libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


# In[3]:


# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    return text.strip()  # Strip leading/trailing whitespace


# New data is unbalanced, contains 24783 rows, 7 columns with the class column as the target feature

# In[4]:


#loading unbalanced dataset
try:
    df = pd.read_csv(r"C:\Users\viole\Desktop\Datasets\data2.csv")
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    exit(1)



# In[5]:


if 'Class' in df.columns:
    df.drop(columns=['Class'], inplace=True)


# In[6]:


text_data = df['tweet'].tolist()
labels = df['class'].tolist()


# In[7]:


preprocessor = FunctionTransformer(lambda x: [preprocess_text(text) for text in x])


# In[8]:


tfidf_vectorizer = TfidfVectorizer(max_features=1000) 


# In[9]:


pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('tfidf', tfidf_vectorizer)
])


# In[10]:


# Fit-transform the pipeline on the text data
try:
    tfidf_matrix = pipeline.fit_transform(text_data)
except Exception as e:
    print(f"Error in pipeline fit-transform: {e}")
    exit(1)


# In[11]:


tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())


# In[12]:


output_file = 'tfidf_data.csv'


# In[13]:


#tfidf_df.to_csv(output_file, index=False)
#print(f"TF-IDF vectorized data saved to {output_file}")


# ### Data Splitting

# Data is now split into training, testing and validation sets, with a ratio of 70:15:15

# In[14]:


# Split data into training, validation, and test sets
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(tfidf_matrix, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# ### Model Training and Testing

# In[15]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


from xgboost import XGBClassifier

xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)
val_predictions_xgb = xgb_classifier.predict(X_val)


# In[27]:


# Evaluate performance on validation set
val_accuracy_xgb = accuracy_score(y_val, val_predictions_xgb)
#print(f'Validation Accuracy (XGBoost): {val_accuracy_xgb:.4f}')

val_classification_report_xgb = classification_report(y_val, val_predictions_xgb)
#print('Classification Report on Validation Set (XGBoost):')
#print(val_classification_report_xgb)


# In[28]:


sns.heatmap(confusion_matrix(y_val, val_predictions_xgb), annot=True)


# In[29]:


# Optionally, evaluate on test set if needed
test_predictions_xgb = xgb_classifier.predict(X_test)
test_accuracy_xgb = accuracy_score(y_test, test_predictions_xgb)

sns.heatmap(confusion_matrix(y_test, test_predictions_xgb), annot=True)



import pickle
import joblib
import numpy as np

# Load the model and vectorizer from the file
loaded_model = joblib.load("xgb_classifier.joblib")
with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_tfidf_vectorizer = pickle.load(f)

class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Not Hate Speech or Offensive Language'}

def classify_text(text):
    # Transform the input data using the loaded TF-IDF vectorizer
    input_transformed = loaded_tfidf_vectorizer.transform([text])

    # Make prediction
    prediction = loaded_model.predict(input_transformed)

    # Get the predicted class label
    predicted_class = class_labels[prediction[0]]
    return predicted_class





if __name__ == "__main__":
    # Ask for user input
    input_text = input("Enter text to classify: ")
    
    # Classify the input text
    predicted_class = classify_text(input_text)
    
    # Print the classification result
    print(f"Input: {input_text} -> Prediction: {predicted_class}")


