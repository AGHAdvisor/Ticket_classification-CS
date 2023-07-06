import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

# Load the training dataset
train_df = pd.read_excel('Clean_Data1.xlsx')  # Update with your file path

# Extract keywords from the manual classification column and create a new column
keywords = ['Collaboration', 'Complaint', 'Compliment', 'Feedback', 'Suggestion',
            'Query', 'Junk', 'Non Relevant', 'Universal', 'Follow Up']
train_df['new_classification'] = train_df['Ticket subject'].apply(
    lambda x: next((kw for kw in keywords if kw.lower() in x.lower()), 'Other')
)

# Drop rows where the new classification is None (no keyword found)
train_df = train_df.dropna(subset=['new_classification'])

# Clean and preprocess the training data messages
train_features = train_df['Brief Description of Feedback'].fillna('').apply(clean_text)
train_labels = train_df['new_classification']

# Convert the text messages into numerical feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_features)

# Train a Random Forest Classifier with hyperparameter tuning
num_trees = 25
classifier = RandomForestClassifier(n_estimators=num_trees)

# Train the classifier with the entire training dataset
classifier.fit(train_features, train_labels)

# Define a function to classify new messages
def classify_message(message):
    cleaned_message = clean_text(message)
    message_features = vectorizer.transform([cleaned_message])
    prediction = classifier.predict(message_features)[0]
    return prediction

# Create the Streamlit app
def main():
    st.title("Ticket Classification App")

    # Upload a file for classification
    uploaded_file = st.file_uploader("Upload a file", type="xlsx")
    if uploaded_file is not None:
        test_df = pd.read_excel(uploaded_file)

        # Check if the message column exists
        if "Message" in test_df.columns:
            test_messages = test_df["Message"]
        elif "Brief Description of Feedback" in test_df.columns:
            test_messages = test_df["Brief Description of Feedback"]
        else:
            st.error("Error: Message column not found in the uploaded file.")
            return

        # Clean and preprocess the test data messages
        test_features = test_messages.fillna('').apply(clean_text)

        # Convert the test data messages into numerical feature vectors using TF-IDF
        test_features = vectorizer.transform(test_features)

        # Predict the classification on the test dataset
        predictions = classifier.predict(test_features)

        # Check if the message column exists
        if "Junk" in test_df.columns:
            test_df['Classified Class'] = test_df['Junk'].apply(
            lambda x: next((kw for kw in keywords if kw.lower() in x.lower()), 'Other')
        )
        elif "Ticket Subject" in test_df.columns:
            test_df['Classified Class'] = test_df['Ticket Subject'].apply(
            lambda x: next((kw for kw in keywords if kw.lower() in x.lower()), 'Other')
        )
        else:
            st.error("Error: Junk or Ticket subject column not found in the uploaded file.")
            return

        # Map "no comment" to "To Check" in the predicted classifications
        predictions = np.where(test_messages.str.lower() == '(no comment)', 'To Check', predictions)

        # Add the predictions to the test dataframe
        test_df['predicted_classification'] = predictions

        # Print the classification report for evaluation
        st.text("Classification Report:")
        st.text(classification_report(test_df['Classified Class'], predictions))

        # Display the classified data
        st.dataframe(test_df)

if __name__ == '__main__':
    main()
