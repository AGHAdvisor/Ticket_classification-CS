import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import base64 as b64
import openpyxl

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    if pd.isnull(text):  # Handle NaN values
        return ""

    # Convert to lowercase
    text = str(text).lower()

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
train_df = pd.read_excel('df2.xlsx')  # Update with your file path

# Extract keywords from the manual classification column and create a new column
keywords = ['Collaboration', 'Complaint', 'Compliment', 'Feedback', 'Suggestion',
            'Query', 'Junk', 'Non Relevant', 'Universal', 'Follow Up']
train_df['new_classification'] = train_df['Ticket form'].apply(
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

# Perform clustering to identify patterns
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
train_clusters = kmeans.fit_predict(train_features)

# Train a Linear SVM Classifier for each cluster
classifiers = []
for cluster in range(num_clusters):
    cluster_indices = np.where(train_clusters == cluster)[0]
    cluster_features = train_features[cluster_indices]
    cluster_labels = train_labels.iloc[cluster_indices]

    classifier = LinearSVC()
    classifier.fit(cluster_features, cluster_labels)
    classifiers.append(classifier)

# Save the trained models
model_files = []
for i, classifier in enumerate(classifiers):
    model_file = f"classifier_model_cluster{i}.joblib"
    joblib.dump(classifier, model_file)
    model_files.append(model_file)

# Define a function to classify new messages
def classify_message(message):
    cleaned_message = clean_text(message)
    message_features = vectorizer.transform([cleaned_message])
    cluster = kmeans.predict(message_features)[0]
    classifier = classifiers[cluster]
    prediction = classifier.predict(message_features)[0]
    confidence_score = np.max(classifier.decision_function(message_features))
    return prediction, confidence_score

# Create the Streamlit app
def main():
    st.set_page_config(
        page_title="Ticket Classification App",
        page_icon="🎫",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Ticket Classification App")

    # Upload a file for classification
    uploaded_file = st.file_uploader("Upload a file", type="xlsx")
    if uploaded_file is not None:
        test_df = pd.read_excel(uploaded_file)

        # Check if the 'Ticket form' column exists
        if 'Ticket form' not in test_df.columns:
            st.error("Error: 'Ticket form' column not found in the uploaded file.")
            return

        # Filter the rows where 'Ticket form' contains 'Junk' keyword
        test_df = test_df[test_df['Ticket form'].str.lower().str.contains('junk')]

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

        # Predict the classification and confidence score on the test dataset
        predictions, confidence_scores = zip(*[classify_message(msg) for msg in test_messages])

        # Map "no comment" to "To Check" in the predicted classifications
        predictions = np.where(test_messages.str.lower() == '(no comment)', 'To Check', predictions)

        # Check if the message column exists
        if "Junk" in test_df.columns:
            test_df['Classified Class'] = test_df['Junk'].apply(
            lambda x: next((kw for kw in keywords if kw.lower() in x.lower()), 'Other')
        )
        elif "Ticket subject" in test_df.columns:
            test_df['Classified Class'] = test_df['Ticket subject'].apply(
            lambda x: next((kw for kw in keywords if kw.lower() in x.lower()), 'Other')
        )
        elif "Ticket form" in test_df.columns:
            test_df['Classified Class'] = test_df['Ticket form'].apply(
            lambda x: next((kw for kw in keywords if kw.lower() in x.lower()), 'Other')
        )
        else:
            st.error("Error: Junk or Ticket subject column not found in the uploaded file.")
            return

        # Add the predictions and confidence scores to the test dataframe
        test_df['predicted_classification'] = predictions
        test_df['confidence_score'] = confidence_scores

        # Print the classification report for evaluation
        st.subheader("Classification Report:")
        st.text(classification_report(test_df['Classified Class'], predictions))

        # Display the classified data
        st.subheader("Classified Data")
        st.dataframe(test_df)

        # Download the classified data as a CSV file
        csv = test_df.to_csv(index=False)
        href = f'<a href="data:file/csv;base64,{b64}" download="classified_data.csv">Download Classified Data</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Visualize the predicted classification distribution
        #st.subheader("Predicted Classification Distribution")
        #prediction_counts = test_df['predicted_classification'].value_counts()
        #plt.figure(figsize=(10, 6))
        #sns.barplot(x=prediction_counts.index, y=prediction_counts.values, palette='muted')
        #plt.xlabel("Predicted Classification")
        #plt.ylabel("Count")
        #plt.xticks(rotation=45)
        #st.pyplot(plt)

if __name__ == '__main__':
    main()
