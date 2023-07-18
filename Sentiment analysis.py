import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
import base64

def preprocess_text(text):
    if pd.isnull(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

def main():
    st.title("Positive Classified Messages")

    # Upload input data
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)

            # Convert 'Message' column to string
            df['Message'] = df['Message'].astype(str)

            # Clean the text
            df['cleaned_text'] = df['Message'].apply(preprocess_text)

            # Create a SentimentIntensityAnalyzer object
            sia = SentimentIntensityAnalyzer()

            # Perform sentiment analysis and filter positive messages
            df['sentiment_score'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
            df['Positive Sentiment'] = df['sentiment_score'].apply(lambda x: True if x > 0 else False)
            positive_messages = df[df['Positive Sentiment']]

            # Display positive messages in a DataFrame
            positive_df = positive_messages[['Message', 'Positive Sentiment']]

            # Download classified dataset as CSV
            csv = positive_df.to_csv(index=False)
            # Encode CSV data as Base64
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="classified_data.csv">Download Classified Data</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Display the DataFrame in Streamlit
            st.subheader("Positive Classified Messages")
            st.dataframe(positive_df)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
