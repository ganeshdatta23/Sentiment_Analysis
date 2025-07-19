
import pandas as pd
import re
import string
import spacy # Import spacy
import os

def preprocess_text(input_path=r'C:\Users\ganes\Downloads\Projects\Sentiment_Analysis\data', output_path=r'C:\Users\ganes\Downloads\Projects\Sentiment_Analysis\data'):
    """
    Cleans and preprocesses text data, including removing emojis, special characters,
    HTML tags, and performing lemmatization.
    """
    processed_data_path = os.path.join(input_path, 'processed_reviews.csv')
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data not found at {processed_data_path}")
        return

    df = pd.read_csv(processed_data_path)
    print(f"Loaded processed data from {processed_data_path}")

    # 1. Remove emojis, special characters, HTML tags
    print("Cleaning text...")
    def clean_text(text):
        text = re.sub(r'<.*?>', '', text) # Remove HTML tags
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text) # Remove emojis
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
        text = text.lower() # Convert to lowercase
        return text

    df['cleaned_review'] = df['review'].apply(clean_text)

    # 2. Implement lemmatization with spaCy
    print("Performing lemmatization...")
    try:
        nlp = spacy.load("en_core_web_sm") # Load the pre-trained model directly
    except Exception as e:
        print(f"Error loading spaCy model. Please ensure it's downloaded: python -m spacy download en_core_web_sm. Error: {e}")
        # Fallback if spaCy model is not available
        df['lemmatized_review'] = df['cleaned_review']
        print("Skipping lemmatization due to spaCy model error.")
        # Save the processed data even if lemmatization fails
        preprocessed_data_path = os.path.join(output_path, 'preprocessed_reviews.csv')
        df.to_csv(preprocessed_data_path, index=False)
        print(f"Preprocessed data saved to {preprocessed_data_path}")
        return

    def lemmatize_text(text):
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    df['lemmatized_review'] = df['cleaned_review'].apply(lemmatize_text)

    # 3. Create TF-IDF vectors (placeholder - typically done during model training)
    print("TF-IDF vectorization would be performed here during model training.")

    # Save the preprocessed data
    preprocessed_data_path = os.path.join(output_path, 'preprocessed_reviews.csv')
    df.to_csv(preprocessed_data_path, index=False)
    print(f"Preprocessed data saved to {preprocessed_data_path}")

if __name__ == "__main__":
    preprocess_text()
