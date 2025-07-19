
import pandas as pd
import os
import spacy

def extract_insights(input_path=r'C:\Users\ganes\Downloads\Projects\Sentiment_Analysis\data'):
    """
    Extracts insights from processed reviews, including key phrases and product aspects.
    """
    preprocessed_data_path = os.path.join(input_path, 'preprocessed_reviews.csv')
    if not os.path.exists(preprocessed_data_path):
        print(f"Error: Preprocessed data not found at {preprocessed_data_path}")
        return

    df = pd.read_csv(preprocessed_data_path)
    print(f"Loaded preprocessed data from {preprocessed_data_path}")

    # Load spaCy model for NER and dependency parsing
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Error loading spaCy model for insight extraction: {e}")
        print("Skipping insight extraction.")
        return

    print("Identifying key phrases with spaCy NER...")
    # Example: Extracting named entities (persons, organizations, etc.)
    df['entities'] = df['lemmatized_review'].apply(lambda text: [(ent.text, ent.label_) for ent in nlp(text).ents])

    print("Extracting product aspects using dependency parsing (placeholder)...")
    # This would involve more complex NLP logic to identify nouns and their modifiers
    # related to product features. For now, it's a conceptual placeholder.
    df['aspects'] = df['lemmatized_review'].apply(lambda text: "[placeholder_aspects]")

    print("Generating sentiment trend reports (conceptual placeholder)...")
    # This would involve aggregating sentiment over time or by product category
    # and generating summary statistics or visualizations.

    print("Insights extracted and added to DataFrame (conceptual).")
    # You might save this DataFrame or use it to generate reports/visualizations
    # For now, just print a sample of the DataFrame with new columns
    print(df[['review', 'sentiment', 'entities', 'aspects']].head())

if __name__ == "__main__":
    extract_insights()
