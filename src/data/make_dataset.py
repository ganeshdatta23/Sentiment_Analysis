
import pandas as pd
import os

def make_dataset(input_path=r'C:\Users\ganes\Downloads\Projects\Sentiment_Analysis\data', output_path=r'C:\Users\ganes\Downloads\Projects\Sentiment_Analysis\data'):
    """
    Processes raw review data, ensuring binary sentiment labels.
    """
    raw_data_path = os.path.join(input_path, 'raw_reviews.csv')
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data not found at {raw_data_path}")
        return

    df = pd.read_csv(raw_data_path)
    print(f"Loaded raw data from {raw_data_path}")

    # Ensure sentiment is binary: 1 for positive, 0 for negative/neutral
    print("Ensuring binary sentiment labels...")
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 0})

    processed_data_path = os.path.join(output_path, 'processed_reviews.csv')
    df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    make_dataset()
