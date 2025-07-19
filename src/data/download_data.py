
import os
import pandas as pd
import numpy as np

def download_data(output_path=r'C:\Users\ganes\Downloads\Projects\Sentiment_Analysis\data'):
    """
    Simulates data download for sentiment analysis. In a real scenario, this would fetch data from sources
    like Amazon Product Reviews, Yelp Reviews, or IMDB Movie Reviews.
    """
    os.makedirs(output_path, exist_ok=True)
    print(f"Data would be downloaded to: {output_path}")
    
    # Increased dummy data for better stratification
    reviews = [
        "This product is amazing! I love it.",
        "Absolutely terrible, a complete waste of money.",
        "It's okay, nothing special.",
        "Highly recommend, very satisfied.",
        "Disappointing, expected more.",
        "Fantastic purchase, very happy.",
        "Worst experience ever, never buying again.",
        "Decent quality for the price.",
        "So good, I bought another one!",
        "Completely useless, don't bother.",
        "Pretty good, but could be better.",
        "Excellent value and performance.",
        "Frustrating to use, poor design.",
        "Love the features, very intuitive.",
        "Breaks easily, very flimsy.",
        "Solid product, does what it says.",
        "Not what I expected, a bit misleading.",
        "Perfect for my needs, highly satisfied.",
        "Terrible customer service, avoid!",
        "Surprisingly good for the price."
    ]
    sentiments = [
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'neutral', 'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative', 'positive'
    ]

    # Replicate data to ensure enough samples for train_test_split
    num_replications = 50 # Increase this to have more data
    all_reviews = []
    all_sentiments = []
    for _ in range(num_replications):
        all_reviews.extend(reviews)
        all_sentiments.extend(sentiments)

    df = pd.DataFrame({'review': all_reviews, 'sentiment': all_sentiments})
    dummy_data_path = os.path.join(output_path, 'raw_reviews.csv')
    df.to_csv(dummy_data_path, index=False)
    print(f"Dummy raw reviews saved to {dummy_data_path}")

if __name__ == "__main__":
    download_data()
