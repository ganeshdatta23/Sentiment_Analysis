# Sentiment Analysis for Product Reviews

## Objective
Automate analysis of 10K+ customer reviews

## Tools
NLP (NLTK, SpaCy), HuggingFace Transformers, Python, Power BI

## Key Result
88% classification accuracy

## Resources & Data
**Primary Dataset:** Amazon Product Reviews (Electronics/Apparel categories)

**Alternative Datasets:**
- Yelp Reviews Dataset
- IMDB Movie Reviews

## Implementation Plan

### Data Collection
- Scrape reviews using BeautifulSoup (or use pre-cleaned datasets)
- Balance positive/negative samples

### Text Preprocessing
- Clean text: remove emojis, special characters, HTML tags
- Implement lemmatization with spaCy
- Create TF-IDF vectors

### Model Training
- Fine-tune BERT-base using transformers library
- Compare with traditional models (Logistic Regression, SVM)
- Handle class imbalance with SMOTE

### Insight Extraction
- Identify key phrases with spaCy NER
- Extract product aspects using dependency parsing
- Generate sentiment trend reports

### Dashboard Development
- Build Power BI workbook with:
    - Sentiment distribution pie charts
    - Trending topics word clouds
    - Time-series sentiment trends
- Connect to live database (PostgreSQL)
