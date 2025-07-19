
import unittest
import os
import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Assuming models are saved in the 'models' directory
MODEL_DIR = r'C:\Users\ganes\Downloads\Projects\Sentiment_Analysis\models'
DATA_DIR = r'C:\Users\ganes\Downloads\Projects\Sentiment_Analysis\data'

class TestSentimentModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure models and data exist for testing
        cls.lr_model_path = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')
        cls.tfidf_vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
        cls.svm_model_path = os.path.join(MODEL_DIR, 'svm_model.pkl')
        cls.bert_model_dir = os.path.join(MODEL_DIR, 'bert_sentiment_model')
        cls.preprocessed_data_path = os.path.join(DATA_DIR, 'preprocessed_reviews.csv')

        # Check if model files exist before attempting to load
        if not os.path.exists(cls.lr_model_path):
            raise FileNotFoundError(f"Logistic Regression model not found at {cls.lr_model_path}")
        if not os.path.exists(cls.tfidf_vectorizer_path):
            raise FileNotFoundError(f"TF-IDF vectorizer not found at {cls.tfidf_vectorizer_path}")
        if not os.path.exists(cls.svm_model_path):
            raise FileNotFoundError(f"SVM model not found at {cls.svm_model_path}")
        # BERT model directory might not exist if training failed or was skipped
        # We'll handle this gracefully by not raising an error if it's missing
        # but tests related to BERT will then be skipped or fail.
        if not os.path.exists(cls.bert_model_dir):
            print(f"Warning: BERT model directory not found at {cls.bert_model_dir}. BERT tests will be skipped.")
            cls.bert_model = None
            cls.bert_tokenizer = None
        else:
            cls.bert_tokenizer = AutoTokenizer.from_pretrained(cls.bert_model_dir)
            cls.bert_model = AutoModelForSequenceClassification.from_pretrained(cls.bert_model_dir)

        if not os.path.exists(cls.preprocessed_data_path):
            raise FileNotFoundError(f"Preprocessed data not found at {cls.preprocessed_data_path}")

        # Load models and data for tests
        cls.lr_model = joblib.load(cls.lr_model_path)
        cls.tfidf_vectorizer = joblib.load(cls.tfidf_vectorizer_path)
        cls.svm_model = joblib.load(cls.svm_model_path)
        cls.df_preprocessed = pd.read_csv(cls.preprocessed_data_path)

        # Prepare sample data for prediction tests
        cls.sample_review = "This is a great product!"
        cls.sample_review_lemmatized = "this be a great product"

    def test_logistic_regression_prediction(self):
        # Test if LR model can make a prediction
        vectorized_review = self.tfidf_vectorizer.transform([self.sample_review_lemmatized])
        prediction = self.lr_model.predict(vectorized_review)
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape[0], 1)

    def test_svm_prediction(self):
        # Test if SVM model can make a prediction
        vectorized_review = self.tfidf_vectorizer.transform([self.sample_review_lemmatized])
        prediction = self.svm_model.predict(vectorized_review)
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape[0], 1)

    def test_bert_prediction(self):
        if self.bert_model is None:
            self.skipTest("BERT model not loaded, skipping test.")
        # Test if BERT model can make a prediction
        inputs = self.bert_tokenizer(self.sample_review, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).numpy()
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape[0], 1)

if __name__ == '__main__':
    unittest.main()
