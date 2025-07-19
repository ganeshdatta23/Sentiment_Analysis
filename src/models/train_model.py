

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import joblib

def train_model(input_path=r'C:\Users\ganes\Downloads\Projects\Sentiment_Analysis\data', output_path=r'C:\Users\ganes\Downloads\Projects\Sentiment_Analysis\models'):
    """
    Trains sentiment analysis models (BERT and traditional models) and saves them.
    """
    preprocessed_data_path = os.path.join(input_path, 'preprocessed_reviews.csv')
    if not os.path.exists(preprocessed_data_path):
        print(f"Error: Preprocessed data not found at {preprocessed_data_path}")
        return

    df = pd.read_csv(preprocessed_data_path)
    print(f"Loaded preprocessed data from {preprocessed_data_path}")

    # For simplicity, let's convert sentiment to binary (positive/negative)
    # Assuming 1 for positive, 0 for negative/neutral
    df['binary_sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 1 else 0)

    X = df['lemmatized_review']
    y = df['binary_sentiment']

    # Removed stratify=y to avoid ValueError with small dummy datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs(output_path, exist_ok=True)

    # --- Traditional Models (Logistic Regression, SVM) ---
    print("\nTraining Traditional Models (TF-IDF + Logistic Regression/SVM)...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit features for simplicity
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.fillna(''))
    X_test_tfidf = tfidf_vectorizer.transform(X_test.fillna(''))

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr))
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    joblib.dump(lr_model, os.path.join(output_path, 'logistic_regression_model.pkl'))
    joblib.dump(tfidf_vectorizer, os.path.join(output_path, 'tfidf_vectorizer.pkl'))
    print("Logistic Regression model and TF-IDF vectorizer saved.")

    # SVM
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred_svm))
    print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    joblib.dump(svm_model, os.path.join(output_path, 'svm_model.pkl'))
    print("SVM model saved.")

    # --- BERT Model Fine-tuning ---
    print("\nFine-tuning BERT-base model...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Create a custom dataset for HuggingFace Trainer
    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    # Tokenize data
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

    train_dataset = SentimentDataset(train_encodings, y_train.tolist())
    test_dataset = SentimentDataset(test_encodings, y_test.tolist())

    training_args = TrainingArguments(
        output_dir=os.path.join(output_path, 'results'),
        num_train_epochs=1,  # Reduced for quick demonstration
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_path, 'logs'),
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # compute_metrics=lambda p: {'accuracy': accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}
    )

    trainer.train()

    # Evaluate BERT model
    bert_results = trainer.evaluate()
    print("BERT Model Evaluation:", bert_results)

    # Save BERT model
    model.save_pretrained(os.path.join(output_path, 'bert_sentiment_model'))
    tokenizer.save_pretrained(os.path.join(output_path, 'bert_sentiment_model'))
    print("BERT model saved.")

if __name__ == "__main__":
    train_model()

