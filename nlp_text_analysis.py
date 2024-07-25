import os
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import load_data
import config

# Download necessary NLTK data (uncomment if running for the first time)
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        np.random.seed(config.RANDOM_SEED)

    def preprocess_text(self, text):
        """Preprocess text data: remove non-alphabetical characters, lowercase, tokenize, lemmatize, remove stopwords."""
        text = re.sub(r'[^a-zA-Z0-9]+', ' ', text).lower()
        word_tokens = word_tokenize(text)
        lem = [self.lemmatizer.lemmatize(word) for word in word_tokens if word not in self.stop_words]
        return ' '.join(lem)

class DataHandler:
    def __init__(self, text_processor):
        self.text_processor = text_processor

    def load_and_preprocess_data(self, real_path, fake_path):
        """Load, concatenate, and preprocess data."""
        try:
            df_real = load_data(real_path)
            df_fake = load_data(fake_path)
            
            df_real['label'] = 0
            df_fake['label'] = 1
            df = pd.concat([df_real, df_fake], axis=0).sample(frac=1).reset_index(drop=True)
            df['text'] = df['text'].apply(self.text_processor.preprocess_text)
            return df
        except Exception as e:
            print(f"Error loading and preprocessing data: {e}")
            return None

    def train_valid_test_split(self, X, y, train_size=0.7, valid_size=0.15, test_size=0.15):
        """Split data into train, validation, and test sets."""
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(valid_size + test_size), random_state=config.RANDOM_SEED)
        ratio = valid_size / (valid_size + test_size)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=(1.0 - ratio), random_state=config.RANDOM_SEED)
        return X_train, X_valid, X_test, y_train, y_valid, y_test

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def vectorize_data(self, X_train, X_valid, X_test):
        """Vectorize the text data using TF-IDF."""
        xv_train = self.vectorizer.fit_transform(X_train)
        xv_valid = self.vectorizer.transform(X_valid)
        xv_test = self.vectorizer.transform(X_test)
        return xv_train, xv_valid, xv_test, self.vectorizer

# Usage Example
if __name__ == "__main__":
    text_processor = TextProcessor()
    data_handler = DataHandler(text_processor)
    vectorizer = Vectorizer()

    df = data_handler.load_and_preprocess_data("path/to/real", "path/to/fake")
    if df is not None:
        X = df['text']
        y = df['label']
        X_train, X_valid, X_test, y_train, y_valid, y_test = data_handler.train_valid_test_split(X, y)
        xv_train, xv_valid, xv_test, vectorizer_model = vectorizer.vectorize_data(X_train, X_valid, X_test)
