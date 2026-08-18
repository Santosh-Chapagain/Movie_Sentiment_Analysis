# data preprocessing

from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Handle missing values
    if text is None:
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    # Join back into a sentence
    return " ".join(tokens)


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('data loaded properly')

        # Transform the data
        train_data["review"] = train_data["review"].apply(preprocess_text)
        test_data["review"] = test_data["review"].apply(preprocess_text)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(
            data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(
            data_path, "test_processed.csv"), index=False)

        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error(
            'Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
