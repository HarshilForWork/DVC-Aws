"""
Data Preprocessing Script
Cleans and preprocesses the raw SMS spam dataset
"""
import os
import pandas as pd
import numpy as np
import re
import string
import logging
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Setup logging (CLI + File)
def setup_logger(name, log_file):
    """Setup logger with both console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler (CLI)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger(__name__, 'logs/preprocessing.log')


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load raw data from CSV file
    
    Args:
        input_path: Path to input CSV file
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, encoding='latin-1')
    logger.info(f"Loaded data shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing unnecessary columns and handling missing values
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    
    # Keep only the first two columns (v1 and v2)
    df = df.iloc[:, :2]
    df.columns = ['v1', 'v2']
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Remove missing values
    df = df.dropna()
    logger.info(f"Final dataset shape: {df.shape}")
    
    # Convert v1 to binary (ham=0, spam=1)
    df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
    
    logger.info(f"v1 distribution:\n{df['v1'].value_counts()}")
    
    return df


def preprocess_text(text: str) -> str:
    """
    Preprocess text by lowercasing, removing special characters, lemmatization
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text string
    """
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenize, lemmatize and rejoin
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(lemmatized_words)
    
    return text


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to all messages in the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with preprocessed messages
    """
    logger.info("Preprocessing text data...")
    df['processed_v2'] = df['v2'].apply(preprocess_text)
    
    # Remove empty messages after preprocessing
    df = df[df['processed_v2'].str.len() > 0]
    
    logger.info(f"Dataset shape after preprocessing: {df.shape}")
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save preprocessed data to CSV file
    
    Args:
        df: DataFrame to save
        output_path: Path to save the file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to {output_path}")


def main():
    """Main function to execute preprocessing"""
    # Input and output paths
    input_path = "data/raw/spam.csv"
    output_path = "data/interim/preprocessed_spam.csv"
    
    # Load data
    df = load_data(input_path)
    
    # Clean data
    df = clean_data(df)
    
    # Preprocess text
    df = preprocess_dataset(df)
    
    # Save preprocessed data
    save_data(df, output_path)
    
    logger.info("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
