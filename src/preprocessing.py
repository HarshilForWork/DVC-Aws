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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    
    # Keep only the first two columns (label and message)
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Remove missing values
    df = df.dropna()
    logger.info(f"Final dataset shape: {df.shape}")
    
    # Convert label to binary (ham=0, spam=1)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def preprocess_text(text: str) -> str:
    """
    Preprocess text by lowercasing, removing special characters and extra spaces
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text string
    """
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
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    # Remove empty messages after preprocessing
    df = df[df['processed_message'].str.len() > 0]
    
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
