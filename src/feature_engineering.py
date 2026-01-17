"""
Feature Engineering Script
Applies TF-IDF vectorization to create features from text data
"""
import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml

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

logger = setup_logger(__name__, 'logs/feature_engineering.log')


def load_params(params_path: str = "params.yaml") -> dict:
    """
    Load parameters from YAML file
    
    Args:
        params_path: Path to parameters file
        
    Returns:
        Dictionary of parameters
    """
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        return params
    except FileNotFoundError:
        logger.warning(f"Parameters file not found at {params_path}, using defaults")
        return {
            'feature_engineering': {
                'max_features': 3000,
                'ngram_range': [1, 2],
                'min_df': 2
            },
            'split': {
                'test_size': 0.2,
                'random_state': 42
            }
        }


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load preprocessed data from CSV file
    
    Args:
        input_path: Path to input CSV file
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded data shape: {df.shape}")
    return df


def create_tfidf_features(df: pd.DataFrame, params: dict):
    """
    Create TF-IDF features from text data
    
    Args:
        df: Input DataFrame
        params: Dictionary of parameters
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, vectorizer)
    """
    logger.info("Creating TF-IDF features...")
    
    # Extract parameters
    fe_params = params.get('feature_engineering', {})
    split_params = params.get('split', {})
    
    max_features = fe_params.get('max_features', 3000)
    ngram_range = tuple(fe_params.get('ngram_range', [1, 2]))
    min_df = fe_params.get('min_df', 2)
    
    test_size = split_params.get('test_size', 0.2)
    random_state = split_params.get('random_state', 42)
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words='english'
    )
    
    # Prepare features and labels
    X = df['processed_v2']
    y = df['v1']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Train label distribution:\n{y_train.value_counts()}")
    logger.info(f"Test label distribution:\n{y_test.value_counts()}")
    
    # Fit and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    logger.info(f"TF-IDF feature shape: {X_train_tfidf.shape}")
    logger.info(f"Number of features: {len(vectorizer.get_feature_names_out())}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


def save_features(X_train, X_test, y_train, y_test, vectorizer, output_dir: str) -> None:
    """
    Save features and vectorizer to files
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        vectorizer: Fitted TF-IDF vectorizer
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features as pickle files
    with open(os.path.join(output_dir, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    
    with open(os.path.join(output_dir, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    
    with open(os.path.join(output_dir, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    
    with open(os.path.join(output_dir, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    
    # Save vectorizer
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    logger.info(f"Features and vectorizer saved to {output_dir}")


def main():
    """Main function to execute feature engineering"""
    # Input and output paths
    input_path = "data/interim/preprocessed_spam.csv"
    output_dir = "data/processed"
    
    # Load parameters
    params = load_params()
    
    # Load data
    df = load_data(input_path)
    
    # Create TF-IDF features
    X_train, X_test, y_train, y_test, vectorizer = create_tfidf_features(df, params)
    
    # Save features
    save_features(X_train, X_test, y_train, y_test, vectorizer, output_dir)
    
    logger.info("Feature engineering completed successfully!")


if __name__ == "__main__":
    main()
