"""
Data Ingestion Script
Downloads the SMS spam dataset and saves it to data/raw folder
"""
import os
import pandas as pd
import requests
import logging
from pathlib import Path

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

logger = setup_logger(__name__, 'logs/data_ingestion.log')


def download_data(url: str, output_path: str) -> None:
    """
    Download data from URL and save to output path
    
    Args:
        url: URL to download data from
        output_path: Path to save the downloaded data
    """
    try:
        logger.info(f"Downloading data from {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the raw data
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Data successfully downloaded to {output_path}")
        
        # Load and display basic info
        df = pd.read_csv(output_path, encoding='latin-1')
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"\nFirst few rows:\n{df.head()}")
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


def main():
    """Main function to execute data ingestion"""
    # Dataset URL (raw GitHub URL)
    data_url = "https://raw.githubusercontent.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
    
    # Output path
    raw_data_path = "data/raw/spam.csv"
    
    # Download the data
    download_data(data_url, raw_data_path)
    
    logger.info("Data ingestion completed successfully!")


if __name__ == "__main__":
    main()
