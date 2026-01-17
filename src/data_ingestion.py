"""
Data Ingestion Script
Downloads the SMS spam dataset and saves it to data/raw folder
"""
import os
import pandas as pd
import requests
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
