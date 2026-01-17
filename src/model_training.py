"""
Model Training Script
Trains classification models for spam detection
"""
import os
import pickle
import logging
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

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

logger = setup_logger(__name__, 'logs/model_training.log')


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
            'model': {
                'type': 'logistic_regression',
                'C': 1.0,
                'max_iter': 1000
            }
        }


def load_features(data_dir: str):
    """
    Load features from pickle files
    
    Args:
        data_dir: Directory containing feature files
        
    Returns:
        Tuple of (X_train, y_train)
    """
    logger.info(f"Loading features from {data_dir}")
    
    with open(os.path.join(data_dir, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    
    with open(os.path.join(data_dir, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    
    logger.info(f"Loaded X_train shape: {X_train.shape}")
    logger.info(f"Loaded y_train shape: {y_train.shape}")
    
    return X_train, y_train


def get_model(params: dict):
    """
    Initialize model based on parameters
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Initialized model
    """
    model_params = params.get('model', {})
    model_type = model_params.get('type', 'logistic_regression')
    
    logger.info(f"Initializing {model_type} model")
    
    if model_type == 'logistic_regression':
        model = LogisticRegression(
            C=model_params.get('C', 1.0),
            max_iter=model_params.get('max_iter', 1000),
            random_state=42
        )
    elif model_type == 'naive_bayes':
        model = MultinomialNB(
            alpha=model_params.get('alpha', 1.0)
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', None),
            random_state=42
        )
    elif model_type == 'svm':
        model = SVC(
            C=model_params.get('C', 1.0),
            kernel=model_params.get('kernel', 'rbf'),
            random_state=42,
            probability=True
        )
    else:
        logger.warning(f"Unknown model type: {model_type}, using Logistic Regression")
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    
    return model


def train_model(model, X_train, y_train):
    """
    Train the model
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    # Training accuracy
    train_score = model.score(X_train, y_train)
    logger.info(f"Training accuracy: {train_score:.4f}")
    
    return model


def save_model(model, output_path: str) -> None:
    """
    Save trained model to file
    
    Args:
        model: Trained model
        output_path: Path to save the model
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {output_path}")


def main():
    """Main function to execute model training"""
    # Paths
    data_dir = "data/processed"
    model_path = "models/spam_classifier.pkl"
    
    # Load parameters
    params = load_params()
    
    # Load features
    X_train, y_train = load_features(data_dir)
    
    # Initialize model
    model = get_model(params)
    
    # Train model
    model = train_model(model, X_train, y_train)
    
    # Save model
    save_model(model, model_path)
    
    logger.info("Model training pipeline completed successfully!")


if __name__ == "__main__":
    main()
