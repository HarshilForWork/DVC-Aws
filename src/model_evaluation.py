"""
Model Evaluation Script
Evaluates the trained model and logs metrics using DVC Live
"""
import os
import pickle
import logging
import yaml
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from dvclive import Live
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        logger.warning(f"Parameters file not found at {params_path}")
        return {}


def load_model(model_path: str):
    """
    Load trained model from file
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_features(data_dir: str):
    """
    Load test features from pickle files
    
    Args:
        data_dir: Directory containing feature files
        
    Returns:
        Tuple of (X_test, y_test)
    """
    logger.info(f"Loading test features from {data_dir}")
    
    with open(os.path.join(data_dir, 'X_test.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    
    with open(os.path.join(data_dir, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    
    logger.info(f"Loaded X_test shape: {X_test.shape}")
    logger.info(f"Loaded y_test shape: {y_test.shape}")
    
    return X_test, y_test


def evaluate_model(model, X_test, y_test, live):
    """
    Evaluate model and log metrics using DVC Live
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        live: DVC Live instance
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Log metrics with DVC Live
    live.log_metric("accuracy", accuracy)
    live.log_metric("precision", precision)
    live.log_metric("recall", recall)
    live.log_metric("f1_score", f1)
    live.log_metric("roc_auc", roc_auc)
    
    # Log confusion matrix values
    live.log_metric("true_negatives", int(cm[0, 0]))
    live.log_metric("false_positives", int(cm[0, 1]))
    live.log_metric("false_negatives", int(cm[1, 0]))
    live.log_metric("true_positives", int(cm[1, 1]))
    
    # Print metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['ham', 'spam'])}")
    
    # Create confusion matrix plot
    plot_confusion_matrix(cm, live)
    
    # Create ROC curve plot
    plot_roc_curve(y_test, y_pred_proba, roc_auc, live)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return metrics


def plot_confusion_matrix(cm, live):
    """
    Create and save confusion matrix plot
    
    Args:
        cm: Confusion matrix
        live: DVC Live instance
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['ham', 'spam']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Log plot with DVC Live
    live.log_image("confusion_matrix.png", plt.gcf())
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, roc_auc, live):
    """
    Create and save ROC curve plot
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        roc_auc: ROC AUC score
        live: DVC Live instance
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Log plot with DVC Live
    live.log_image("roc_curve.png", plt.gcf())
    plt.close()


def save_metrics(metrics: dict, output_path: str) -> None:
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save metrics
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {output_path}")


def main():
    """Main function to execute model evaluation"""
    # Paths
    model_path = "models/spam_classifier.pkl"
    data_dir = "data/processed"
    metrics_path = "metrics/metrics.json"
    
    # Load parameters
    params = load_params()
    
    # Load model
    model = load_model(model_path)
    
    # Load test features
    X_test, y_test = load_features(data_dir)
    
    # Initialize DVC Live
    with Live(save_dvc_exp=True) as live:
        # Log parameters
        if params:
            live.log_params(params)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, live)
    
    # Save metrics
    save_metrics(metrics, metrics_path)
    
    logger.info("Model evaluation completed successfully!")


if __name__ == "__main__":
    main()
