# SMS Spam Classification MLOps Project

A complete MLOps project for SMS spam classification using DVC for data version control, DVC Live for experiment tracking, and AWS S3 for remote storage.

## Project Structure

```
DVC-aws/
├── data/
│   ├── raw/              # Raw data from source
│   ├── interim/          # Preprocessed data
│   └── processed/        # Feature-engineered data (train/test)
├── src/
│   ├── data_ingestion.py      # Download and load data
│   ├── preprocessing.py       # Clean and preprocess text
│   ├── feature_engineering.py # TF-IDF vectorization
│   ├── model_training.py      # Train ML models
│   └── model_evaluation.py    # Evaluate and log metrics
├── models/               # Trained models
├── metrics/              # Evaluation metrics
├── dvclive/              # DVC Live experiment logs
├── params.yaml           # Hyperparameters
├── dvc.yaml              # DVC pipeline definition
└── requirements.txt      # Python dependencies
```

## Dataset

The project uses the [SMS Spam Collection Dataset](https://github.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-) containing SMS messages labeled as spam or ham (legitimate).

## Features

- **Data Version Control**: Track data changes with DVC
- **Experiment Tracking**: Log metrics and plots with DVC Live
- **Pipeline Automation**: Reproducible ML pipeline with DVC
- **Remote Storage**: Store data and models on AWS S3
- **Feature Engineering**: TF-IDF vectorization with n-grams
- **Multiple Models**: Logistic Regression, Naive Bayes, Random Forest, SVM

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize DVC

```bash
dvc init
```

### 3. Configure AWS S3 Remote Storage

```bash
# Add S3 remote
dvc remote add -d myremote s3://your-bucket-name/dvc-storage

# Configure AWS credentials
dvc remote modify myremote access_key_id YOUR_AWS_ACCESS_KEY
dvc remote modify myremote secret_access_key YOUR_AWS_SECRET_KEY

# Or use AWS credentials from environment/config
dvc remote modify myremote profile YOUR_AWS_PROFILE
```

### 4. Run the Pipeline

```bash
# Run the complete pipeline
dvc repro

# Or run individual stages
dvc repro data_ingestion
dvc repro preprocessing
dvc repro feature_engineering
dvc repro model_training
dvc repro model_evaluation
```

### 5. Push Data to S3

```bash
# Push data and models to S3
dvc push
```

## Pipeline Stages

### 1. Data Ingestion
- Downloads SMS spam dataset
- Saves raw data to `data/raw/spam.csv`

### 2. Preprocessing
- Cleans text data (lowercase, remove punctuation, URLs, etc.)
- Removes duplicates and missing values
- Converts labels to binary (ham=0, spam=1)
- Saves to `data/interim/preprocessed_spam.csv`

### 3. Feature Engineering
- Applies TF-IDF vectorization
- Creates unigrams and bigrams
- Splits data into train/test sets
- Saves features to `data/processed/`

### 4. Model Training
- Trains classification model (configurable in `params.yaml`)
- Saves trained model to `models/spam_classifier.pkl`

### 5. Model Evaluation
- Evaluates model on test set
- Logs metrics with DVC Live (accuracy, precision, recall, F1, ROC-AUC)
- Creates confusion matrix and ROC curve plots
- Saves metrics to `metrics/metrics.json`

## Experiment Tracking

View experiment results:

```bash
# Show metrics
dvc metrics show

# Show plots
dvc plots show

# Compare experiments
dvc exp show
```

## Modifying Parameters

Edit `params.yaml` to change:
- Model type (logistic_regression, naive_bayes, random_forest, svm)
- Feature engineering parameters (max_features, ngram_range, min_df)
- Train/test split ratio
- Model hyperparameters

Then run:
```bash
dvc repro
```

## AWS S3 Setup Guide

### Create S3 Bucket

1. Go to AWS S3 Console
2. Create a new bucket (e.g., `my-mlops-dvc-bucket`)
3. Note the bucket name and region

### Configure AWS Credentials

Option 1: AWS CLI
```bash
aws configure
```

Option 2: Environment Variables
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

Option 3: IAM Role (for EC2)
- Attach IAM role with S3 access to your EC2 instance

## Pull Data from S3

To reproduce the project on another machine:

```bash
git clone <repository>
cd DVC-aws
pip install -r requirements.txt
dvc pull
dvc repro
```

## License

MIT License

## Author

Your Name
