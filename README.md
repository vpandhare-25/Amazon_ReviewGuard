# Amazon Review Analyzer

A machine learning pipeline that detects and classifies Amazon product reviews using NLP, feature engineering, and ensemble learning techniques. The project processes thousands of reviews, extracts meaningful textual features, and predicts review sentiment with high accuracy.

## Overview

This project was built to explore how machine learning can be applied to large-scale customer feedback data. The system performs data preprocessing, feature extraction, model training, hyperparameter tuning, and evaluation to generate reliable sentiment predictions.

The trained model can classify unseen reviews and provide confidence scores for predictions.

## Features

* Review sentiment classification
* Automated text preprocessing pipeline
* Feature engineering and extraction
* Hyperparameter tuning workflow
* Model persistence with Joblib
* Evaluation and testing utilities
* Metadata and feature tracking for reproducibility

## Tech Stack

### Languages

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib

### Concepts

* Natural Language Processing (NLP)
* Sentiment Analysis
* Feature Engineering
* Ensemble Learning
* Hyperparameter Optimization
* Model Evaluation

## Dataset

The project uses Amazon review datasets containing customer reviews and associated sentiment labels.

Preprocessing steps include:

* Data cleaning
* Missing value handling
* Text normalization
* Feature extraction
* Dataset transformation

## Project Structure

```text
Amazon-Review-Analyzer/
│
├── .vscode/
│   └── settings.json
│
├── model/
│   ├── review_classifier.pkl
│   ├── feature_names.json
│   └── model_metadata.json
│
├── src/
│
├── test/
│
├── train_model.py
├── tune_model.py
├── finaltest.py
├── eda_starter.py
│
├── fake-reviews.csv
├── processed-dataset.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Model Training

The training pipeline:

1. Loads and preprocesses review data
2. Extracts numerical text features
3. Splits data into training and testing sets
4. Trains machine learning classifiers
5. Evaluates performance using multiple metrics
6. Saves the best-performing model

Run training:

```bash
python train_model.py
```

## Hyperparameter Tuning

To optimize model performance:

```bash
python tune_model.py
```

This performs parameter search and identifies the strongest-performing configuration.

## Testing

Evaluate the trained model:

```bash
python finaltest.py
```

## Exploratory Data Analysis

Analyze review distributions and dataset characteristics:

```bash
python eda_starter.py
```

## Results

* Processed 30,000+ Amazon reviews
* Achieved approximately 92% classification accuracy
* Generated detailed evaluation metrics including:

  * Precision
  * Recall
  * F1 Score
  * Confusion Matrix
  * ROC-AUC

## Future Improvements

* Transformer-based models (BERT)
* Real-time inference API
* Interactive dashboard for predictions
* Explainable AI visualizations
* Multi-class sentiment analysis
* Cloud deployment

## What I Learned

Through this project I gained hands-on experience with machine learning workflows, NLP preprocessing, feature engineering, model evaluation, hyperparameter tuning, and building reproducible data science pipelines for real-world text classification problems.
