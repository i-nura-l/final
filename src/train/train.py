"""
Main training script for sentiment analysis model.
Trains model, evaluates performance, and saves artifacts.
"""
import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loader import DataLoader
from preprocessing import TextPreprocessor, TextVectorizer
from evaluation import ModelEvaluator, compare_models


def prepare_data(data_path=None):
    """
    Load and prepare data for training.
    
    Args:
        data_path: Optional path to training data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor, vectorizer)
    """
    print("\n" + "=" * 50)
    print("Loading and Preparing Data")
    print("=" * 50)
    
    # Load data
    loader = DataLoader()
    df = loader.load_data(data_path=data_path, data_type="train")
    
    # Check required columns
    if 'text' not in df.columns:
        raise ValueError("Dataset must have 'text' column")
    
    if 'sentiment' not in df.columns:
        # Check for alternative column names
        for col in ['label', 'target', 'class']:
            if col in df.columns:
                df = df.rename(columns={col: 'sentiment'})
                break
    
    if 'sentiment' not in df.columns:
        raise ValueError("Dataset must have 'sentiment' or 'label' column")
    
    # Remove any missing values
    df = df.dropna(subset=['text', 'sentiment'])
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Preprocess text
    print("\nPreprocessing text...")
    preprocessor = TextPreprocessor(
        use_stemming=False,
        use_lemmatization=True,
        remove_stopwords=True,
        lowercase=True
    )
    
    df['processed_text'] = preprocessor.preprocess_corpus(df['text'])
    
    # Split data
    X = df['processed_text']
    y = df['sentiment']
    
    # Convert labels to binary if needed
    if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
        # Map pos/neg to 1/0
        label_map = {'pos': 1, 'neg': 0, 'positive': 1, 'negative': 0}
        y = y.map(label_map).astype(int)
        if y.isna().any():
            print("Warning: Some labels could not be mapped. Unique labels:", df['sentiment'].unique())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Vectorize text
    print("\nVectorizing text with TF-IDF...")
    vectorizer = TextVectorizer(method='tfidf', max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    
    return X_train_vec, X_test_vec, y_train, y_test, preprocessor, vectorizer


def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and compare performance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (best_model, best_model_name, all_results)
    """
    print("\n" + "=" * 50)
    print("Training Models")
    print("=" * 50)
    
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Linear SVM': LinearSVC(max_iter=1000, random_state=42)
    }
    
    results = {}
    trained_models = {}
    evaluator = ModelEvaluator()
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = evaluator.evaluate(y_test, y_pred, labels=[0, 1])
        results[name] = metrics
        trained_models[name] = model
        
        print(f"{name} - Accuracy: {metrics['accuracy']:.4f}")
    
    # Compare results
    print("\n")
    compare_models(results)
    
    # Select best model based on accuracy
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = trained_models[best_model_name]
    
    print(f"\n*** Selected Best Model: {best_model_name} ***")
    
    return best_model, best_model_name, results


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to training data CSV file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    models_dir = output_dir / 'models'
    predictions_dir = output_dir / 'predictions'
    figures_dir = output_dir / 'figures'
    
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor, vectorizer = prepare_data(args.data_path)
    
    # Train models
    best_model, best_model_name, all_results = train_models(X_train, y_train, X_test, y_test)
    
    # Final evaluation on best model
    print("\n" + "=" * 50)
    print("Final Model Evaluation")
    print("=" * 50)
    
    evaluator = ModelEvaluator()
    y_pred = best_model.predict(X_test)
    final_metrics = evaluator.evaluate(y_test, y_pred, labels=[0, 1])
    
    evaluator.print_metrics(final_metrics)
    
    # Save metrics
    metrics_path = predictions_dir / 'test_metrics.json'
    evaluator.save_metrics(metrics_path, final_metrics)
    
    # Generate confusion matrix
    cm_path = figures_dir / 'confusion_matrix.png'
    evaluator.plot_confusion_matrix(y_test, y_pred, 
                                   labels=['negative', 'positive'],
                                   save_path=cm_path)
    
    # Generate classification report
    report_path = predictions_dir / 'classification_report.txt'
    evaluator.generate_classification_report(y_test, y_pred,
                                            labels=['negative', 'positive'],
                                            save_path=report_path)
    
    # Save model artifacts
    print("\n" + "=" * 50)
    print("Saving Model Artifacts")
    print("=" * 50)
    
    model_path = models_dir / 'best_model.pkl'
    preprocessor_path = models_dir / 'preprocessor.pkl'
    vectorizer_path = models_dir / 'vectorizer.pkl'
    
    joblib.dump(best_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'metrics': final_metrics,
        'all_model_results': all_results
    }
    
    metadata_path = models_dir / 'model_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"\nFinal Test Accuracy: {final_metrics['accuracy']:.4f}")
    
    if final_metrics['accuracy'] >= 0.85:
        print("✓ Target accuracy threshold (0.85) achieved!")
    else:
        print("✗ Target accuracy threshold (0.85) not met.")
        print("  Consider: collecting more data, tuning hyperparameters, or trying different models.")


if __name__ == "__main__":
    main()
