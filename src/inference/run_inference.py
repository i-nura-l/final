"""
Inference script for sentiment analysis model.
Loads trained model and generates predictions.
"""
import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loader import DataLoader
from evaluation import ModelEvaluator


def load_model_artifacts(models_dir):
    """
    Load trained model and preprocessing artifacts.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        tuple: (model, preprocessor, vectorizer)
    """
    print("\n" + "=" * 50)
    print("Loading Model Artifacts")
    print("=" * 50)
    
    models_path = Path(models_dir)
    
    model_path = models_path / 'best_model.pkl'
    preprocessor_path = models_path / 'preprocessor.pkl'
    vectorizer_path = models_path / 'vectorizer.pkl'
    
    # Check if files exist
    for path, name in [(model_path, 'Model'), 
                       (preprocessor_path, 'Preprocessor'), 
                       (vectorizer_path, 'Vectorizer')]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}")
    
    # Load artifacts
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    vectorizer = joblib.load(vectorizer_path)
    
    print(f"✓ Model loaded from: {model_path}")
    print(f"✓ Preprocessor loaded from: {preprocessor_path}")
    print(f"✓ Vectorizer loaded from: {vectorizer_path}")
    
    return model, preprocessor, vectorizer


def run_inference(model, preprocessor, vectorizer, data_path=None, has_labels=True):
    """
    Run inference on input data.
    
    Args:
        model: Trained model
        preprocessor: Text preprocessor
        vectorizer: Text vectorizer
        data_path: Optional path to input data
        has_labels: Whether input data has ground truth labels
        
    Returns:
        tuple: (predictions, input_data, metrics)
    """
    print("\n" + "=" * 50)
    print("Running Inference")
    print("=" * 50)
    
    # Load data
    loader = DataLoader()
    df = loader.load_data(data_path=data_path, data_type="inference")
    
    print(f"Loaded {len(df)} samples for inference")
    
    # Check if text column exists
    if 'text' not in df.columns:
        raise ValueError("Input data must have 'text' column")
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = preprocessor.preprocess_corpus(df['text'])
    
    # Vectorize
    print("Vectorizing text...")
    X = vectorizer.transform(df['processed_text'])
    
    # Predict
    print("Generating predictions...")
    predictions = model.predict(X)
    
    # Map predictions back to labels
    label_map = {0: 'neg', 1: 'pos'}
    df['predicted_sentiment'] = [label_map[pred] for pred in predictions]
    
    # Evaluate if labels are available
    metrics = None
    if has_labels and 'sentiment' in df.columns:
        print("\nEvaluating predictions...")
        
        # Convert labels
        true_labels = df['sentiment']
        if true_labels.dtype == 'object':
            reverse_label_map = {'neg': 0, 'pos': 1, 'negative': 0, 'positive': 1}
            true_labels = true_labels.map(reverse_label_map)
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(true_labels, predictions, labels=[0, 1])
        evaluator.print_metrics(metrics)
    
    return df, predictions, metrics


def save_predictions(df, predictions, metrics, output_dir):
    """
    Save predictions and metrics.
    
    Args:
        df: DataFrame with predictions
        predictions: Array of predictions
        metrics: Evaluation metrics (if available)
        output_dir: Directory to save outputs
    """
    print("\n" + "=" * 50)
    print("Saving Predictions")
    print("=" * 50)
    
    predictions_dir = Path(output_dir) / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_path = predictions_dir / 'predictions.csv'
    df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    # Save metrics if available
    if metrics:
        evaluator = ModelEvaluator()
        metrics_path = predictions_dir / 'inference_metrics.json'
        evaluator.save_metrics(metrics_path, metrics)


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(description='Run inference with trained sentiment model')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to input data CSV file')
    parser.add_argument('--model-dir', type=str, default='outputs/models',
                       help='Directory containing trained model')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save predictions')
    parser.add_argument('--has-labels', action='store_true', default=True,
                       help='Whether input data has ground truth labels')
    
    args = parser.parse_args()
    
    # Load model
    model, preprocessor, vectorizer = load_model_artifacts(args.model_dir)
    
    # Run inference
    df, predictions, metrics = run_inference(
        model, preprocessor, vectorizer, 
        data_path=args.data_path,
        has_labels=args.has_labels
    )
    
    # Save results
    save_predictions(df, predictions, metrics, args.output_dir)
    
    print("\n" + "=" * 50)
    print("Inference Complete!")
    print("=" * 50)
    
    if metrics:
        print(f"\nInference Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
