"""
Model evaluation utilities for sentiment analysis.
"""
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluate model performance and save metrics."""
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.metrics = {}
    
    def evaluate(self, y_true, y_pred, labels=None):
        """
        Evaluate model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', pos_label=labels[1] if labels else 1),
            'recall': recall_score(y_true, y_pred, average='binary', pos_label=labels[1] if labels else 1),
            'f1': f1_score(y_true, y_pred, average='binary', pos_label=labels[1] if labels else 1)
        }
        
        self.metrics = metrics
        return metrics
    
    def print_metrics(self, metrics=None):
        """
        Print evaluation metrics.
        
        Args:
            metrics: Optional metrics dict, uses stored metrics if None
        """
        if metrics is None:
            metrics = self.metrics
        
        print("\n" + "=" * 50)
        print("Model Evaluation Metrics")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        print("=" * 50)
    
    def save_metrics(self, filepath, metrics=None):
        """
        Save metrics to file.
        
        Args:
            filepath: Path to save metrics
            metrics: Optional metrics dict, uses stored metrics if None
        """
        if metrics is None:
            metrics = self.metrics
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Also save as text for easy reading
        txt_filepath = filepath.replace('.json', '.txt')
        with open(txt_filepath, 'w') as f:
            f.write("Model Evaluation Metrics\n")
            f.write("=" * 50 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
            f.write("=" * 50 + "\n")
        
        print(f"Metrics saved to: {filepath}")
        print(f"Metrics saved to: {txt_filepath}")
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names for display
            save_path: Optional path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels if labels else ['Class 0', 'Class 1'],
                   yticklabels=labels if labels else ['Class 0', 'Class 1'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def generate_classification_report(self, y_true, y_pred, labels=None, save_path=None):
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            save_path: Optional path to save report
            
        Returns:
            str: Classification report
        """
        target_names = labels if labels else ['negative', 'positive']
        report = classification_report(y_true, y_pred, target_names=target_names)
        
        print("\nClassification Report:")
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("Classification Report\n")
                f.write("=" * 50 + "\n")
                f.write(report)
            print(f"Classification report saved to: {save_path}")
        
        return report


def compare_models(results_dict):
    """
    Compare multiple model results.
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
    """
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    
    for model_name, metrics in results_dict.items():
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f}")
    
    print("=" * 70)
    
    # Find best model
    best_model = max(results_dict.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
