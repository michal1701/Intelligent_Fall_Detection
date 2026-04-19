"""
Evaluation metrics and result reporting for SAFE dataset experiments.
Implements evaluation as described in SAFE paper Section 5.2.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ResultEvaluator:
    """Evaluate and report results for ML models on SAFE dataset."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
        
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_type: str = "mel_spectrogram"
    ) -> Dict[str, float]:
        """
        Evaluate a single model and return metrics.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            feature_type: Type of features used
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'model': model_name,
            'feature_type': feature_type,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['true_positives'] = int(cm[1, 1])
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: Optional[str] = None
    ):
        """
        Plot and optionally save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Non-Fall', 'Fall'],
            yticklabels=['Non-Fall', 'Fall']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def create_results_table(self, results_list: List[Dict[str, float]]) -> pd.DataFrame:
        """
        Create a results table from evaluation results.
        
        Args:
            results_list: List of result dictionaries
            
        Returns:
            DataFrame with results
        """
        df = pd.DataFrame(results_list)
        
        # Sort by accuracy (descending)
        df = df.sort_values('accuracy', ascending=False)
        
        return df
    
    def print_results_summary(self, results_df: pd.DataFrame):
        """
        Print a formatted summary of results.
        
        Args:
            results_df: DataFrame with results
        """
        print("\n" + "="*80)
        print("SAFE Dataset - Machine Learning Results on Spectrogram Features")
        print("="*80)
        print(f"\n{'Model':<20} {'Feature':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*80)
        
        for _, row in results_df.iterrows():
            print(f"{row['model']:<20} {row['feature_type']:<20} "
                  f"{row['accuracy']*100:.2f}%      {row['precision']*100:.2f}%      "
                  f"{row['recall']*100:.2f}%      {row['f1_score']*100:.2f}%")
        
        print("="*80)
        
        # Print best model
        best_row = results_df.iloc[0]
        print(f"\nBest Model: {best_row['model']} ({best_row['feature_type']})")
        print(f"  Accuracy: {best_row['accuracy']*100:.2f}%")
        print(f"  F1-Score: {best_row['f1_score']*100:.2f}%")
        print("="*80)
    
    def save_results(self, results_df: pd.DataFrame, filepath: str):
        """
        Save results to CSV file.
        
        Args:
            results_df: DataFrame with results
            filepath: Path to save CSV file
        """
        results_df.to_csv(filepath, index=False)
        print(f"\nResults saved to {filepath}")
    
    def plot_results_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = 'accuracy',
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of models by metric.
        
        Args:
            results_df: DataFrame with results
            metric: Metric to plot ('accuracy', 'precision', 'recall', 'f1_score')
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 6))
        
        # Group by feature type if multiple feature types exist
        if 'feature_type' in results_df.columns:
            for feature_type in results_df['feature_type'].unique():
                df_feat = results_df[results_df['feature_type'] == feature_type]
                plt.bar(
                    df_feat['model'],
                    df_feat[metric],
                    label=feature_type,
                    alpha=0.7
                )
            plt.legend()
        else:
            plt.bar(results_df['model'], results_df[metric])
        
        plt.title(f'Model Comparison - {metric.capitalize()}')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.close()
