"""
Machine Learning models for fall detection using spectrogram features.
Implements models as described in SAFE paper Section 4.2.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional
import joblib


class MLModelTrainer:
    """
    Trainer for various ML models on spectrogram features.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize trainer with random state for reproducibility."""
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        
    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of ML models to train.
        Returns models as specified in SAFE paper Section 4.2.
        """
        models = {
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            ),
            'LinearSVM': LinearSVC(
                random_state=self.random_state,
                max_iter=10000
            ),
            'SVMRBF': SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True
            ),
            'DecisionTree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        }
        return models
    
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        use_grid_search: bool = False
    ) -> Any:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Trained model
        """
        models = self.get_models()
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = models[model_name]
        
        if use_grid_search and model_name in ['LogisticRegression', 'LinearSVM']:
            # Grid search for linear models (as mentioned in SAFE paper)
            if model_name == 'LogisticRegression':
                param_grid = {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            else:  # LinearSVM
                param_grid = {
                    'C': [0.1, 1.0, 10.0, 100.0]
                }
            
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
        else:
            # Train without grid search
            model.fit(X_train, y_train)
        
        self.models[model_name] = model
        return model
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        use_grid_search: bool = False
    ) -> Dict[str, Any]:
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_grid_search: Whether to use grid search for linear models
            
        Returns:
            Dictionary of trained models
        """
        models = self.get_models()
        trained_models = {}
        
        print(f"Training {len(models)} models...")
        for model_name in models.keys():
            print(f"\nTraining {model_name}...")
            trained_model = self.train_model(
                model_name,
                X_train,
                y_train,
                use_grid_search=use_grid_search and model_name in ['LogisticRegression', 'LinearSVM']
            )
            trained_models[model_name] = trained_model
        
        return trained_models
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a model and return metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary')
        }
        
        return metrics
    
    def cross_validate(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> np.ndarray:
        """
        Perform cross-validation on a model.
        
        Args:
            model_name: Name of the model
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Array of cross-validation scores
        """
        models = self.get_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        return scores
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")
        
        joblib.dump(self.models[model_name], filepath)
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load a model from disk."""
        self.models[model_name] = joblib.load(filepath)
        print(f"Model {model_name} loaded from {filepath}")
