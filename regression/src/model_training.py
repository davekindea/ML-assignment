"""
Model Training Module for Regression Problem
Handles model selection, training, and hyperparameter tuning.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import joblib
from pathlib import Path
from utils import SPLITS_DIR, MODELS_DIR, save_model

class ModelTrainer:
    """
    Class for training regression models.
    """
    
    def __init__(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        y : pd.Series or np.array
            Target variable
        test_size : float
            Proportion of data for test set
        val_size : float
            Proportion of training data for validation set
        random_state : int
            Random seed for reproducibility
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def split_data(self):
        """
        Split data into training, validation, and test sets.
        """
        print("Splitting data into train/validation/test sets...")
        
        # First split: separate test set
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Second split: separate train and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_val, self.y_train_val,
            test_size=self.val_size,
            random_state=self.random_state
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Save splits
        self._save_splits()
    
    def _save_splits(self):
        """Save data splits to files."""
        splits = {
            'X_train': self.X_train,
            'X_val': self.X_val,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test
        }
        
        for name, data in splits.items():
            filepath = SPLITS_DIR / f"{name}.csv"
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                pd.DataFrame(data).to_csv(filepath, index=False)
    
    def get_models(self):
        """
        Get dictionary of regression models to try.
        
        Returns:
        --------
        dict
            Dictionary of model names and instances
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=self.random_state),
            'Lasso Regression': Lasso(random_state=self.random_state),
            'Elastic Net': ElasticNet(random_state=self.random_state),
            'Random Forest': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=self.random_state),
            'XGBoost': XGBRegressor(random_state=self.random_state),
            'LightGBM': LGBMRegressor(random_state=self.random_state, verbose=-1),
            'SVR': SVR(),
            'K-Nearest Neighbors': KNeighborsRegressor(n_jobs=-1),
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            'AdaBoost': AdaBoostRegressor(random_state=self.random_state)
        }
        return models
    
    def train_baseline_models(self, models=None, cv=5):
        """
        Train multiple baseline models and compare performance.
        
        Parameters:
        -----------
        models : dict, optional
            Dictionary of models to train (uses get_models() if None)
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with model performance metrics
        """
        if self.X_train is None:
            self.split_data()
        
        if models is None:
            models = self.get_models()
        
        print("\n" + "="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)
        
        results = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Cross-validation score (using R²)
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=kf, scoring='r2', n_jobs=-1)
                
                # Validation score
                val_score = model.score(self.X_val, self.y_val)
                
                results.append({
                    'Model': name,
                    'CV Mean R²': cv_scores.mean(),
                    'CV Std': cv_scores.std(),
                    'Validation R²': val_score
                })
                
                self.models[name] = model
                
                print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                print(f"  Validation R²: {val_score:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Validation R²', ascending=False)
        
        print("\n" + "="*60)
        print("BASELINE MODEL COMPARISON")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Select best model
        self.best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        print(f"\nBest baseline model: {self.best_model_name}")
        
        return results_df
    
    def tune_hyperparameters(self, model_name, param_grid=None, method='grid', n_iter=50, cv=5):
        """
        Tune hyperparameters for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to tune
        param_grid : dict, optional
            Parameter grid for tuning (uses default if None)
        method : str
            Tuning method ('grid', 'random', 'optuna')
        n_iter : int
            Number of iterations for random/optuna search
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        sklearn model
            Best model with tuned hyperparameters
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train baseline models first.")
        
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print(f"{'='*60}")
        
        base_model = self.models[model_name]
        
        # Default parameter grids
        if param_grid is None:
            param_grid = self._get_default_param_grid(model_name)
        
        if method == 'grid':
            print("Using Grid Search...")
            search = GridSearchCV(
                base_model, param_grid, 
                cv=cv, scoring='r2', 
                n_jobs=-1, verbose=1
            )
            search.fit(self.X_train, self.y_train)
            best_model = search.best_estimator_
            print(f"\nBest parameters: {search.best_params_}")
            print(f"Best CV score: {search.best_score_:.4f}")
        
        elif method == 'random':
            print("Using Random Search...")
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=n_iter, cv=cv, scoring='r2',
                n_jobs=-1, random_state=self.random_state, verbose=1
            )
            search.fit(self.X_train, self.y_train)
            best_model = search.best_estimator_
            print(f"\nBest parameters: {search.best_params_}")
            print(f"Best CV score: {search.best_score_:.4f}")
        
        elif method == 'optuna':
            print("Using Optuna (Bayesian Optimization)...")
            best_model = self._optuna_tune(model_name, param_grid, n_trials=n_iter, cv=cv)
        
        # Update model
        self.models[f"{model_name}_tuned"] = best_model
        self.best_model = best_model
        self.best_model_name = f"{model_name}_tuned"
        
        # Evaluate on validation set
        val_score = best_model.score(self.X_val, self.y_val)
        print(f"Validation R²: {val_score:.4f}")
        
        return best_model
    
    def _get_default_param_grid(self, model_name):
        """Get default parameter grid for a model."""
        grids = {
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky']
            },
            'Lasso Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'max_iter': [1000, 2000]
            },
            'Elastic Net': {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 100]
            },
            'SVR': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        return grids.get(model_name, {})
    
    def _optuna_tune(self, model_name, param_grid, n_trials=50, cv=5):
        """Use Optuna for hyperparameter tuning."""
        def objective(trial):
            if model_name == 'Random Forest':
                model = RandomForestRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 5, 30),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif model_name == 'XGBoost':
                model = XGBRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    random_state=self.random_state
                )
            else:
                # Fallback to base model
                model = self.models[model_name]
            
            scores = cross_val_score(model, self.X_train, self.y_train, 
                                   cv=cv, scoring='r2', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Create best model
        best_params = study.best_params
        if model_name == 'Random Forest':
            best_model = RandomForestRegressor(**best_params, random_state=self.random_state, n_jobs=-1)
        elif model_name == 'XGBoost':
            best_model = XGBRegressor(**best_params, random_state=self.random_state)
        else:
            best_model = self.models[model_name]
        
        best_model.fit(self.X_train, self.y_train)
        print(f"\nBest parameters: {best_params}")
        print(f"Best CV score: {study.best_value:.4f}")
        
        return best_model
    
    def save_best_model(self, filename='best_regression_model.pkl'):
        """Save the best model to disk."""
        if self.best_model is None:
            raise ValueError("No best model to save. Train models first.")
        
        save_model(self.best_model, filename)
        print(f"Best model ({self.best_model_name}) saved successfully")

