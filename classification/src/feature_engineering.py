"""
Feature Engineering Module for Classification Problem
Handles feature creation, encoding, scaling, and selection.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.decomposition import PCA
from pathlib import Path
import joblib
from utils import PROCESSED_DATA_DIR, MODELS_DIR

class FeatureEngineer:
    """
    Class for feature engineering operations.
    """
    
    def __init__(self, target_column=None):
        """
        Initialize the feature engineer.
        
        Parameters:
        -----------
        target_column : str, optional
            Name of the target column
        """
        self.target_column = target_column
        self.scaler = None
        self.encoders = {}
        self.feature_selector = None
        self.pca = None
        self.feature_names = None
        
    def encode_categorical(self, df, columns=None, method='auto'):
        """
        Encode categorical variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list, optional
            List of columns to encode (all categorical if None)
        method : str
            Encoding method ('auto', 'label', 'onehot', 'target')
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        if columns is None:
            # Automatically detect categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if self.target_column in categorical_cols:
                categorical_cols.remove(self.target_column)
        else:
            categorical_cols = columns
        
        print(f"\nEncoding {len(categorical_cols)} categorical columns using {method} method...")
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            if method == 'auto':
                # Use label encoding for binary, one-hot for multi-class
                unique_count = df[col].nunique()
                if unique_count == 2:
                    method_col = 'label'
                else:
                    method_col = 'onehot' if unique_count <= 10 else 'label'
            else:
                method_col = method
            
            if method_col == 'label':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                print(f"  {col}: Label encoded ({unique_count} unique values)")
            
            elif method_col == 'onehot':
                # Create one-hot encoded columns
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                print(f"  {col}: One-hot encoded ({unique_count} unique values -> {len(dummies.columns)} columns)")
            
            elif method_col == 'target':
                # Target encoding (mean encoding)
                if self.target_column is None:
                    raise ValueError("Target column must be specified for target encoding")
                target_mean = df.groupby(col)[self.target_column].mean()
                df_encoded[col] = df[col].map(target_mean)
                print(f"  {col}: Target encoded")
        
        return df_encoded
    
    def scale_features(self, df, columns=None, method='standard', fit=True):
        """
        Scale numerical features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list, optional
            List of columns to scale (all numerical if None)
        method : str
            Scaling method ('standard', 'minmax', 'robust')
        fit : bool
            Whether to fit the scaler (True for training, False for testing)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if columns is None:
            # Automatically detect numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column in numerical_cols:
                numerical_cols.remove(self.target_column)
        else:
            numerical_cols = columns
        
        if len(numerical_cols) == 0:
            print("No numerical columns to scale")
            return df_scaled
        
        print(f"\nScaling {len(numerical_cols)} numerical columns using {method} method...")
        
        if method == 'standard':
            if fit or self.scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(df[numerical_cols])
            df_scaled[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        elif method == 'minmax':
            if fit or self.scaler is None:
                self.scaler = MinMaxScaler()
                self.scaler.fit(df[numerical_cols])
            df_scaled[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        elif method == 'robust':
            if fit or self.scaler is None:
                self.scaler = RobustScaler()
                self.scaler.fit(df[numerical_cols])
            df_scaled[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df_scaled
    
    def create_interaction_features(self, df, columns=None, max_interactions=10):
        """
        Create interaction features between numerical columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list, optional
            List of columns to create interactions for
        max_interactions : int
            Maximum number of interaction features to create
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with interaction features
        """
        df_interactions = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column in columns:
                columns.remove(self.target_column)
        
        print(f"\nCreating interaction features from {len(columns)} columns...")
        
        interaction_count = 0
        for i, col1 in enumerate(columns):
            if interaction_count >= max_interactions:
                break
            for col2 in columns[i+1:]:
                if interaction_count >= max_interactions:
                    break
                interaction_name = f"{col1}_x_{col2}"
                df_interactions[interaction_name] = df[col1] * df[col2]
                interaction_count += 1
        
        print(f"Created {interaction_count} interaction features")
        return df_interactions
    
    def create_polynomial_features(self, df, columns=None, degree=2, max_features=10):
        """
        Create polynomial features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list, optional
            List of columns to create polynomial features for
        degree : int
            Degree of polynomial
        max_features : int
            Maximum number of polynomial features to create
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column in columns:
                columns.remove(self.target_column)
        
        print(f"\nCreating polynomial features (degree={degree}) from {len(columns)} columns...")
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df[columns])
        
        # Create feature names
        feature_names = poly.get_feature_names_out(columns)
        
        # Limit to max_features
        if len(feature_names) > max_features:
            # Keep original features and top interactions
            selected_indices = list(range(len(columns)))  # Original features
            remaining = len(feature_names) - len(columns)
            if remaining > 0:
                # Add some polynomial features (simplified selection)
                selected_indices.extend(range(len(columns), min(len(columns) + max_features - len(columns), len(feature_names))))
            poly_features = poly_features[:, selected_indices]
            feature_names = feature_names[selected_indices]
        
        # Create dataframe with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Remove original columns and add polynomial features
        df_poly = df.drop(columns=columns)
        df_poly = pd.concat([df_poly, poly_df], axis=1)
        
        print(f"Created {len(feature_names)} polynomial features")
        return df_poly
    
    def select_features(self, X, y, method='mutual_info', k=10):
        """
        Select top k features using various methods.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        y : pd.Series or np.array
            Target variable
        method : str
            Feature selection method ('mutual_info', 'chi2', 'f_classif')
        k : int
            Number of features to select
        
        Returns:
        --------
        pd.DataFrame or np.array
            Selected features
        """
        print(f"\nSelecting top {k} features using {method} method...")
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        elif method == 'chi2':
            selector = SelectKBest(score_func=chi2, k=min(k, X.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        self.feature_selector = selector
        
        # Get selected feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            selected_features = X.columns[selector.get_support()].tolist()
            print(f"Selected features: {selected_features}")
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return X_selected
    
    def apply_pca(self, X, n_components=None, variance_threshold=0.95):
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        n_components : int, optional
            Number of components (if None, use variance_threshold)
        variance_threshold : float
            Cumulative variance threshold (used if n_components is None)
        
        Returns:
        --------
        pd.DataFrame or np.array
            Transformed features
        """
        print(f"\nApplying PCA...")
        
        if n_components is None:
            # Find number of components that explain variance_threshold variance
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            print(f"  Selected {n_components} components to explain {variance_threshold*100}% variance")
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        explained_variance = self.pca.explained_variance_ratio_.sum()
        print(f"  Explained variance: {explained_variance*100:.2f}%")
        
        if isinstance(X, pd.DataFrame):
            pca_columns = [f'PC{i+1}' for i in range(n_components)]
            X_pca = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        return X_pca
    
    def save_scaler(self, filename='scaler.pkl'):
        """Save the fitted scaler."""
        if self.scaler is not None:
            filepath = MODELS_DIR / filename
            joblib.dump(self.scaler, filepath)
            print(f"Scaler saved to {filepath}")
    
    def save_encoders(self, filename='encoders.pkl'):
        """Save the fitted encoders."""
        if self.encoders:
            filepath = MODELS_DIR / filename
            joblib.dump(self.encoders, filepath)
            print(f"Encoders saved to {filepath}")


