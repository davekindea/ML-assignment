"""
Data Preprocessing Module for Classification Problem
Handles data loading, cleaning, and initial preparation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_data, RAW_DATA_DIR, PROCESSED_DATA_DIR, save_results

class DataPreprocessor:
    """
    Class for preprocessing classification data.
    """
    
    def __init__(self, data_path=None, target_column=None):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        data_path : str or Path, optional
            Path to the raw data file
        target_column : str, optional
            Name of the target column
        """
        self.data_path = data_path
        self.target_column = target_column
        self.df = None
        self.processed_df = None
        
    def load_raw_data(self, file_path=None, file_type='csv'):
        """
        Load raw data from file.
        
        Parameters:
        -----------
        file_path : str or Path, optional
            Path to data file (uses self.data_path if not provided)
        file_type : str
            Type of file ('csv', 'excel', 'json')
        """
        if file_path is None:
            file_path = self.data_path
        
        if file_path is None:
            raise ValueError("Please provide a data file path")
        
        print(f"Loading data from {file_path}...")
        self.df = load_data(file_path, file_type)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """
        Perform initial data exploration.
        """
        if self.df is None:
            raise ValueError("Please load data first using load_raw_data()")
        
        print("=" * 50)
        print("DATA EXPLORATION")
        print("=" * 50)
        
        print("\n1. Dataset Shape:")
        print(f"   Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        print("\n2. Column Names and Data Types:")
        print(self.df.dtypes)
        
        print("\n3. First Few Rows:")
        print(self.df.head())
        
        print("\n4. Dataset Info:")
        print(self.df.info())
        
        print("\n5. Missing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        print("\n6. Duplicate Rows:")
        print(f"   Number of duplicates: {self.df.duplicated().sum()}")
        
        print("\n7. Basic Statistics:")
        print(self.df.describe())
        
        if self.target_column and self.target_column in self.df.columns:
            print(f"\n8. Target Distribution ({self.target_column}):")
            print(self.df[self.target_column].value_counts())
            print(f"\n   Target distribution percentages:")
            print(self.df[self.target_column].value_counts(normalize=True) * 100)
    
    def handle_missing_values(self, strategy='auto', fill_value=None):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        strategy : str
            Strategy to handle missing values:
            - 'auto': Automatically choose based on data type
            - 'drop': Drop rows with missing values
            - 'fill': Fill with specified value
            - 'mean': Fill numerical with mean, categorical with mode
            - 'median': Fill numerical with median, categorical with mode
            - 'mode': Fill with mode
        fill_value : any, optional
            Value to fill missing values with (if strategy='fill')
        """
        if self.df is None:
            raise ValueError("Please load data first")
        
        print("\nHandling missing values...")
        missing_before = self.df.isnull().sum().sum()
        print(f"Missing values before: {missing_before}")
        
        self.processed_df = self.df.copy()
        
        if strategy == 'drop':
            self.processed_df = self.processed_df.dropna()
        elif strategy == 'auto':
            # Fill numerical columns with median, categorical with mode
            for col in self.processed_df.columns:
                if self.processed_df[col].isnull().sum() > 0:
                    if self.processed_df[col].dtype in ['int64', 'float64']:
                        self.processed_df[col].fillna(self.processed_df[col].median(), inplace=True)
                    else:
                        self.processed_df[col].fillna(self.processed_df[col].mode()[0] if len(self.processed_df[col].mode()) > 0 else 'Unknown', inplace=True)
        elif strategy == 'mean':
            for col in self.processed_df.select_dtypes(include=[np.number]).columns:
                self.processed_df[col].fillna(self.processed_df[col].mean(), inplace=True)
            for col in self.processed_df.select_dtypes(exclude=[np.number]).columns:
                mode_val = self.processed_df[col].mode()[0] if len(self.processed_df[col].mode()) > 0 else 'Unknown'
                self.processed_df[col].fillna(mode_val, inplace=True)
        elif strategy == 'median':
            for col in self.processed_df.select_dtypes(include=[np.number]).columns:
                self.processed_df[col].fillna(self.processed_df[col].median(), inplace=True)
            for col in self.processed_df.select_dtypes(exclude=[np.number]).columns:
                mode_val = self.processed_df[col].mode()[0] if len(self.processed_df[col].mode()) > 0 else 'Unknown'
                self.processed_df[col].fillna(mode_val, inplace=True)
        elif strategy == 'mode':
            for col in self.processed_df.columns:
                if self.processed_df[col].dtype in ['int64', 'float64']:
                    mode_val = self.processed_df[col].mode()[0] if len(self.processed_df[col].mode()) > 0 else 0
                else:
                    mode_val = self.processed_df[col].mode()[0] if len(self.processed_df[col].mode()) > 0 else 'Unknown'
                self.processed_df[col].fillna(mode_val, inplace=True)
        elif strategy == 'fill':
            if fill_value is None:
                raise ValueError("fill_value must be provided when strategy='fill'")
            self.processed_df = self.processed_df.fillna(fill_value)
        
        missing_after = self.processed_df.isnull().sum().sum()
        print(f"Missing values after: {missing_after}")
        print(f"Removed/Filled: {missing_before - missing_after} missing values")
    
    def remove_duplicates(self):
        """
        Remove duplicate rows from the dataset.
        """
        if self.processed_df is None:
            self.processed_df = self.df.copy()
        
        duplicates_before = self.processed_df.duplicated().sum()
        self.processed_df = self.processed_df.drop_duplicates()
        duplicates_after = self.processed_df.duplicated().sum()
        
        print(f"\nRemoved {duplicates_before - duplicates_after} duplicate rows")
    
    def handle_outliers(self, method='iqr', columns=None):
        """
        Handle outliers in numerical columns.
        
        Parameters:
        -----------
        method : str
            Method to handle outliers ('iqr', 'zscore', 'remove')
        columns : list, optional
            List of columns to process (all numerical if None)
        """
        if self.processed_df is None:
            self.processed_df = self.df.copy()
        
        if columns is None:
            columns = self.processed_df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column in columns:
                columns.remove(self.target_column)
        
        print(f"\nHandling outliers using {method} method...")
        
        for col in columns:
            if col not in self.processed_df.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.processed_df[col].quantile(0.25)
                Q3 = self.processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.processed_df[col] < lower_bound) | 
                           (self.processed_df[col] > upper_bound)).sum()
                print(f"  {col}: {outliers} outliers detected")
                
                if method == 'remove':
                    self.processed_df = self.processed_df[
                        (self.processed_df[col] >= lower_bound) & 
                        (self.processed_df[col] <= upper_bound)
                    ]
                else:  # cap
                    self.processed_df[col] = self.processed_df[col].clip(lower_bound, upper_bound)
            
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(self.processed_df[col]))
                outliers = (z_scores > 3).sum()
                print(f"  {col}: {outliers} outliers detected")
                
                if method == 'remove':
                    self.processed_df = self.processed_df[z_scores < 3]
                else:  # cap
                    threshold = 3
                    mean = self.processed_df[col].mean()
                    std = self.processed_df[col].std()
                    self.processed_df[col] = self.processed_df[col].clip(
                        mean - threshold * std, 
                        mean + threshold * std
                    )
    
    def save_processed_data(self, filename='processed_data.csv'):
        """
        Save processed data to file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save
        """
        if self.processed_df is None:
            raise ValueError("No processed data to save. Run preprocessing steps first.")
        
        filepath = PROCESSED_DATA_DIR / filename
        self.processed_df.to_csv(filepath, index=False)
        print(f"\nProcessed data saved to {filepath}")
        return filepath
    
    def get_processed_data(self):
        """
        Get the processed dataframe.
        
        Returns:
        --------
        pd.DataFrame
            Processed dataset
        """
        if self.processed_df is None:
            return self.df
        return self.processed_df


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("Please provide the dataset path and target column name")
    print("Example:")
    print("  preprocessor = DataPreprocessor(data_path='data/raw/dataset.csv', target_column='target')")
    print("  preprocessor.load_raw_data()")
    print("  preprocessor.explore_data()")
    print("  preprocessor.handle_missing_values()")
    print("  preprocessor.remove_duplicates()")
    print("  preprocessor.save_processed_data()")


