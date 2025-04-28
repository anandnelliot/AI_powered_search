# user_preprocess.py
import pandas as pd
import numpy as np
import os
import re
import logging
from typing import List, Dict, Any, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from logger.logger import get_logger

logger = get_logger(__file__)


def clean_text(text: str) -> str:
    """
    Clean text data by removing HTML tags and special characters, and normalizing whitespaces.
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Compile regex patterns for efficiency
    patterns = {
        'html': re.compile(r'<[^>]+>'),
        'special_chars': re.compile(r'[^\w\s,\.]'),  # Keep commas for CSV lists
        'whitespace': re.compile(r'\s+')
    }

    # Apply patterns in sequence
    text = patterns['html'].sub('', str(text))
    text = patterns['special_chars'].sub(' ', text)
    text = patterns['whitespace'].sub(' ', text).strip()
    return text.lower()

def preprocess_user_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the user data by cleaning text, handling missing values, and normalizing format."""
    
    # Create a copy of the DataFrame to avoid modifying the original data
    df = df.copy()
    
    # Define text columns that need cleaning
    text_columns = ['user_name', 'designation', 'supplier', 
                    'country', 'state', 'city']
    
    # Clean text columns
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
            logger.debug(f"Cleaned text in column: {col}")
    
    # Handle missing values with appropriate strategies
    # For text columns, replace NaN with empty string
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("")
    
    # For location columns, handle missing values
    location_columns = ['country', 'state', 'city']
    location_fill_values = {'country': '', 'state': '', 'city': ''}
    
    for col in location_columns:
        if col in df.columns:
            df[col] = df[col].fillna(location_fill_values.get(col, ''))
    
    # Convert text columns to lowercase for consistency
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower()
    
    return df

def save_processed_data(users_df: pd.DataFrame, base_path: str = './processed_data') -> None:
    """Save the processed user data to CSV files."""
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Save datasets
    users_df.to_csv(f"{base_path}/users.csv", index=False)
    
    logger.info(f"Saved processed data to {base_path}")
    logger.info(f"  - users.csv: {len(users_df)} rows")
    
# SKLearn Custom Transformers
class TextCleaner(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for cleaning text columns."""
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].apply(clean_text)
        return X_copy

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for handling missing values."""
    
    def __init__(self, text_columns=None, location_columns=None):
        self.text_columns = text_columns or []
        self.location_columns = location_columns or []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Handle text columns
        for col in self.text_columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna("")
        
        # Handle location columns
        location_fill_values = {'country': '', 'state': '', 'city': ''}
        for col in self.location_columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(location_fill_values.get(col, ''))
        
        return X_copy


def create_preprocessing_pipeline():
    """Create and return a full preprocessing pipeline for user data."""
    # Define column names
    text_columns = ['user_name', 'designation', 'supplier', 
                    'country', 'state', 'city']
    location_columns = ['country', 'state', 'city']
    
    # Create pipeline
    preprocessing_pipeline = Pipeline([
        ('text_cleaner', TextCleaner(columns=text_columns)),
        ('missing_handler', MissingValueHandler(text_columns=text_columns, location_columns=location_columns))
    ])
    
    return preprocessing_pipeline