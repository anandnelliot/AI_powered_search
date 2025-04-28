#=============================================================================================#
                             # Import necessary libraries                
#=============================================================================================#
# Importing libraries
import pandas as pd
import numpy as np  
import os
import re       
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
    #compile regex patterns for efficiency
        # Remove HTML tags
    patterns = {
        'html' : re.compile(r'<[^>]+>'),
        'special_chars' : re.compile(r'[^\w\s]'),
        'whitespace' : re.compile(r'\s+')
    }

    #Apply patterns in sequence
    text = patterns['html'].sub('', str(text))
    text = patterns['special_chars'].sub(' ', text)
    text = patterns['whitespace'].sub(' ', text).strip()
    return text.lower()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by cleaning text, handling missing values, and normaliizing format."""

    # Create a copy of the DataFrame to avoid modifying the original data
    df = df.copy()

    #Define text columns that need cleaning
    text_columns = ['bicc_categoryname', 'bicsc_subcategoryname', 'product_name',
                    'MCPrD_DisplayName', 'description', 'bicpm_productname']

    # Clean text columns
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
            logger.debug(f"Cleaned text in column: {col}") 

    # Handle missing values with appropriate strategies
    # For text columns, replace NaN with empty string
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("0")

    # For location columns, handle missing values
    location_columns = ['CyM_CountryName', 'SM_StateName', 'CM_CityName']
    location_fill_values = {'CyM_CountryName': 'Unknown', 'SM_StateName': 'Unknown', 'CM_CityName': 'Unknown'}

    for col in location_columns:
        if col in df.columns:
            df[col] = df[col].fillna(location_fill_values.get(col, 'Unknown'))
    
    # Convert text columns to lowercase for consistency
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower()
    
    logger.info(f"Preprocessing complete. Processed {len(df)} rows and {len(df.columns)} columns.")
    return df

def save_processed_data(products_df: pd.DataFrame, base_path: str = './processed_data') -> None:
    """Save the processed data to CSV files."""
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Save datasets
    products_df.to_csv(f"{base_path}/products.csv", index=False)
    
    logger.info(f"Saved processed data to {base_path}")
    logger.info(f"  - products.csv: {len(products_df)} rows")

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
        location_fill_values = {'CyM_CountryName': 'Unknown', 'SM_StateName': 'Unknown', 'CM_CityName': 'Unknown'}
        for col in self.location_columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(location_fill_values.get(col, 'Unknown'))
        
        return X_copy

class LowercaseConverter(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for converting text to lowercase."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.select_dtypes(include=['object']).columns:
            X_copy[col] = X_copy[col].str.lower()
        return X_copy

class ColumnRenamer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for renaming dataframe columns."""
    
    def __init__(self, column_mapping):
        self.column_mapping = column_mapping
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.rename(columns=self.column_mapping)
        return X_copy

def get_column_mapping():
    """Returns the standard column mapping for renaming."""
    return {
        'bicc_categoryname': 'category',
        'bicsc_subcategoryname': 'subcategory',
        'bicpm_productname': 'subsubcategory',
        'MCPrD_DisplayName': 'product_name',
        'MCPrD_ProdDesc': 'product_description',
        'MCM_CompanyName': 'supplier',
        'CyM_CountryName': 'country',
        'SM_StateName': 'state',
        'CM_CityName': 'city'
    }

def create_preprocessing_pipeline():
    """Create and return a full preprocessing pipeline."""
    # Define column names
    text_columns = ['bicc_categoryname', 'bicsc_subcategoryname', 'product_name', 
                   'MCPrD_DisplayName', 'description', 'MCM_CompanyName', 'bicpm_productname']
    location_columns = ['CyM_CountryName', 'SM_StateName', 'CM_CityName']
    
    # Create pipeline
    preprocessing_pipeline = Pipeline([
        ('text_cleaner', TextCleaner(columns=text_columns)),
        ('missing_handler', MissingValueHandler(text_columns=text_columns, location_columns=location_columns)),
        ('lowercase_converter', LowercaseConverter()),
        ('column_renamer', ColumnRenamer(get_column_mapping()))
    ])
    
    return preprocessing_pipeline
