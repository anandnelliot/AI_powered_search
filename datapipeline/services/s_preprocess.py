#=============================================================================================#
                             # Import necessary libraries                
#=============================================================================================#
# Importing libraries
import pandas as pd
import numpy as np  
import os
import re       
from typing import List, Dict, Any, Tuple, Optional
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
    """Preprocess the data by cleaning text, handling missing values, and normalizing format."""

    # Create a copy of the DataFrame to avoid modifying the original data
    df = df.copy()

    #Define text columns that need cleaning
    text_columns = ['bicc_categoryname', 'bicsc_subcategoryname', 'service_name',
                    'MCSvD_DisplayName', 'MCSvD_ServDesc', 'bicsm_servicename']

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

def save_processed_data(services_df: pd.DataFrame, base_path: str = './processed_data') -> None:
    """Save the processed data to CSV files."""
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Save services dataset
    output_path = f"{base_path}/services.csv"
    services_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved processed data to {base_path}")
    logger.info(f"  - services.csv: {len(services_df)} rows")
    
    # Return the path for potential further processing
    return output_path

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
    
    def __init__(self, text_columns=None, location_columns=None, default_fill=""):
        self.text_columns = text_columns or []
        self.location_columns = location_columns or []
        self.default_fill = default_fill
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Handle text columns
        for col in self.text_columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(self.default_fill)
        
        # Handle location columns
        location_fill_values = {'CyM_CountryName': 'Unknown', 'SM_StateName': 'Unknown', 'CM_CityName': 'Unknown'}
        for col in self.location_columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(location_fill_values.get(col, 'Unknown'))
        
        # Fill remaining NaNs with default value if they're in object columns
        for col in X_copy.select_dtypes(include=['object']).columns:
            if col not in self.text_columns and col not in self.location_columns:
                X_copy[col] = X_copy[col].fillna(self.default_fill)
        
        return X_copy

class LowercaseConverter(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for converting text to lowercase."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.select_dtypes(include=['object']).columns:
            # Check if column contains string values before converting
            if X_copy[col].dtype == object and X_copy[col].notna().any():
                X_copy[col] = X_copy[col].astype(str).str.lower()
        return X_copy

class ColumnRenamer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for renaming dataframe columns."""
    
    def __init__(self, column_mapping):
        self.column_mapping = column_mapping
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        # Only rename columns that exist in the dataframe
        mapping = {k: v for k, v in self.column_mapping.items() if k in X_copy.columns}
        X_copy = X_copy.rename(columns=mapping)
        return X_copy

class StringLengthLimiter(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for limiting string lengths."""
    
    def __init__(self, column_limits: Dict[str, Dict[str, int]]):
        """
        Initialize with column limits.
        
        Args:
            column_limits: Dictionary mapping column names to their min/max length limits
                Example: {'column_name': {'min': 3, 'max': 100}}
        """
        self.column_limits = column_limits
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col, limits in self.column_limits.items():
            if col in X_copy.columns and X_copy[col].dtype == object:
                # Apply max length limit if specified
                if 'max' in limits:
                    X_copy[col] = X_copy[col].astype(str).apply(
                        lambda x: x[:limits['max']] if len(x) > limits['max'] else x
                    )
                
                # Apply min length padding if specified
                if 'min' in limits:
                    X_copy[col] = X_copy[col].astype(str).apply(
                        lambda x: x.ljust(limits['min']) if len(x) < limits['min'] and len(x) > 0 else x
                    )
        
        return X_copy

def get_column_mapping():
    """Returns the standard column mapping for renaming."""
    return {
        'service_id': 'service_id',
        'bicc_categoryname': 'category',
        'bicsc_subcategoryname': 'subcategory',
        'bicsm_servicename': 'subsubcategory',
        'MCSvD_DisplayName': 'service_name',
        'MCSvD_ServDesc': 'service_description',
        'MCM_CompanyName': 'supplier',
        'CyM_CountryName': 'country',
        'SM_StateName': 'state',
        'CM_CityName': 'city',
    }

def create_preprocessing_pipeline(string_length_limits: Optional[Dict] = None):
    """
    Create and return a full preprocessing pipeline.
    
    Args:
        string_length_limits: Optional dictionary of string length limits
    """
    # Define column names
    text_columns = ['bicc_categoryname', 'bicsc_subcategoryname', 'service_name', 
                   'MCSvD_DisplayName', 'MCSvD_ServDesc',
                   'bicsm_servicename', 'MCM_CompanyName']
    location_columns = ['CyM_CountryName', 'SM_StateName', 'CM_CityName']
    
    # Default string length limits if not provided
    if string_length_limits is None:
        string_length_limits = {
            'service_name': {'min': 3, 'max': 200},
            'service_description': {'min': 0, 'max': 5000},
            'category': {'min': 2, 'max': 100},
            'subcategory': {'min': 0, 'max': 100}
        }
    
    # Create pipeline
    preprocessing_pipeline = Pipeline([
        ('text_cleaner', TextCleaner(columns=text_columns)),
        ('missing_handler', MissingValueHandler(text_columns=text_columns, location_columns=location_columns, default_fill="")),
        ('lowercase_converter', LowercaseConverter()),
        ('column_renamer', ColumnRenamer(get_column_mapping())),
        ('string_limiter', StringLengthLimiter(string_length_limits))
    ])
    
    return preprocessing_pipeline