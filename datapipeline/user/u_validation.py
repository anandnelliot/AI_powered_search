# user_validation.py
import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, List, Any

from logger.logger import get_logger

logger = get_logger(__file__)


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the user data against defined rules and constraints.
    Returns a dictionary with validation results.
    """
    errors = []
    warnings = []
    
    # Check required fields
    required_fields = ['user_id', 'login', 'user_name', 'email_id']
    for field in required_fields:
        if field not in df.columns:
            errors.append(f"Required field '{field}' is missing from the dataset")
        elif df[field].isna().any():
            num_missing = df[field].isna().sum()
            errors.append(f"Required field '{field}' has {num_missing} missing values")
    
    # Validate email format
    if 'email_id' in df.columns:
        valid_email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        invalid_emails = df.loc[df['email_id'].notna(), 'email_id'].apply(
            lambda x: not bool(re.match(valid_email_pattern, str(x)))
        )
        if invalid_emails.any():
            num_invalid = invalid_emails.sum()
            errors.append(f"Found {num_invalid} invalid email addresses")
            
    # Check for duplicate user IDs
    if 'user_id' in df.columns:
        duplicate_ids = df['user_id'].duplicated()
        if duplicate_ids.any():
            num_duplicates = duplicate_ids.sum()
            errors.append(f"Found {num_duplicates} duplicate user IDs")
    
    # Validate country names
    if 'country' in df.columns:
        empty_countries = (df['country'] == '') | df['country'].isna()
        if empty_countries.any():
            num_empty = empty_countries.sum()
            warnings.append(f"Found {num_empty} users with missing country information")
    
    # Validate that user_name is not just whitespace or very short
    if 'user_name' in df.columns:
        short_names = df.loc[df['user_name'].notna(), 'user_name'].apply(
            lambda x: len(str(x).strip()) < 3
        )
        if short_names.any():
            num_short = short_names.sum()
            warnings.append(f"Found {num_short} users with very short names (less than 3 characters)")
    
    # Final validation result
    is_valid = len(errors) == 0
    
    return {
        'valid': is_valid,
        'errors': errors,
        'warnings': warnings
    }

def generate_validation_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive validation report for the user data.
    Includes data quality metrics and custom validation checks.
    """
    # Get basic validation results
    validation_result = validate_data(df)
    
    # Create the report structure
    report = {
        'validation_result': validation_result,
        'data_metrics': {
            'total_records': len(df),
            'completeness': {},
            'uniqueness': {},
        },
        'field_statistics': {}
    }
    
    # Calculate completeness for each column
    for col in df.columns:
        filled_values = (~df[col].isna() & (df[col] != '')).sum()
        report['data_metrics']['completeness'][col] = round(filled_values / len(df) * 100, 2)
    
    # Calculate uniqueness for key fields
    key_fields = ['user_id', 'login', 'email_id']
    for field in key_fields:
        if field in df.columns:
            unique_values = df[field].nunique()
            report['data_metrics']['uniqueness'][field] = round(unique_values / df[field].count() * 100, 2)
    
    # Calculate field-specific statistics
    if 'email_id' in df.columns:
        # Email domain distribution
        valid_emails = df.loc[df['email_id'].notna(), 'email_id']
        if not valid_emails.empty:
            try:
                domains = valid_emails.str.extract(r'@([^@]+)$')[0]
                top_domains = domains.value_counts().head(5).to_dict()
                report['field_statistics']['email_domains'] = top_domains
            except Exception as e:
                logger.warning(f"Could not extract email domains: {str(e)}")
    
    if 'designation' in df.columns:
        # Designation statistics
        designations = df.loc[df['designation'].notna(), 'designation']
        if not designations.empty:
            top_designations = designations.value_counts().head(5).to_dict()
            report['field_statistics']['designations'] = top_designations
    
    if 'country' in df.columns:
        # Country statistics
        countries = df.loc[df['country'].notna(), 'country']
        if not countries.empty:
            top_countries = countries.value_counts().head(5).to_dict()
            report['field_statistics']['countries'] = top_countries
    
    return report

def validate_email(email: str) -> bool:
    """Validate an email address format."""
    if pd.isna(email) or email == "":
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_text_field(text: str, min_length: int = 1, max_length: int = 100) -> bool:
    """Validate a text field's length and content."""
    if pd.isna(text) or text == "":
        return False
    
    text = str(text).strip()
    if len(text) < min_length or len(text) > max_length:
        return False
    
    return True