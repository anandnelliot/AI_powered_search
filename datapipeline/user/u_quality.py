# user_quality.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

from logger.logger import get_logger

logger = get_logger(__file__)


def analyze_data(df: pd.DataFrame) -> Dict:
    """Generate statistical summary and data quality report for user data."""
    stats = {
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},  # Convert dtypes to strings for JSON compatibility
        'unique_counts': df.nunique().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    # Add basic stats for text columns
    text_stats = {}
    for col in df.select_dtypes(include=['object']):
        if df[col].notna().any():
            text_stats[col] = {
                'avg_length': df[col].str.len().mean(),
                'max_length': df[col].str.len().max(),
                'empty_strings': (df[col] == '').sum()
            }
    stats['text_stats'] = text_stats
    
    # Add location statistics
    if 'country' in df.columns:
        stats['location_stats'] = {
            'country_counts': df['country'].value_counts().to_dict(),
            'top_countries': df['country'].value_counts().head(5).to_dict()
        }
    
    # Add company statistics
    if 'supplier' in df.columns:
        stats['supplier_stats'] = {
            'supplier_counts': df['supplier'].value_counts().to_dict(),
            'top_suppliers': df['supplier'].value_counts().head(5).to_dict(),
            'users_without_supplier': (df['supplier'].isna() | (df['supplier'] == '')).sum()
        }
    
    return stats

def data_quality_checks(df: pd.DataFrame) -> List[str]:
    """Perform data quality checks on user data and return a list of issues."""
    issues = []
    
    # Critical columns that should not be missing
    critical_columns = ['user_id', 'user_name', 'email_id']
    
    # Check for missing critical values
    for col in critical_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            issues.append(f"Missing values in critical column: {col}")
    
    # Check for empty user names
    if 'user_name' in df.columns:
        empty_user_names = (df['user_name'] == '').sum()
        if empty_user_names > 0:
            issues.append(f"Found {empty_user_names} empty user names")
    
    # Check for duplicate user IDs
    if 'user_id' in df.columns:
        dup_count = df.duplicated(subset=['user_id']).sum()
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate user IDs")
    
    # Check for invalid email formats
    if 'email_id' in df.columns:
        invalid_emails = 0
        for email in df['email_id']:
            if pd.notna(email) and email != '':
                if '@' not in email or '.' not in email.split('@')[-1]:
                    invalid_emails += 1
        if invalid_emails > 0:
            issues.append(f"Found {invalid_emails} potentially invalid email addresses")
    
    return issues

def generate_user_quality_report(df: pd.DataFrame) -> Dict:
    """Generate a comprehensive quality report for user data."""
    report = {
        'total_users': len(df),
        'users_by_country': df['country'].value_counts().to_dict() if 'country' in df.columns else {},
        'users_by_supplier': df['supplier'].value_counts().to_dict() if 'supplier' in df.columns else {},
        'data_completeness': {}
    }
    
    # Calculate completeness for each column
    for col in df.columns:
        non_empty = (~df[col].isna() & (df[col] != '')).sum()
        report['data_completeness'][col] = round(non_empty / len(df) * 100, 2)
    
    # Calculate users per supplier by country
    if all(col in df.columns for col in ['country', 'supplier']):
        country_supplier_counts = df.groupby(['country', 'supplier']).size().reset_index(name='count')
        report['users_by_country_supplier'] = country_supplier_counts.to_dict('records')
    
    return report

def summarize_data(df: pd.DataFrame) -> Dict:
    """Generate a summary of the user dataset."""
    summary = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    # Country distribution
    if 'country' in df.columns:
        summary['country_distribution'] = df['country'].value_counts().to_dict()
    
    # Supplier distribution
    if 'supplier' in df.columns:
        summary['supplier_distribution'] = df['supplier'].value_counts().to_dict()
    
    # User statistics by designation
    if 'designation' in df.columns:
        summary['designation_distribution'] = df['designation'].value_counts().to_dict()
    
    return summary