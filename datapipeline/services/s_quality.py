# data_quality.py
import pandas as pd
from typing import Dict, List

from logger.logger import get_logger

logger = get_logger(__file__)

def analyze_data(df: pd.DataFrame) -> Dict:
    """Generate statistical summary and data quality report."""
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
    
    return stats

def data_quality_checks(df: pd.DataFrame) -> List[str]:
    """Perform data quality checks and return a list of issues."""
    issues = []
    
    # Define critical columns for services
    critical_columns = ['service_id', 'service_name', 'MCSvD_DisplayName', 'category', 
                        'bicc_categoryname', 'supplier', 'MCM_CompanyName']
    
    # Check for missing critical values
    for col in critical_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            issues.append(f"Missing values in critical column: {col}")
    
    # Check for data consistency
    service_name_cols = ['service_name', 'MCSvD_DisplayName', 'display_name']
    for col in service_name_cols:
        if col in df.columns and df[col].isna().any():
            issues.append(f"Some service rows are missing {col}")
    
    # Check duplicates on service_id
    id_col = 'service_id'
    if id_col in df.columns:
        dup_count = df.duplicated(subset=[id_col]).sum()
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate service IDs")
    
    # Check for empty text in important fields
    text_cols = ['service_name', 'display_name', 'description', 'service_description', 
                'supplier', 'category', 'subcategory', 'service_type']
    for col in text_cols:
        if col in df.columns:
            empty_count = (df[col] == '').sum()
            if empty_count > 0:
                issues.append(f"Found {empty_count} empty strings in {col}")
    
    return issues

def summarize_data(df: pd.DataFrame) -> Dict:
    """Generate a summary of the dataset."""
    summary = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    # Category counts
    if 'category' in df.columns:
        summary['category_counts'] = df['category'].value_counts().to_dict()
    elif 'bicc_categoryname' in df.columns:
        summary['category_counts'] = df['bicc_categoryname'].value_counts().to_dict()
    
    # Service type counts
    if 'service_type' in df.columns:
        summary['service_type_counts'] = df['service_type'].value_counts().to_dict()
    elif 'bicsm_servicename' in df.columns:
        summary['service_type_counts'] = df['bicsm_servicename'].value_counts().to_dict()
    
    # Company counts
    if 'supplier' in df.columns:
        summary['company_count'] = df['supplier'].nunique()
    elif 'MCM_CompanyName' in df.columns:
        summary['company_count'] = df['MCM_CompanyName'].nunique()
    
    return summary