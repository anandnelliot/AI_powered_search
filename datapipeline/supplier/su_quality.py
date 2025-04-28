# su_quality.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from logger.logger import get_logger

logger = get_logger(__file__)

def analyze_data(df: pd.DataFrame) -> Dict:
    """Generate statistical summary and data quality report for supplier data."""
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
    
    # Add supplier-specific statistics
    if all(col in df.columns for col in ['product_count', 'service_count']):
        stats['supplier_stats'] = {
            'avg_products_per_supplier': df['product_count'].mean(),
            'max_products_per_supplier': df['product_count'].max(),
            'suppliers_with_no_products': (df['product_count'] == 0).sum(),
            'avg_services_per_supplier': df['service_count'].mean(),
            'max_services_per_supplier': df['service_count'].max(),
            'suppliers_with_no_services': (df['service_count'] == 0).sum(),
            'suppliers_with_both': ((df['product_count'] > 0) & (df['service_count'] > 0)).sum()
        }
    
    # Add location statistics
    if 'country' in df.columns:
        stats['location_stats'] = {
            'country_counts': df['country'].value_counts().to_dict(),
            'top_countries': df['country'].value_counts().head(5).to_dict()
        }
    
    return stats

def data_quality_checks(df: pd.DataFrame) -> List[str]:
    """Perform data quality checks on supplier data and return a list of issues."""
    issues = []
    
    # Critical columns that should not be missing
    critical_columns = ['supplier_id', 'supplier_name', 'country', 'state']
    
    # Check for missing critical values
    for col in critical_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            issues.append(f"Missing values in critical column: {col}")
    
    # Check for empty company names
    if 'supplier_name' in df.columns:
        empty_supplier_names = (df['supplier_name'] == '').sum()
        if empty_supplier_names > 0:
            issues.append(f"Found {empty_supplier_names} empty company names")
    
    # Check for duplicate company IDs
    if 'supplier_id' in df.columns:
        dup_count = df.duplicated(subset=['supplier_id']).sum()
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate company IDs")
    
    # Check for data consistency
    if all(col in df.columns for col in ['product_count', 'products']):
        # Verify that product_count matches the actual count of products
        mismatch_count = 0
        for _, row in df.iterrows():
            if row['products']:
                actual_count = len(str(row['products']).split(', ')) if row['products'] else 0
                if actual_count != row['product_count']:
                    mismatch_count += 1
        
        if mismatch_count > 0:
            issues.append(f"Found {mismatch_count} rows with product count inconsistencies")
    
    # Check for suppliers with excessive product or service counts (potential data issues)
    if 'product_count' in df.columns:
        high_product_count = df[df['product_count'] > 100].shape[0]
        if high_product_count > 0:
            issues.append(f"Found {high_product_count} suppliers with more than 100 products (verify data quality)")
    
    return issues

def generate_supplier_quality_report(df: pd.DataFrame) -> Dict:
    """Generate a comprehensive quality report for supplier data."""
    report = {
        'total_suppliers': len(df),
        'suppliers_with_products': (df['product_count'] > 0).sum() if 'product_count' in df.columns else 'N/A',
        'suppliers_with_services': (df['service_count'] > 0).sum() if 'service_count' in df.columns else 'N/A',
        'suppliers_by_country': df['country'].value_counts().to_dict() if 'country' in df.columns else {},
        'data_completeness': {}
    }
    
    # Calculate completeness for each column
    for col in df.columns:
        non_empty = (~df[col].isna() & (df[col] != '')).sum()
        report['data_completeness'][col] = round(non_empty / len(df) * 100, 2)
    
    # Calculate average products and services per supplier by country
    if all(col in df.columns for col in ['country', 'product_count']):
        report['avg_products_by_country'] = df.groupby('country')['product_count'].mean().to_dict()
    if all(col in df.columns for col in ['country', 'service_count']):
        report['avg_services_by_country'] = df.groupby('country')['service_count'].mean().to_dict()
    
    return report

def summarize_data(df: pd.DataFrame) -> Dict:
    """Generate a summary of the supplier dataset."""
    summary = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    # Country distribution
    if 'country' in df.columns:
        summary['country_distribution'] = df['country'].value_counts().to_dict()
    
    # Product and service statistics
    if 'product_count' in df.columns:
        summary['product_stats'] = {
            'total_products': df['product_count'].sum(),
            'avg_products_per_supplier': round(df['product_count'].mean(), 2),
            'suppliers_with_products': (df['product_count'] > 0).sum(),
            'suppliers_without_products': (df['product_count'] == 0).sum()
        }
    
    if 'service_count' in df.columns:
        summary['service_stats'] = {
            'total_services': df['service_count'].sum(),
            'avg_services_per_supplier': round(df['service_count'].mean(), 2),
            'suppliers_with_services': (df['service_count'] > 0).sum(),
            'suppliers_without_services': (df['service_count'] == 0).sum()
        }
    
    return summary