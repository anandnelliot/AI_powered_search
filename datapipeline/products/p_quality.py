# data_quality.py
import pandas as pd
from typing import Dict, List, Optional
import json
import os
from datetime import datetime

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
    
    # We need to handle both original column names and renamed column names
    original_critical_columns = ['bicc_categoryname', 'MCPrD_DisplayName', 'MCM_CompanyName', 'product_name']
    renamed_critical_columns = ['category', 'display_name', 'company_name', 'product_name']
    
    # Check for missing critical values using both original and renamed columns
    critical_columns = original_critical_columns + renamed_critical_columns
    for col in critical_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            issues.append(f"Missing values in critical column: {col}")
    
    # Check for data consistency
    if 'product_name' in df.columns:
        if df['product_name'].isna().any():
            issues.append("Some product rows are missing product_name")
    
    # Check duplicates on product_id
    product_id_col = 'product_id'
    if product_id_col in df.columns:
        dup_count = df.duplicated(subset=[product_id_col]).sum()
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate product IDs")
    
    # Check for empty text in important fields
    text_cols = ['product_name', 'display_name', 'description', 'company_name', 
                'category', 'subcategory', 'product_type']
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
    
    # Company counts
    if 'company_name' in df.columns:
        summary['company_count'] = df['company_name'].nunique()
    elif 'MCM_CompanyName' in df.columns:
        summary['company_count'] = df['MCM_CompanyName'].nunique()
    
    return summary

def save_quality_report(stats: Dict, issues: List[str], output_dir: str = './reports'):
    """Save data quality metrics and issues to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        'timestamp': timestamp,
        'statistics': stats,
        'issues': issues,
        'issue_count': len(issues)
    }
    
    # Save to file
    filename = f"{output_dir}/quality_report_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Quality report saved to {filename}")
    return filename

def compare_quality_reports(before_report: Dict, after_report: Dict) -> Dict:
    """
    Compare quality metrics before and after processing to measure improvement.
    
    Args:
        before_report: Data quality report before processing
        after_report: Data quality report after processing
        
    Returns:
        Dictionary with improvement metrics
    """
    comparison = {
        'metrics_compared_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'improvements': {},
        'regressions': {},
        'unchanged': {}
    }
    
    # Compare issue counts
    before_issues = before_report.get('issue_count', 0)
    after_issues = after_report.get('issue_count', 0)
    
    comparison['overall_improvement'] = {
        'issues_before': before_issues,
        'issues_after': after_issues,
        'issue_reduction': before_issues - after_issues,
        'percentage_improvement': ((before_issues - after_issues) / before_issues * 100) if before_issues > 0 else 0
    }
    
    # Compare missing values
    before_missing = sum(before_report.get('statistics', {}).get('missing_values', {}).values())
    after_missing = sum(after_report.get('statistics', {}).get('missing_values', {}).values())
    
    comparison['missing_values'] = {
        'before': before_missing,
        'after': after_missing,
        'reduction': before_missing - after_missing,
        'percentage_improvement': ((before_missing - after_missing) / before_missing * 100) if before_missing > 0 else 0
    }
    
    return comparison