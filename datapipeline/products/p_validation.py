# p_validation.py
"""
Validation script for product data pipeline.
Validates raw data before processing and provides validation reports.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re

from logger.logger import get_logger

logger = get_logger(__file__)


class DataValidator:
    """Class for validating product data before processing."""
    
    def __init__(self):
        # Define validation rules
        self.required_columns = [
            'product_id', 'product_name', 'description',
            'bicc_categoryname', 'bicsc_subcategoryname',
            'MCM_CompanyName', 'CyM_CountryName'
        ]
        
        # Regex patterns for different validations
        self.patterns = {
            'alphanumeric': re.compile(r'^[a-zA-Z0-9\s\-_]+$'),
            'email': re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'),
            'html': re.compile(r'<[^>]+>')
        }
        
        # Column-specific validation rules
        self.column_rules = {
            'product_id': {'type': 'numeric', 'required': True, 'unique': True},
            'product_name': {'type': 'text', 'required': True, 'min_length': 2, 'max_length': 200},
            'description': {'type': 'text', 'required': False, 'max_html': True},
            'bicc_categoryname': {'type': 'categorical', 'required': True},
            'bicsc_subcategoryname': {'type': 'categorical', 'required': False},
            'MCM_CompanyName': {'type': 'text', 'required': True}
        }
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate the dataframe against all rules and return status and results.
        
        Args:
            df: The dataframe to validate
            
        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - Dictionary with detailed validation results
        """
        validation_results = {
            'passed': True,
            'missing_columns': [],
            'empty_columns': {},
            'data_type_issues': {},
            'duplicate_issues': {},
            'constraint_violations': {},
            'records_analyzed': len(df)
        }
        
        # Check for required columns
        for col in self.required_columns:
            if col not in df.columns:
                validation_results['missing_columns'].append(col)
                validation_results['passed'] = False
        
        if validation_results['missing_columns']:
            logger.warning(f"Validation failed: Missing required columns: {validation_results['missing_columns']}")
            return False, validation_results
        
        # Check for empty columns
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                validation_results['empty_columns'][col] = null_count
                if col in self.column_rules and self.column_rules[col].get('required', False):
                    validation_results['passed'] = False
        
        # Check data types and constraints
        for col, rules in self.column_rules.items():
            if col not in df.columns:
                continue
                
            # Check data type
            if rules.get('type') == 'numeric':
                non_numeric = df[col].apply(lambda x: not pd.isna(x) and not str(x).isdigit()).sum()
                if non_numeric > 0:
                    validation_results['data_type_issues'][col] = f"{non_numeric} non-numeric values"
                    validation_results['passed'] = False
            
            # Check uniqueness
            if rules.get('unique', False):
                duplicate_count = df[col].duplicated().sum()
                if duplicate_count > 0:
                    validation_results['duplicate_issues'][col] = duplicate_count
                    validation_results['passed'] = False
            
            # Check text constraints
            if rules.get('type') == 'text':
                # Check min length
                if 'min_length' in rules:
                    too_short = df[col].apply(
                        lambda x: len(str(x)) < rules['min_length'] if not pd.isna(x) else False
                    ).sum()
                    if too_short > 0:
                        if col not in validation_results['constraint_violations']:
                            validation_results['constraint_violations'][col] = []
                        validation_results['constraint_violations'][col].append(
                            f"{too_short} values shorter than {rules['min_length']} chars"
                        )
                        validation_results['passed'] = False
                
                # Check max length
                if 'max_length' in rules:
                    too_long = df[col].apply(
                        lambda x: len(str(x)) > rules['max_length'] if not pd.isna(x) else False
                    ).sum()
                    if too_long > 0:
                        if col not in validation_results['constraint_violations']:
                            validation_results['constraint_violations'][col] = []
                        validation_results['constraint_violations'][col].append(
                            f"{too_long} values longer than {rules['max_length']} chars"
                        )
                        validation_results['passed'] = False
                
                # Check for HTML content
                if rules.get('max_html', False):
                    has_html = df[col].apply(
                        lambda x: bool(self.patterns['html'].search(str(x))) if not pd.isna(x) else False
                    ).sum()
                    if has_html > 0:
                        if col not in validation_results['constraint_violations']:
                            validation_results['constraint_violations'][col] = []
                        validation_results['constraint_violations'][col].append(
                            f"{has_html} values contain HTML tags"
                        )
                        # This is just a warning, doesn't fail validation
        
        return validation_results['passed'], validation_results
    
    def generate_validation_report(self, validation_results: Dict) -> str:
        """Generate a human-readable validation report from validation results."""
        report = ["DATA VALIDATION REPORT", "=" * 30]
        
        report.append(f"Status: {'PASSED' if validation_results['passed'] else 'FAILED'}")
        report.append(f"Records analyzed: {validation_results['records_analyzed']}")
        
        if validation_results['missing_columns']:
            report.append("\nMissing Columns:")
            for col in validation_results['missing_columns']:
                report.append(f"  - {col}")
        
        if validation_results['empty_columns']:
            report.append("\nEmpty Values:")
            for col, count in validation_results['empty_columns'].items():
                report.append(f"  - {col}: {count} missing values")
        
        if validation_results['data_type_issues']:
            report.append("\nData Type Issues:")
            for col, issue in validation_results['data_type_issues'].items():
                report.append(f"  - {col}: {issue}")
        
        if validation_results['duplicate_issues']:
            report.append("\nDuplicate Issues:")
            for col, count in validation_results['duplicate_issues'].items():
                report.append(f"  - {col}: {count} duplicate values")
        
        if validation_results['constraint_violations']:
            report.append("\nConstraint Violations:")
            for col, issues in validation_results['constraint_violations'].items():
                report.append(f"  - {col}:")
                for issue in issues:
                    report.append(f"      * {issue}")
        
        return "\n".join(report)

def validate_raw_data(data: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Validate the raw data before it enters the preprocessing pipeline.
    
    Args:
        data: DataFrame containing raw data to validate
        
    Returns:
        Tuple containing:
            - Boolean indicating if validation passed
            - Dictionary with detailed validation results
    """
    validator = DataValidator()
    is_valid, results = validator.validate_data(data)
    
    # Log validation results
    report = validator.generate_validation_report(results)
    if is_valid:
        logger.info("Data validation passed!")
        logger.debug(report)
    else:
        logger.warning("Data validation failed!")
        logger.warning(report)
    
    return is_valid, results

def get_validation_summary(results: Dict) -> Dict:
    """Return a simplified summary of validation results for reporting."""
    return {
        'passed': results['passed'],
        'records_analyzed': results['records_analyzed'],
        'missing_column_count': len(results['missing_columns']),
        'total_empty_values': sum(results['empty_columns'].values()) if results['empty_columns'] else 0,
        'data_type_issues_count': len(results['data_type_issues']),
        'duplicate_issues_count': len(results['duplicate_issues']),
        'constraint_violations_count': sum(len(issues) for issues in results['constraint_violations'].values()) if results['constraint_violations'] else 0
    }