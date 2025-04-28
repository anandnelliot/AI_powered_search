# s_validation.py
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

from logger.logger import get_logger

logger = get_logger(__file__)

class DataValidator:
    """Class for validating service data throughout the processing pipeline."""
    
    def __init__(self, validation_rules: Optional[Dict] = None, 
                 output_dir: str = './validation_reports'):
        """
        Initialize the validator with rules and output directory.
        
        Args:
            validation_rules: Dictionary containing validation rules
            output_dir: Directory to save validation reports
        """
        self.validation_rules = validation_rules or self._get_default_rules()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _get_default_rules(self) -> Dict:
        """Define default validation rules."""
        return {
            'required_columns': [
                'service_id', 'service_name', 'category', 'supplier'
            ],
            'unique_columns': [
                'service_id'
            ],
            'non_empty_columns': [
                'service_name', 'category', 'supplier'
            ],
            'max_null_percentage': 10.0,  # Maximum percentage of nulls allowed in non-required columns
            'min_category_count': 3,      # Each category should have at least this many services
            'min_rows': 10,               # Minimum number of rows expected in the dataset
            'string_length_limits': {
                'service_name': {'min': 3, 'max': 200},
                'service_description': {'min': 0, 'max': 5000},
                'category': {'min': 2, 'max': 100},
                'subcategory': {'min': 0, 'max': 100}
            }
        }
    
    def validate_raw_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate raw data before preprocessing.
        
        Args:
            df: Raw DataFrame to validate
            
        Returns:
            Tuple containing (passed_validation, list_of_issues)
        """
        issues = []
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
            
        # Check for minimum row count
        if len(df) < self.validation_rules['min_rows']:
            issues.append(f"DataFrame has fewer than {self.validation_rules['min_rows']} rows")
        
        # Check for required columns in raw data
        raw_required_columns = ['service_id', 'MCSvD_DisplayName', 'bicc_categoryname', 'MCM_CompanyName']
        missing_columns = [col for col in raw_required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns in raw data: {', '.join(missing_columns)}")
        
        # Check for service_id uniqueness
        if 'service_id' in df.columns:
            duplicate_ids = df['service_id'].duplicated().sum()
            if duplicate_ids > 0:
                issues.append(f"Found {duplicate_ids} duplicate service_id values")
        
        # Check data types
        expected_types = {
            'service_id': ['int64', 'int32'],
            'MCSvD_DisplayName': ['object', 'string']
        }
        
        for col, expected_type_list in expected_types.items():
            if col in df.columns:
                if not any(str(df[col].dtype).startswith(t) for t in expected_type_list):
                    issues.append(f"Column {col} has type {df[col].dtype}, expected one of {expected_type_list}")
        
        # Save validation report
        self._save_validation_report('raw_data', issues, df)
        
        return len(issues) == 0, issues
    
    def validate_processed_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate processed data after preprocessing.
        
        Args:
            df: Processed DataFrame to validate
            
        Returns:
            Tuple containing (passed_validation, list_of_issues)
        """
        issues = []
        
        # Check for required columns in processed data
        missing_columns = [col for col in self.validation_rules['required_columns'] if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns after processing: {', '.join(missing_columns)}")
        
        # Check unique constraints
        for col in self.validation_rules['unique_columns']:
            if col in df.columns:
                duplicate_count = df[col].duplicated().sum()
                if duplicate_count > 0:
                    issues.append(f"Found {duplicate_count} duplicate values in column {col}")
        
        # Check for non-empty fields
        for col in self.validation_rules['non_empty_columns']:
            if col in df.columns:
                empty_count = (df[col].isna() | (df[col] == "") | (df[col] == "0")).sum()
                if empty_count > 0:
                    issues.append(f"Found {empty_count} empty values in required column {col}")
        
        # Check null percentage in each column
        for col in df.columns:
            null_pct = (df[col].isna().sum() / len(df)) * 100
            if null_pct > self.validation_rules['max_null_percentage']:
                issues.append(f"Column {col} has {null_pct:.2f}% null values, exceeding threshold of {self.validation_rules['max_null_percentage']}%")
        
        # Check string length limits
        for col, limits in self.validation_rules['string_length_limits'].items():
            if col in df.columns:
                # Only check non-null string values
                mask = df[col].notna() & (df[col] != "")
                if mask.any():
                    string_lengths = df.loc[mask, col].astype(str).str.len()
                    
                    # Check minimum length
                    if 'min' in limits:
                        too_short = (string_lengths < limits['min']).sum()
                        if too_short > 0:
                            issues.append(f"Found {too_short} values in {col} shorter than minimum length {limits['min']}")
                    
                    # Check maximum length
                    if 'max' in limits:
                        too_long = (string_lengths > limits['max']).sum()
                        if too_long > 0:
                            issues.append(f"Found {too_long} values in {col} longer than maximum length {limits['max']}")
        
        # Check minimum category counts
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            small_categories = category_counts[category_counts < self.validation_rules['min_category_count']]
            if not small_categories.empty:
                issues.append(f"Found {len(small_categories)} categories with fewer than {self.validation_rules['min_category_count']} services")
                for cat, count in small_categories.items():
                    issues.append(f"  - Category '{cat}' has only {count} services")
        
        # Save validation report
        self._save_validation_report('processed_data', issues, df)
        
        return len(issues) == 0, issues
    
    def validate_for_export(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Final validation before data export.
        
        Args:
            df: DataFrame to validate before export
            
        Returns:
            Tuple containing (passed_validation, list_of_issues)
        """
        issues = []
        
        # Check that we have enough data to export
        if len(df) < self.validation_rules['min_rows']:
            issues.append(f"Export data has only {len(df)} rows, minimum required is {self.validation_rules['min_rows']}")
        
        # Check for any invalid data that might cause export issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for problematic characters in string columns that might affect CSV export
                if df[col].str.contains(r'[\n\r,"]', regex=True).any():
                    issues.append(f"Column {col} contains characters that may cause CSV export issues (newlines, commas, quotes)")
        
        # Save validation report
        self._save_validation_report('export_data', issues, df)
        
        return len(issues) == 0, issues
    
    def _save_validation_report(self, stage: str, issues: List[str], df: pd.DataFrame) -> None:
        """
        Save validation report to file.
        
        Args:
            stage: Pipeline stage (raw_data, processed_data, export_data)
            issues: List of validation issues
            df: DataFrame being validated
        """
        report = {
            'timestamp': self.timestamp,
            'stage': stage,
            'passed': len(issues) == 0,
            'issues': issues,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'null_counts': df.isnull().sum().to_dict(),
            'data_summary': {
                'row_count': len(df),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            }
        }
        
        # Add category counts if available
        if 'category' in df.columns or 'bicc_categoryname' in df.columns:
            cat_col = 'category' if 'category' in df.columns else 'bicc_categoryname'
            report['data_summary']['category_counts'] = df[cat_col].value_counts().to_dict()
        
        # Write report to file
        filename = f"{self.output_dir}/{stage}_validation_{self.timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report for {stage} saved to {filename}")
        if issues:
            logger.warning(f"Validation found {len(issues)} issues in {stage}")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info(f"Validation passed for {stage}")


def validate_data_pipeline(raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> bool:
    """
    Validate data at different stages of the pipeline.
    
    Args:
        raw_data: Raw data from database
        processed_data: Data after preprocessing
        
    Returns:
        True if validation passes, False otherwise
    """
    validator = DataValidator()
    
    # Validate raw data
    raw_valid, raw_issues = validator.validate_raw_data(raw_data)
    if not raw_valid:
        logger.error(f"Raw data validation failed with {len(raw_issues)} issues")
        for issue in raw_issues:
            logger.error(f"  - {issue}")
    
    # Validate processed data
    proc_valid, proc_issues = validator.validate_processed_data(processed_data)
    if not proc_valid:
        logger.error(f"Processed data validation failed with {len(proc_issues)} issues")
        for issue in proc_issues:
            logger.error(f"  - {issue}")
    
    # Validate data for export (same as processed data in this case)
    export_valid, export_issues = validator.validate_for_export(processed_data)
    if not export_valid:
        logger.error(f"Export validation failed with {len(export_issues)} issues")
        for issue in export_issues:
            logger.error(f"  - {issue}")
    
    # Return overall validation result
    passed = raw_valid and proc_valid and export_valid
    if passed:
        logger.info("All validation checks passed!")
    else:
        logger.warning("Some validation checks failed. See log for details.")
    
    return passed


if __name__ == "__main__":
    # Configure logging when run as a standalone script
    
    # Test with a small sample DataFrame
    df = pd.DataFrame({
        'service_id': [1, 2, 3],
        'service_name': ['Service A', 'Service B', 'Service C'],
        'category': ['Cat A', 'Cat B', 'Cat A'],
        'supplier': ['Supplier X', 'Supplier Y', 'Supplier Z']
    })
    
    validator = DataValidator()
    valid, issues = validator.validate_processed_data(df)
    print(f"Validation passed: {valid}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")