# su_validation.py
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

from logger.logger import get_logger

logger = get_logger(__file__)

def validate_file(file_path: str) -> Tuple[bool, List[str]]:
    """
    Validate if a file exists, has the correct format, and contains required columns.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check if file exists
    if not os.path.exists(file_path):
        issues.append(f"File not found: {file_path}")
        return False, issues
    
    # Check if file is not empty
    if os.path.getsize(file_path) == 0:
        issues.append(f"File is empty: {file_path}")
        return False, issues
    
    # Check file extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in ['.csv', '.xlsx', '.xls']:
        issues.append(f"Unsupported file format: {ext}. Expected .csv or .xlsx")
        return False, issues
    
    try:
        # Try to read the file as CSV or Excel
        if ext.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Check required columns
        required_columns = ['supplier_id', 'supplier_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check if file has any data rows
        if len(df) == 0:
            issues.append(f"File has no data rows: {file_path}")
            
        return len(issues) == 0, issues
        
    except Exception as e:
        issues.append(f"Error reading file: {str(e)}")
        return False, issues

def check_data_consistency(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Check data consistency in the supplier dataframe.
    
    Args:
        df: Supplier DataFrame to validate
        
    Returns:
        Tuple of (is_consistent, list_of_inconsistencies)
    """
    inconsistencies = []
    
    # Check for duplicate supplier_id
    if 'supplier_id' in df.columns:
        duplicates = df[df.duplicated(['supplier_id'], keep=False)]
        if not duplicates.empty:
            inconsistencies.append(f"Found {len(duplicates)} duplicate supplier IDs")
            logger.warning(f"Duplicate supplier IDs: {duplicates['supplier_id'].unique().tolist()[:5]}...")
    
    # Validate products_and_services consistency
    if 'products_and_services' in df.columns:
        empty_pas = df[df['products_and_services'].isna() | (df['products_and_services'] == '')].shape[0]
        if empty_pas > 0:
            inconsistencies.append(f"{empty_pas} suppliers have no products or services listed")
    
    # Validate location hierarchies
    if all(col in df.columns for col in ['country', 'state']):
        # Check for states without countries
        missing_country = df[(df['state'].notna()) & (df['state'] != '') & 
                            (df['country'].isna() | (df['country'] == ''))].shape[0]
        if missing_country > 0:
            inconsistencies.append(f"{missing_country} suppliers have state but no country")
    
    # Check for empty but required fields
    for field in ['supplier_name', 'country']:
        if field in df.columns:
            empty_fields = df[df[field].isna() | (df[field] == '')].shape[0]
            if empty_fields > 0:
                inconsistencies.append(f"{empty_fields} suppliers have empty {field}")
    
    # Validate classification values if present
    if 'classification' in df.columns:
        valid_classifications = ['manufacturer', 'distributor', 'service provider', 'consultant']
        invalid_class = df[~df['classification'].str.lower().isin(valid_classifications) & 
                          df['classification'].notna() & (df['classification'] != '')].shape[0]
        if invalid_class > 0:
            inconsistencies.append(f"{invalid_class} suppliers have invalid classification")
    
    return len(inconsistencies) == 0, inconsistencies

def validate_supplier_ids(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate supplier IDs format and consistency.
    
    Args:
        df: Supplier DataFrame to validate
        
    Returns:
        Tuple of (are_ids_valid, list_of_issues)
    """
    issues = []
    
    if 'supplier_id' not in df.columns:
        issues.append("supplier_id column not found in DataFrame")
        return False, issues
    
    # Check for null IDs
    null_ids = df['supplier_id'].isna().sum()
    if null_ids > 0:
        issues.append(f"Found {null_ids} null supplier IDs")
    
    # Check for non-numeric IDs (if IDs should be numeric)
    if df['supplier_id'].dtype not in [np.int64, np.float64]:
        try:
            # Try to convert to numeric and check if there are any errors
            non_numeric = pd.to_numeric(df['supplier_id'], errors='coerce').isna().sum()
            if non_numeric > 0:
                issues.append(f"Found {non_numeric} non-numeric supplier IDs")
        except:
            issues.append("Unable to validate supplier ID format")
    
    # Check for zero or negative IDs
    invalid_ids = ((df['supplier_id'] <= 0) & df['supplier_id'].notna()).sum()
    if invalid_ids > 0:
        issues.append(f"Found {invalid_ids} zero or negative supplier IDs")
    
    return len(issues) == 0, issues

def validate_business_rules(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate business rules for supplier data.
    
    Args:
        df: Supplier DataFrame to validate
        
    Returns:
        Tuple of (rules_followed, list_of_violations)
    """
    violations = []
    
    # Rule 1: Every supplier should have at least one product or service
    if 'products_and_services' in df.columns:
        no_products_services = df[df['products_and_services'].isna() | 
                                 (df['products_and_services'] == '')].shape[0]
        if no_products_services > 0:
            violations.append(f"Business rule violation: {no_products_services} suppliers have no products or services")
    
    # Rule 2: Every supplier must have a country
    if 'country' in df.columns:
        no_country = df[df['country'].isna() | (df['country'] == '')].shape[0]
        if no_country > 0:
            violations.append(f"Business rule violation: {no_country} suppliers have no country")
    
    # Rule 3: Every manufacturer should have at least one product
    if all(col in df.columns for col in ['classification', 'products_and_services']):
        manufacturers = df[df['classification'].str.lower() == 'manufacturer']
        if not manufacturers.empty:
            manufacturers_no_products = manufacturers[manufacturers['products_and_services'].isna() | 
                                                    (manufacturers['products_and_services'] == '')].shape[0]
            if manufacturers_no_products > 0:
                violations.append(f"Business rule violation: {manufacturers_no_products} manufacturers have no products")
    
    return len(violations) == 0, violations

def run_all_validations(df: pd.DataFrame) -> Dict[str, Union[bool, List[str]]]:
    """
    Run all validation checks on the supplier DataFrame.
    
    Args:
        df: Supplier DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Running comprehensive data validations...")
    results = {
        'overall_valid': True,
        'validation_details': {}
    }
    
    # Run consistency checks
    consistency_valid, inconsistencies = check_data_consistency(df)
    results['validation_details']['consistency'] = {
        'valid': consistency_valid,
        'issues': inconsistencies
    }
    
    # Run supplier ID validations
    ids_valid, id_issues = validate_supplier_ids(df)
    results['validation_details']['supplier_ids'] = {
        'valid': ids_valid,
        'issues': id_issues
    }
    
    # Run business rule validations
    rules_valid, violations = validate_business_rules(df)
    results['validation_details']['business_rules'] = {
        'valid': rules_valid,
        'issues': violations
    }
    
    # Update overall validity
    results['overall_valid'] = consistency_valid and ids_valid and rules_valid
    
    # Log validation summary
    issue_count = len(inconsistencies) + len(id_issues) + len(violations)
    if results['overall_valid']:
        logger.info("Data validation successful - no issues found")
    else:
        logger.warning(f"Data validation found {issue_count} issues")
        
    return results

def fix_common_issues(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Attempt to fix common data issues automatically.
    
    Args:
        df: Supplier DataFrame to fix
        
    Returns:
        Tuple of (fixed_dataframe, list_of_fixes_applied)
    """
    fixes_applied = []
    fixed_df = df.copy()
    
    # Fix 1: Convert supplier_ids to integers where possible
    if 'supplier_id' in fixed_df.columns:
        try:
            # Store original values
            original_ids = fixed_df['supplier_id'].copy()
            # Try to convert to integers
            fixed_df['supplier_id'] = pd.to_numeric(fixed_df['supplier_id'], errors='coerce')
            # Count fixes
            fix_count = (fixed_df['supplier_id'].notna() & (fixed_df['supplier_id'] != original_ids)).sum()
            if fix_count > 0:
                fixes_applied.append(f"Converted {fix_count} supplier IDs to numeric format")
        except Exception as e:
            logger.warning(f"Could not fix supplier IDs: {str(e)}")
    
    # Fix 2: Standardize classifications
    if 'classification' in fixed_df.columns:
        classification_map = {
            'mfr': 'manufacturer',
            'manufacturer': 'manufacturer',
            'dist': 'distributor',
            'distributor': 'distributor',
            'service': 'service provider',
            'services': 'service provider',
            'service provider': 'service provider',
            'consultant': 'consultant',
            'consulting': 'consultant'
        }
        
        original_values = fixed_df['classification'].copy()
        fixed_df['classification'] = fixed_df['classification'].str.lower().map(classification_map)
        
        # Restore null values
        fixed_df.loc[original_values.isna(), 'classification'] = None
        
        # Count fixes
        fix_count = (fixed_df['classification'].notna() & 
                     (fixed_df['classification'] != original_values) & 
                     (original_values.notna())).sum()
        
        if fix_count > 0:
            fixes_applied.append(f"Standardized {fix_count} classification values")
    
    # Fix 3: Fill missing countries with 'Unknown' for suppliers with states or cities
    if all(col in fixed_df.columns for col in ['country', 'state', 'city']):
        missing_country = ((fixed_df['country'].isna() | (fixed_df['country'] == '')) & 
                          ((fixed_df['state'].notna()) | (fixed_df['city'].notna())))
        
        if missing_country.sum() > 0:
            fixed_df.loc[missing_country, 'country'] = 'unknown'
            fixes_applied.append(f"Added 'unknown' country for {missing_country.sum()} suppliers with state/city but no country")
    
    return fixed_df, fixes_applied