# su_pipeline.py

import os
import sys
import json
import pandas as pd
from datetime import datetime
import argparse

from datapipeline.supplier.su_fetcher import fetch_data
from datapipeline.supplier.su_preprocess import create_preprocessing_pipeline, save_processed_data
from datapipeline.supplier.su_quality import analyze_data, data_quality_checks, summarize_data, generate_supplier_quality_report
from datapipeline.supplier.su_validation import run_all_validations, fix_common_issues, validate_file

from logger.logger import get_logger
from utils.utils import load_config
from utils.data_drift_detector import check_and_update_reference

logger = get_logger(__file__)

def run_pipeline(input_file=None):
    """
    Execute the full supplier data processing pipeline.
    
    Args:
        input_file: Optional path to input file. If None, data will be fetched from database
    """
    try:
        # Step 0: Load config and prepare directories
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "..", "config.yaml")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")

        data_dir = config.get("data_dir", "data/")
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Data will be saved to: {data_dir}")

        reports_dir = os.path.join(data_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        reference_data_path = os.path.join(data_dir, "reference_supplier_data.csv")

        # Step 1: Fetch or load supplier data
        if input_file:
            logger.info(f"Step 1: Loading supplier data from file: {input_file}")
            is_valid, issues = validate_file(input_file)
            if not is_valid:
                logger.error(f"Invalid input file: {'; '.join(issues)}")
                return None, {'error': f"Invalid input file: {'; '.join(issues)}"}
            
            if input_file.endswith('.csv'):
                raw_data = pd.read_csv(input_file)
            elif input_file.endswith(('.xlsx', '.xls')):
                raw_data = pd.read_excel(input_file)
            else:
                logger.error(f"Unsupported file format: {input_file}")
                return None, {'error': f"Unsupported file format: {input_file}"}
        else:
            logger.info("Step 1: Fetching supplier data from database...")
            raw_data = fetch_data()

        # Step 2: Initial validation
        logger.info("Step 2: Running initial data validations...")
        validation_results = run_all_validations(raw_data)

        if not validation_results['overall_valid']:
            logger.warning("Data validation found issues. Attempting to fix common problems...")
            raw_data, fixes_applied = fix_common_issues(raw_data)
            for fix in fixes_applied:
                logger.info(f"Applied fix: {fix}")

            validation_results = run_all_validations(raw_data)

            if not validation_results['overall_valid']:
                logger.warning("Some issues couldn't be fixed automatically. Proceeding with caution.")
                for category, details in validation_results['validation_details'].items():
                    if not details['valid']:
                        for issue in details['issues']:
                            logger.warning(f"Validation issue ({category}): {issue}")
        else:
            logger.info("Initial data validation passed.")

        # Step 3: Analyze data
        logger.info("Step 3: Analyzing initial data quality...")
        initial_stats = analyze_data(raw_data)
        logger.info(f"Initial data analysis complete. Found {sum(initial_stats['missing_values'].values())} missing values.")

        # Step 4: Preprocess
        logger.info("Step 4: Setting up and applying preprocessing pipeline...")
        preprocessing_pipeline = create_preprocessing_pipeline()
        processed_data = preprocessing_pipeline.fit_transform(raw_data)

        # Step 5: Quality check
        logger.info("Step 5: Performing data quality checks...")
        quality_issues = data_quality_checks(processed_data)
        if quality_issues:
            logger.warning("Data quality issues found:")
            for issue in quality_issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No data quality issues found.")

        # Step 6: Final validation
        logger.info("Step 6: Running final validations on processed data...")
        final_validation = run_all_validations(processed_data)
        if not final_validation['overall_valid']:
            logger.warning("Some issues remain in the processed data:")
            for category, details in final_validation['validation_details'].items():
                if not details['valid']:
                    for issue in details['issues']:
                        logger.warning(f"Final validation issue ({category}): {issue}")
        else:
            logger.info("Final validation passed.")

        # Step 7: Summary + quality report
        logger.info("Step 7: Generating data summary...")
        data_summary = summarize_data(processed_data)
        quality_report = generate_supplier_quality_report(processed_data)

        logger.info(f"Data summary: Total suppliers: {data_summary['row_count']}")
        logger.info(f"Top countries: {sorted(data_summary.get('country_distribution', {}).items(), key=lambda x: x[1], reverse=True)[:5]}")

        # Step 8: Save outputs
        logger.info("Step 8: Saving processed data and reports...")
        save_processed_data(processed_data, data_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        validation_path = os.path.join(reports_dir, f"validation_report_{timestamp}.json")
        quality_path = os.path.join(reports_dir, f"quality_report_{timestamp}.json")

        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        with open(quality_path, 'w') as f:
            json.dump(quality_report, f, indent=2)

        # Step 9: Drift detection
        logger.info("Step 9: Checking for new data using reference CSV...")
        has_changed = check_and_update_reference(
            df=processed_data,
            reference_data_path=reference_data_path,
            id_column='supplier_id'  # Update to your actual unique ID column
        )

        if has_changed:
            logger.info("New supplier data detected. Exiting with code 1 to trigger downstream jobs.")
            sys.exit(1)
        else:
            logger.info("No new supplier data detected. Exiting with code 0.")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the supplier data processing pipeline')
    parser.add_argument('--input', '-i', help='Input file path (CSV or Excel). If not provided, data will be fetched from database.')
    args = parser.parse_args()

    run_pipeline(input_file=args.input)
