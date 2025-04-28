# user_pipeline.py

import os
import sys
import json
import pandas as pd
from datetime import datetime

from datapipeline.user.u_fetcher import fetch_data
from datapipeline.user.u_preprocess import create_preprocessing_pipeline, save_processed_data, preprocess_user_data
from datapipeline.user.u_quality import analyze_data, data_quality_checks, summarize_data, generate_user_quality_report
from datapipeline.user.u_validation import validate_data, generate_validation_report

from logger.logger import get_logger
from utils.utils import load_config
from utils.data_drift_detector import check_and_update_reference

logger = get_logger(__file__)


def run_pipeline():
    """Execute the full user data processing pipeline."""
    try:
        # Step 0: Load config and prepare directories
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "..", "config.yaml")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        
        data_dir = config.get("data_dir", "data/")
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Data will be saved to: {data_dir}")
        
        reports_dir = config.get("reports_dir", os.path.join(data_dir, "reports"))
        os.makedirs(reports_dir, exist_ok=True)
        logger.info(f"Reports will be saved to: {reports_dir}")
        
        # Step 1: Fetch user data
        logger.info("Step 1: Fetching user data from database...")
        raw_data = fetch_data()
        
        # Step 2: Analyze initial data quality
        logger.info("Step 2: Analyzing initial data quality...")
        initial_stats = analyze_data(raw_data)
        logger.info(f"Initial data analysis complete. Found {sum(initial_stats['missing_values'].values())} missing values.")
        
        # Step 3: Create and apply preprocessing pipeline
        logger.info("Step 3: Setting up and applying preprocessing pipeline...")
        preprocessing_pipeline = create_preprocessing_pipeline()
        processed_data = preprocessing_pipeline.fit_transform(raw_data)
        # Alternatively, you can use the standalone function:
        # processed_data = preprocess_user_data(raw_data)
        
        # Step 4: Perform data quality checks
        logger.info("Step 4: Performing data quality checks on processed data...")
        quality_issues = data_quality_checks(processed_data)
        if quality_issues:
            logger.warning("Data quality issues found:")
            for issue in quality_issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No data quality issues found.")
        
        # Step 5: Validate the data
        logger.info("Step 5: Validating user data...")
        validation_result = validate_data(processed_data)
        if validation_result['valid']:
            logger.info("Data validation successful.")
        else:
            logger.warning("Data validation failed:")
            for error in validation_result['errors']:
                logger.warning(f"  - {error}")
        
        # Step 6: Generate summary and quality report
        logger.info("Step 6: Generating data summary...")
        data_summary = summarize_data(processed_data)
        quality_report = generate_user_quality_report(processed_data)
        validation_report = generate_validation_report(processed_data)
        
        logger.info(f"Data summary: Total users: {data_summary['row_count']}")
        logger.info(f"Top countries: {sorted(data_summary.get('country_distribution', {}).items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        # Step 7: Save processed data and reports
        logger.info("Step 7: Saving processed data...")
        save_processed_data(processed_data, data_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        quality_report_path = os.path.join(reports_dir, f"user_quality_report_{timestamp}.json")
        validation_report_path = os.path.join(reports_dir, f"user_validation_report_{timestamp}.json")
        
        with open(quality_report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        with open(validation_report_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        # Step 8: Data drift / change detection
        reference_data_path = os.path.join(data_dir, "reference_user_data.csv")
        logger.info("Step 8: Checking for new data using reference CSV...")
        has_changed = check_and_update_reference(
            df=processed_data,
            reference_data_path=reference_data_path,
            id_column='user_id'  # Adjust to your unique ID column if needed
        )
        
        if has_changed:
            logger.info("New user data detected. Exiting with code 1 to trigger downstream jobs.")
            sys.exit(1)
        else:
            logger.info("No new user data detected. Exiting with code 0.")
            sys.exit(0)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    run_pipeline()
