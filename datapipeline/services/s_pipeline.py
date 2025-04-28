# main_pipeline.py

import os
import sys
import pandas as pd

from datapipeline.services.s_fetcher import fetch_data
from datapipeline.services.s_preprocess import create_preprocessing_pipeline, save_processed_data
from datapipeline.services.s_quality import analyze_data, data_quality_checks, summarize_data
from datapipeline.services.s_validation import validate_data_pipeline, DataValidator
from utils.utils import load_config
from utils.data_drift_detector import check_and_update_reference
from logger.logger import get_logger

logger = get_logger(__file__)

def run_pipeline(validate_only=False):
    try:
        # Step 0: Load config and prepare directories
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "..", "config.yaml")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        
        data_dir = config.get("data_dir", "data/")
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Data will be saved to: {data_dir}")
        
        # Update: Get reports directory from config
        reports_dir = config.get("reports_dir", os.path.join(data_dir, "reports"))
        os.makedirs(reports_dir, exist_ok=True)
        logger.info(f"Reports will be saved to: {reports_dir}")
        
        reference_data_path = os.path.join(data_dir, "reference_services_data.csv")

        # Step 1: Fetch data
        logger.info("Step 1: Fetching service data from database...")
        raw_data = fetch_data()

        # Step 2: Analyze initial data quality
        logger.info("Step 2: Analyzing initial data quality...")
        initial_stats = analyze_data(raw_data)
        logger.info(f"Initial analysis: {sum(initial_stats['missing_values'].values())} missing values.")

        # Step 3: Validate raw data
        validator = DataValidator()
        logger.info("Step 3: Validating raw data...")
        is_valid, issues = validator.validate_raw_data(raw_data)
        if not is_valid:
            logger.warning(f"Validation issues found: {len(issues)}")
            for issue in issues:
                logger.warning(f"  - {issue}")
            if validate_only or '--strict' in sys.argv:
                logger.error("Aborting pipeline due to validation failure.")
                return None
        else:
            logger.info("Raw data validation passed.")

        if validate_only:
            logger.info("Validation only mode complete.")
            return issues

        # Step 4: Preprocessing
        logger.info("Step 4: Preprocessing service data...")
        preprocessing_pipeline = create_preprocessing_pipeline()
        processed_data = preprocessing_pipeline.fit_transform(raw_data)

        # Step 5: Data quality checks
        logger.info("Step 5: Performing data quality checks...")
        quality_issues = data_quality_checks(processed_data)
        if quality_issues:
            logger.warning("Data quality issues found:")
            for issue in quality_issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No data quality issues found.")

        # Step 6: Generate summary
        logger.info("Step 6: Generating summary...")
        summary = summarize_data(processed_data)
        logger.info(f"Summary:\n{summary}")

        # Step 7: Validate processed data
        logger.info("Step 7: Validating processed data...")
        if not validate_data_pipeline(raw_data, processed_data):
            logger.warning("Processed data validation failed.")
        else:
            logger.info("Processed data validation passed.")

        # Step 8: Save processed data
        logger.info("Step 8: Saving processed data...")
        save_processed_data(processed_data, data_dir)

        # Step 9: Data drift / change detection
        logger.info("Step 9: Checking for new data using reference CSV...")
        has_changed = check_and_update_reference(
            df=processed_data,
            reference_data_path=reference_data_path,
            id_column='service_id'  # Replace with the correct ID column if different
        )

        # Exit based on data drift detection
        if has_changed:
            logger.info("New service data detected. Exiting with code 1 to trigger downstream jobs.")
            sys.exit(1)
        else:
            logger.info("No new service data detected. Exiting with code 0.")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Pipeline failed due to error: {str(e)}", exc_info=True)
        sys.exit(2)

if __name__ == "__main__":
    validate_only = '--validate-only' in sys.argv
    run_pipeline(validate_only=validate_only)
