import os
import sys
import json
import pandas as pd
from datetime import datetime

from logger.logger import get_logger
from datapipeline.products.p_fetcher import fetch_data
from datapipeline.products.p_preprocess import create_preprocessing_pipeline, save_processed_data
from datapipeline.products.p_quality import analyze_data, data_quality_checks, summarize_data
from datapipeline.products.p_validation import validate_raw_data, get_validation_summary
from utils.utils import load_config
from utils.data_drift_detector import check_and_update_reference

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
        logger.info(f"Data will be saved to directory: {data_dir}")

        # Get reports directory from config
        reports_dir = config.get("reports_dir", os.path.join(data_dir, "reports"))
        os.makedirs(reports_dir, exist_ok=True)
        logger.info(f"Reports will be saved to directory: {reports_dir}")

        # Use data dir from config to store reference data for drift detection
        reference_data_path = os.path.join(data_dir, "reference_product_data.csv")

        # Step 1: Fetch data
        logger.info("Step 1: Fetching data from database...")
        raw_data = fetch_data()

        # Step 2: Validate raw data
        logger.info("Step 2: Validating raw data...")
        is_valid, validation_results = validate_raw_data(raw_data)
        validation_summary = get_validation_summary(validation_results)

        if not is_valid:
            logger.error("Data validation failed. See logs for details.")
            logger.info(f"Validation summary: {validation_summary}")
            if validate_only or '--strict' in sys.argv:
                logger.error("Aborting pipeline due to validation failure.")
                return None
            else:
                logger.warning("Continuing pipeline despite validation failures.")

        if validate_only:
            logger.info("Validation completed successfully.")
            return validation_results

        # Step 3: Analyze initial data quality
        logger.info("Step 3: Analyzing initial data quality...")
        initial_stats = analyze_data(raw_data)
        logger.info(f"Initial data analysis complete. Found {sum(initial_stats['missing_values'].values())} missing values.")

        # Step 4: Preprocessing
        logger.info("Step 4: Setting up and applying preprocessing pipeline...")
        preprocessing_pipeline = create_preprocessing_pipeline()
        processed_data = preprocessing_pipeline.fit_transform(raw_data)

        # Step 5: Data quality checks on processed data
        logger.info("Step 5: Performing data quality checks...")
        quality_issues = data_quality_checks(processed_data)
        if quality_issues:
            logger.warning("Data quality issues found:")
            for issue in quality_issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No data quality issues found.")

        # Step 6: Generate data summary
        logger.info("Step 6: Generating data summary...")
        data_summary = summarize_data(processed_data)
        logger.info(f"Data summary: {data_summary}")

        # Step 7: Save processed data
        logger.info("Step 7: Saving processed data...")
        save_processed_data(processed_data, data_dir)

        # Step 8: Save validation report to the reports directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        validation_report_path = os.path.join(reports_dir, f"validation_report_{timestamp}.json")
        with open(validation_report_path, 'w') as f:
            json.dump(validation_summary, f, indent=2)
        logger.info(f"Validation report saved to: {validation_report_path}")

        # Step 9: Data drift / change detection
        logger.info("Step 9: Checking if new data arrived using reference CSV...")
        data_changed = check_and_update_reference(
            df=processed_data,
            reference_data_path=reference_data_path,
            id_column='product_id'
        )

        # Final exit code decision
        if data_changed:
            logger.info("New data detected; exiting with code 1 to trigger downstream pipelines.")
            sys.exit(1)
        else:
            logger.info("No new data detected; exiting with code 0.")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(2)

if __name__ == "__main__":
    validate_only = '--validate-only' in sys.argv
    run_pipeline(validate_only=validate_only)
