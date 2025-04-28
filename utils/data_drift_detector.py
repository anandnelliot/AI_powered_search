import os
import pandas as pd

def save_reference_data(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def load_reference_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def has_data_changed(
    reference_df: pd.DataFrame,
    new_df: pd.DataFrame,
    id_column: str = 'product_id'
) -> bool:
    """
    Checks if new unique entries are detected based on distinct values in the id_column.

    Args:
        reference_df: The previously saved DataFrame for comparison.
        new_df: The new DataFrame to compare against the reference.
        id_column: The column used to detect new unique entries.

    Returns:
        bool: True if new data is detected, False otherwise.
    """
    ref_items = set(reference_df[id_column].unique())
    new_items = set(new_df[id_column].unique())

    return bool(new_items - ref_items)

def check_and_update_reference(
    df: pd.DataFrame,
    reference_data_path: str,
    id_column: str = 'product_id'
) -> bool:
    """
    Checks for new data and updates the reference data if new data is found.

    Args:
        df: The DataFrame to check.
        reference_data_path: Path to the stored reference data CSV.
        id_column: The column used to detect new unique entries.

    Returns:
        bool: True if new data detected and reference updated, False otherwise.
    """
    if os.path.exists(reference_data_path):
        reference_df = load_reference_data(reference_data_path)
        data_changed = has_data_changed(reference_df, df, id_column)

        if data_changed:
            save_reference_data(df, reference_data_path)
            return True
        else:
            return False
    else:
        save_reference_data(df, reference_data_path)
        return True
