# src/data_ingestion/loader.py

import pandas as pd
import pathlib
from typing import Optional, Dict, Any  # For type hinting

# Import our configuration settings
from src import config


def load_data_from_csv(
    csv_file_path: pathlib.Path, read_csv_options: Optional[Dict[str, Any]] = None
) -> Optional[pd.DataFrame]:
    """
    Loads data from a specified CSV file into a pandas DataFrame.

    Args:
        csv_file_path (pathlib.Path): The full path to the CSV file.
        read_csv_options (Optional[Dict[str, Any]]): A dictionary of optional
            keyword arguments to pass to pandas.read_csv() function.
            Example: {'sep': ';', 'header': None}

    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame containing the loaded data,
                                or None if loading fails.
    """
    print(f"Attempting to load data from: {csv_file_path}")

    if not csv_file_path.exists():
        print(f"Error: File not found at {csv_file_path}")
        return None

    # Use an empty dictionary if no options are provided
    if read_csv_options is None:
        read_csv_options = {}

    try:
        # Load the CSV file into a DataFrame
        dataframe = pd.read_csv(csv_file_path, **read_csv_options)

        if dataframe.empty:
            print(
                f"Warning: The CSV file at {csv_file_path} is empty or contains only headers."
            )
            return None  # Or an empty DataFrame: pd.DataFrame()

        print(f"Successfully loaded data. DataFrame shape: {dataframe.shape}")
        return dataframe

    except pd.errors.EmptyDataError:
        print(f"Error: No data or columns to parse in CSV file: {csv_file_path}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during file loading
        print(f"An unexpected error occurred while loading {csv_file_path}: {e}")
        return None


if __name__ == "__main__":
    # This block allows you to test this module directly.
    # How to run for testing:
    # 1. Activate your conda environment: conda activate dynamic_dnn_env
    # 2. Navigate to your project root: cd path/to/dynamic_dnn_trainer
    # 3. Run: python -m src.data_ingestion.loader

    print("--- Testing Data Loader Module ---")

    # We use RAW_DATA_FILE_PATH from our config.py
    loaded_df = load_data_from_csv(csv_file_path=config.RAW_DATA_FILE_PATH)

    if loaded_df is not None:
        print("\nSample of loaded data (first 5 rows):")
        print(loaded_df.head())
        print(f"\nLoaded DataFrame info:")
        loaded_df.info()
    else:
        print("\nData loading failed during test.")

    # Example of testing with a non-existent file (optional)
    # print("\n--- Testing with a non-existent file ---")
    # non_existent_path = config.RAW_DATA_DIR / "does_not_exist.csv"
    # load_data_from_csv(csv_file_path=non_existent_path)
