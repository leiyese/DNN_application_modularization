# src/data_ingestion/loader.py

import pandas as pd
import pathlib
from typing import Optional, Dict, Any

# Import the configuration.
# Since loader.py is in src/data_ingestion/ and config.py is in src/,
# we can use a relative import if we treat 'src' as the top-level package
# when running scripts from the project root.
# However, for broader compatibility and easier understanding,
# let's assume we'll add 'src' to sys.path or run main.py from the root.
# For direct execution or testing within src, this might need adjustment.
# A common practice is to structure imports as if 'src' is a package.
from src import (
    config,
)  # This assumes 'src' is in PYTHONPATH or you run from project root


def load_csv_data(
    file_path: pathlib.Path, pandas_read_csv_kwargs: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path (pathlib.Path): The path to the CSV file.
        pandas_read_csv_kwargs (Optional[Dict[str, Any]]): Optional dictionary of
            keyword arguments to pass directly to pd.read_csv().

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
        pd.errors.EmptyDataError: If the CSV file is empty.
        Exception: For other pandas related read errors.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Error: The file was not found at {file_path}")

    if pandas_read_csv_kwargs is None:
        pandas_read_csv_kwargs = {}

    try:
        df = pd.read_csv(file_path, **pandas_read_csv_kwargs)
        if df.empty:
            # pd.read_csv might not raise EmptyDataError for a file with only headers
            # depending on the pandas version and arguments.
            # Explicitly check if the DataFrame (post-header) is empty.
            raise pd.errors.EmptyDataError(f"No data found in CSV file: {file_path}")
        print(f"Successfully loaded data from: {file_path}")
        print(f"DataFrame shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError as e:
        # This catches files that are completely empty or just have headers
        # and pandas considers them empty.
        print(f"Pandas EmptyDataError: {e}")
        raise
    except Exception as e:
        # Catch other potential pandas errors during read_csv
        print(f"Error loading CSV file {file_path}: {e}")
        raise


if __name__ == "__main__":
    # This block is for testing the loader module directly.
    # To run this:
    # 1. Make sure your conda environment is activated.
    # 2. Navigate to the 'dynamic_dnn_trainer' project root directory in your terminal.
    # 3. Run: python -m src.data_ingestion.loader

    print("--- Testing load_csv_data function ---")

    # Use the raw data file path from our config
    # Note: config.RAW_DATA_FILE is already a pathlib.Path object
    try:
        # Example: Load the heart disease dataset
        # Ensure data/raw/heart_disease_uci.csv exists
        if config.RAW_DATA_FILE.exists():
            heart_data = load_csv_data(file_path=config.RAW_DATA_FILE)
            print("\nHeart Disease Data (first 5 rows):")
            print(heart_data.head())

            # Example: Test with non-existent file
            # print("\n--- Testing with a non-existent file ---")
            # non_existent_file = config.RAW_DATA_DIR / "non_existent.csv"
            # load_csv_data(file_path=non_existent_file)

        else:
            print(f"Test data file not found: {config.RAW_DATA_FILE}")
            print(
                "Please ensure 'heart_disease_uci.csv' is in the 'data/raw/' directory."
            )

    except FileNotFoundError as e:
        print(e)
    except pd.errors.EmptyDataError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
