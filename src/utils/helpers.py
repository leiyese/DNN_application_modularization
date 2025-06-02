# src/utils/helpers.py

import pickle
import pathlib
from typing import Any
import os


def save_pickle(obj: Any, filepath: pathlib.Path) -> None:
    """
    Saves a Python object to a pickle file.

    Args:
        obj (Any): The Python object to save.
        filepath (pathlib.Path): The path (including filename) to save the pickle file.
    """
    # Ensure the directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        print(f"Successfully saved object to: {filepath}")
    except Exception as e:
        print(f"Error saving object to {filepath}: {e}")
        raise


def load_pickle(filepath: pathlib.Path) -> Any:
    """
    Loads a Python object from a pickle file.

    Args:
        filepath (pathlib.Path): The path to the pickle file.

    Returns:
        Any: The loaded Python object.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
        Exception: For other pickle loading errors.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Error: Pickle file not found at {filepath}")
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        print(f"Successfully loaded object from: {filepath}")
        return obj
    except Exception as e:
        print(f"Error loading object from {filepath}: {e}")
        raise


def ensure_dir_exists(dir_path: pathlib.Path) -> None:
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        dir_path (pathlib.Path): The path to the directory.
    """
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        # print(f"Directory ensured/created: {dir_path}") # Optional: for verbose logging
    except Exception as e:
        print(f"Error creating directory {dir_path}: {e}")
        raise


if __name__ == "__main__":
    # For testing, Run: python -m src.utils.helpers

    print("--- Testing utility functions ---")

    # Define a test directory and file path within outputs for testing
    # This uses the config.py for base paths, so import it.
    # We need to be careful with imports if running directly vs. as part of a larger app.
    # For 'python -m src.utils.helpers' from project root, 'from src import config' should work.
    try:
        from src import config as cfg  # Use an alias to avoid name clashes if any

        test_output_dir = cfg.PROJECT_ROOT / "outputs" / "test_utils"
        test_pickle_file = test_output_dir / "test_object.pkl"

        # Test ensure_dir_exists
        print(f"\n--- Testing ensure_dir_exists for: {test_output_dir} ---")
        ensure_dir_exists(test_output_dir)
        if test_output_dir.exists() and test_output_dir.is_dir():
            print("ensure_dir_exists test PASSED.")
        else:
            print("ensure_dir_exists test FAILED.")

        # Test save_pickle and load_pickle
        print(f"\n--- Testing save_pickle and load_pickle with: {test_pickle_file} ---")
        sample_data = {"name": "Test Object", "version": 1.0, "data": [1, 2, 3, 4, 5]}

        save_pickle(sample_data, test_pickle_file)
        loaded_data = load_pickle(test_pickle_file)

        if loaded_data == sample_data:
            print("save_pickle and load_pickle test PASSED.")
            print(f"Loaded data: {loaded_data}")
        else:
            print("save_pickle and load_pickle test FAILED.")
            print(f"Original: {sample_data}")
            print(f"Loaded: {loaded_data}")

        # Clean up the test file and directory (optional)
        # if test_pickle_file.exists():
        #     os.remove(test_pickle_file)
        # if test_output_dir.exists() and not any(test_output_dir.iterdir()): # only remove if empty
        #     os.rmdir(test_output_dir)
        # print("\nCleaned up test files/directories.")

    except ImportError:
        print("Could not import 'src.config'. Make sure you are running this test")
        print("from the project root directory using 'python -m src.utils.helpers'")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
