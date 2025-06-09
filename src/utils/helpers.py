# src/utils/helpers.py

import pickle
import pathlib
from typing import Any  # For type hinting

# We don't need to import 'config' here unless these helpers
# specifically need default paths from config, which they currently don't.


def save_object_as_pickle(python_object: Any, file_path: pathlib.Path) -> bool:
    """
    Saves a given Python object to a file using pickle.

    Args:
        python_object (Any): The Python object to be saved.
        file_path (pathlib.Path): The complete file path (including filename and .pkl extension)
                                  where the object should be saved.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    print(f"Attempting to save object to: {file_path}")
    try:
        # Ensure the parent directory of the file exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Open the file in binary write mode ("wb") and dump the object
        with open(file_path, "wb") as f:
            pickle.dump(python_object, f)
        print(f"Object successfully saved to: {file_path}")
        return True
    except Exception as e:
        print(f"Error: Could not save object to {file_path}. Reason: {e}")
        return False


def load_object_from_pickle(file_path: pathlib.Path) -> Any:
    """
    Loads a Python object from a pickle file.

    Args:
        file_path (pathlib.Path): The path to the pickle file.

    Returns:
        Any: The loaded Python object, or None if loading fails or file not found.
    """
    print(f"Attempting to load object from: {file_path}")
    if not file_path.exists():
        print(f"Error: Pickle file not found at {file_path}")
        return None

    try:
        # Open the file in binary read mode ("rb") and load the object
        with open(file_path, "rb") as f:
            loaded_object = pickle.load(f)
        print(f"Object successfully loaded from: {file_path}")
        return loaded_object
    except Exception as e:
        print(f"Error: Could not load object from {file_path}. Reason: {e}")
        return None


def ensure_directory_exists(directory_path: pathlib.Path) -> bool:
    """
    Checks if a directory exists, and creates it if it doesn't.

    Args:
        directory_path (pathlib.Path): The path to the directory.

    Returns:
        bool: True if directory exists or was successfully created, False otherwise.
    """
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        # print(f"Directory ensured/created: {directory_path}") # Optional verbose log
        return True
    except Exception as e:
        print(f"Error: Could not create directory {directory_path}. Reason: {e}")
        return False


if __name__ == "__main__":
    # This block allows you to test this module directly.
    # Run: python -m src.utils.helpers

    print("--- Testing Utilities Module ---")

    # For testing, we need a path. Let's use a path relative to this script
    # or define a temporary test directory.
    # It's better to use paths from config if they are for actual project outputs.
    # For a self-contained test here, let's create a temp test dir.

    # Get the project root from config to create a test path

    from src import config as project_config

    test_dir_base = project_config.PROJECT_ROOT / "outputs" / "temp_utils_test"

    # 1. Test ensure_directory_exists
    print("\n--- Testing ensure_directory_exists ---")
    test_subdir = test_dir_base / "my_test_subdir"
    if ensure_directory_exists(test_subdir):
        print(f"Successfully ensured directory: {test_subdir}")
        if not test_subdir.is_dir():
            print(
                "ERROR: Directory was reported as ensured, but does not exist or is not a directory."
            )
    else:
        print(f"Failed to ensure directory: {test_subdir}")

    # 2. Test save_object_as_pickle and load_object_from_pickle
    print("\n--- Testing save and load pickle ---")
    sample_object_to_save = {
        "message": "Hello from pickle!",
        "data_points": [10, 20, 30],
    }
    pickle_file_path = test_subdir / "my_sample_object.pkl"

    if save_object_as_pickle(sample_object_to_save, pickle_file_path):
        print("Save successful. Now attempting to load...")
        loaded_object = load_object_from_pickle(pickle_file_path)

        if loaded_object is not None:
            if loaded_object == sample_object_to_save:
                print("Load successful! Original and loaded objects match.")
                print(f"Loaded object content: {loaded_object}")
            else:
                print(
                    "ERROR: Load successful, but loaded object does NOT match original."
                )
                print(f"Original: {sample_object_to_save}")
                print(f"Loaded: {loaded_object}")
        else:
            print("ERROR: Loading the pickled object failed.")
    else:
        print("ERROR: Saving the object as pickle failed.")

    # Optional: Clean up the test directory and file
    # import shutil
    # if test_dir_base.exists():
    #     print(f"\nCleaning up test directory: {test_dir_base}")
    #     shutil.rmtree(test_dir_base)
