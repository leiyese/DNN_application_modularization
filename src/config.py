# src/config.py

import pathlib

# --- Path Setup ---
# Get the root directory of the project (dynamic_dnn_trainer)
# This assumes config.py is in dynamic_dnn_trainer/src/
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Define directories for data and outputs
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
PLOTS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots"
REPORTS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"  # For later

# --- Dataset Information ---
DATASET_FILENAME = "heart_disease_uci.csv"
RAW_DATA_FILE_PATH = RAW_DATA_DIR / DATASET_FILENAME

# --- Feature Definitions ---
# IMPORTANT: Verify these column names against your actual CSV file!
# Target column for multi-class classification (e.g., values 0, 1, 2, 3, 4)
TARGET_COLUMN = "num"  # Example: 'num' or 'target' or 'condition'

# List of columns that are categorical
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# List of columns that are numerical
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# --- Data Splitting Parameters ---
TEST_SET_SIZE = 0.2  # Proportion of data to allocate to the test set (e.g., 20%)
RANDOM_SEED = 42  # Seed for random operations to ensure reproducibility

# --- Preprocessing Artifact Paths ---
# These are paths where we will save our fitted preprocessor objects
PREPROCESSOR_FILENAME = "feature_preprocessor.pkl"
PREPROCESSOR_SAVE_PATH = PROCESSED_DATA_DIR / PREPROCESSOR_FILENAME

# For multi-class, if target is already 0,1,2,3,4, LabelEncoder might not be strictly
# necessary but can be used to ensure consistent integer mapping if original target was string.
# Let's assume our 'num' column is already integers 0-4.
# If your target was strings, you'd definitely use and save a LabelEncoder.
# TARGET_ENCODER_FILENAME = "target_encoder.pkl"
# TARGET_ENCODER_SAVE_PATH = PROCESSED_DATA_DIR / TARGET_ENCODER_FILENAME


# --- Model Configuration (Placeholders for Part 2 - Multi-Class) ---
# Number of unique classes in our target variable
# For 'num' column with values 0, 1, 2, 3, 4, this is 5.
NUM_CLASSES = 5

DNN_MODEL_PARAMS = {
    "architecture": {
        "layers": [
            {"units": 64, "activation": "relu", "dropout_rate": 0.2},
            {"units": 32, "activation": "relu", "dropout_rate": 0.1},
        ],
        # For multi-class classification with NUM_CLASSES outputs
        "output_layer_units": NUM_CLASSES,
        "output_layer_activation": "softmax",
    },
    "compilation": {
        "optimizer": "adam",
        # For integer targets (0, 1, 2, ... N-1)
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
    },
    "training": {"epochs": 50, "batch_size": 32},  # Example, will adjust
}


# --- Helper to print config for verification ---
def print_config_summary():
    print("--- Configuration Summary ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data File: {RAW_DATA_FILE_PATH}")
    print(f"Target Column: {TARGET_COLUMN}")
    print(f"Numerical Features: {NUMERICAL_FEATURES}")
    print(f"Categorical Features: {CATEGORICAL_FEATURES}")
    print(f"Number of Classes for Model: {NUM_CLASSES}")
    print("-----------------------------")


if __name__ == "__main__":
    # This part runs if you execute "python src/config.py" directly
    # Useful for a quick check of your paths and settings.
    print_config_summary()

    # Ensure output directories exist (good practice to put this here or in main script)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nOutput directories ensured.")
