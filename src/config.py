import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# --- Data Paths ---
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
PLOTS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots"
REPORTS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"

# Ensure output directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Dataset Specific Configuration ---
# Using Heart Disease UCI dataset as an example
DATASET_NAME = "heart_disease_uci.csv"
RAW_DATA_FILE = RAW_DATA_DIR / DATASET_NAME


# --- Feature Engineering & Preprocessing ---
# Based on the Heart Disease UCI dataset columns
# (You might need to adjust these based on the exact CSV column names)
TARGET_COLUMN = "num"  # Or 'condition', 'num' depending on the CSV version
CATEGORICAL_FEATURES = [
    "sex",
    "cp",  # Chest pain type
    "fbs",  # Fasting blood sugar > 120 mg/dl
    "restecg",  # Resting electrocardiographic results
    "exang",  # Exercise induced angina
    "slope",  # The slope of the peak exercise ST segment
    "ca",  # Number of major vessels (0-3) colored by flourosopy
    "thal",  # Thalassemia
]
NUMERICAL_FEATURES = [
    "age",
    "trestbps",  # Resting blood pressure
    "chol",  # Serum cholestoral in mg/dl
    "thalch",  # Maximum heart rate achieved
    "oldpeak",  # ST depression induced by exercise relative to rest
]

# All features to be used for training (excluding target)
FEATURES_TO_USE = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


# HYPERPARAMETERS
# --- Model Training Parameters (Initial placeholders) ---
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# --- DNN Specific Configuration (Placeholders for Part 2) ---
DNN_ARCHITECTURE = {
    "layers": [
        {"units": 64, "activation": "relu", "dropout": 0.2},
        {"units": 32, "activation": "relu", "dropout": 0.1},
    ],
    "output_layer_activation": "sigmoid",  # For binary classification
}
DNN_OPTIMIZER = "adam"
DNN_LOSS = "binary_crossentropy"
DNN_METRICS = ["accuracy"]
DNN_EPOCHS = 50
DNN_BATCH_SIZE = 32


# --- Logging ---
LOG_LEVEL = "INFO"  # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL

# You can add more configurations as the project grows
# For example, different model types, hyperparameter grids, etc.

if __name__ == "__main__":
    # A small test to print out the paths if you run this file directly
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data File: {RAW_DATA_FILE}")
    print(f"Processed Data Dir: {PROCESSED_DATA_DIR}")
    print(f"Categorical Features: {CATEGORICAL_FEATURES}")
    print(f"Numerical Features: {NUMERICAL_FEATURES}")
    print(f"Target Column: {TARGET_COLUMN}")
