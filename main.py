# main.py

import argparse
import sys

# Add src to Python path. This is one way to do it when main.py is at the project root.
# More robust solutions might involve installing your src as a package.
# For now, this direct manipulation of sys.path is common for scripts.
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Now we can import from src
from src import config
from src.data_ingestion import loader
from src.eda import exploratory_analysis as eda
from src.preprocessing import transformers
from src.utils import helpers  # Though not directly used in this main_v1


def run_part1_pipeline(args):
    """
    Orchestrates the data loading, EDA, and preprocessing steps.
    """
    print("--- Starting Part 1: Data Ingestion and Preprocessing Pipeline ---")

    # 1. Load Raw Data
    print("\n--- 1. Loading Raw Data ---")
    if not config.RAW_DATA_FILE.exists():
        print(f"ERROR: Raw data file not found at {config.RAW_DATA_FILE}. Exiting.")
        sys.exit(1)
    raw_df = loader.load_csv_data(file_path=config.RAW_DATA_FILE)
    print(f"Raw data loaded. Shape: {raw_df.shape}")

    # 2. Perform Exploratory Data Analysis (EDA)
    if args.run_eda:
        print("\n--- 2. Performing Exploratory Data Analysis (EDA) ---")
        eda_plots_dir = config.PLOTS_OUTPUT_DIR / "main_pipeline_eda"
        eda_plots_dir.mkdir(parents=True, exist_ok=True)

        _ = eda.generate_descriptive_stats(
            raw_df.copy()
        )  # Use .copy() if EDA modifies df
        _ = eda.get_null_counts(raw_df.copy())
        eda.plot_histograms_for_numerical_features(
            df=raw_df.copy(),
            numerical_features=config.NUMERICAL_FEATURES,
            save_dir=eda_plots_dir,
        )
        eda.plot_correlation_matrix(
            df=raw_df.copy(),
            numerical_features=config.NUMERICAL_FEATURES,
            save_path=eda_plots_dir / "correlation_matrix_main.png",
        )
        print(f"EDA reports and plots saved to: {eda_plots_dir}")
    else:
        print("\n--- Skipping EDA as per --no-eda flag ---")

    # 3. Split Data
    print("\n--- 3. Splitting Data ---")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = transformers.split_data(
        df=raw_df,
        target_column=config.TARGET_COLUMN,
        test_size=config.TEST_SPLIT_SIZE,
        random_state=config.RANDOM_STATE,
        stratify_col=raw_df[config.TARGET_COLUMN],
    )
    print(f"Data split. X_train_raw shape: {X_train_raw.shape}")

    # 4. Preprocess Features
    print("\n--- 4. Preprocessing Features ---")
    # Define paths for saving fitted preprocessor and target encoder from this main run
    main_preprocessor_path = config.PROCESSED_DATA_DIR / "main_fitted_preprocessor.pkl"
    X_train_proc, X_test_proc, fitted_preprocessor = transformers.preprocess_features(
        X_train=X_train_raw.copy(),
        X_test=X_test_raw.copy(),
        numerical_features=config.NUMERICAL_FEATURES,
        categorical_features=config.CATEGORICAL_FEATURES,
        preprocessor_save_path=main_preprocessor_path,
        fit_preprocessor=True,  # Always fit when running this main script for now
    )
    print(f"Feature preprocessing complete. X_train_proc shape: {X_train_proc.shape}")  # type: ignore
    print(f"Fitted preprocessor saved to: {main_preprocessor_path}")

    # 5. Encode Target Variable
    print("\n--- 5. Encoding Target Variable ---")
    main_target_encoder_path = (
        config.PROCESSED_DATA_DIR / "main_fitted_target_encoder.pkl"
    )
    y_train_enc, y_test_enc, fitted_target_encoder = transformers.encode_target(
        y_train=y_train_raw.copy(),
        y_test=y_test_raw.copy(),
        encoder_save_path=main_target_encoder_path,
        fit_encoder=True,  # Always fit when running this main script for now
    )
    print(f"Target encoding complete. y_train_enc shape: {y_train_enc.shape}")
    print(f"Fitted target encoder saved to: {main_target_encoder_path}")

    # For Part 1, we stop here. Processed data is ready.
    # We can save X_train_proc, X_test_proc, y_train_enc, y_test_enc if needed.
    # For example:
    # helpers.save_pickle(X_train_proc, config.PROCESSED_DATA_DIR / "X_train_processed.pkl")
    # helpers.save_pickle(y_train_enc, config.PROCESSED_DATA_DIR / "y_train_encoded.pkl")
    # helpers.save_pickle(X_test_proc, config.PROCESSED_DATA_DIR / "X_test_processed.pkl")
    # helpers.save_pickle(y_test_enc, config.PROCESSED_DATA_DIR / "y_test_encoded.pkl")
    # print(f"Processed data splits saved to {config.PROCESSED_DATA_DIR}")

    print("\n--- Part 1: Data Ingestion and Preprocessing Pipeline COMPLETED ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Part 1 of the DNN training pipeline: Data Ingestion & Preprocessing."
    )
    parser.add_argument(
        "--no-eda",
        action="store_false",  # Default is True (run_eda=True)
        dest="run_eda",  # If --no-eda is present, run_eda becomes False
        help="Skip the EDA step.",
    )
    # We can add more arguments later, e.g., --config-file

    args = parser.parse_args()
    run_part1_pipeline(args)
