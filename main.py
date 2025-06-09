# main.py

import os

# IMPORTANT: Set this environment variable BEFORE importing TensorFlow
# This is a common way to tell TensorFlow (especially with CUDA) to not see any GPUs.
# Its effect on Metal PluggableDevice can vary, but it's a standard first attempt.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf  # Now import TensorFlow

# Explicitly configure TensorFlow to use only CPU after import
# This is a more direct way for TensorFlow itself.
try:
    tf.config.set_visible_devices([], "GPU")
    print("--- TensorFlow: Attempted to disable GPU visibility. ---")
    physical_devices_gpu = tf.config.list_physical_devices("GPU")
    if not physical_devices_gpu:
        print("--- TensorFlow: No GPUs are visible. Running on CPU. ---")
    else:
        print(
            f"--- TensorFlow: WARNING - GPUs still visible: {physical_devices_gpu}. CPU forcing might not be fully effective. ---"
        )
except Exception as e:
    print(
        f"--- TensorFlow: Error during GPU disabling: {e}. Will proceed, may use GPU if available. ---"
    )


# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Import necessary modules from src
from src import config
from src.data_ingestion import loader
from src.eda import exploratory_analysis as eda
from src.preprocessing import transformers
from src.modeling import dnn_builder, trainer
from src.evaluation import plots
from src.utils import helpers


def run_dnn_pipeline(args):
    """
    Orchestrates the full DNN pipeline, attempting to run on CPU only.
    """
    print("--- Starting Full DNN Training Pipeline (CPU ONLY ATTEMPT) ---")

    print(f"--- TensorFlow Version: {tf.__version__} ---")
    # Re-check visible devices to confirm
    print(
        f"--- Visible Physical GPUs (after all configurations): {tf.config.list_physical_devices('GPU')} ---"
    )
    print(f"--- Visible Physical CPUs: {tf.config.list_physical_devices('CPU')} ---")

    # === PART 1: Data Ingestion and Preprocessing ===
    print("\n>>> PART 1: Data Ingestion and Preprocessing <<<")

    # 1. Load Raw Data
    print("\n--- 1.1 Loading Raw Data ---")
    raw_df = loader.load_data_from_csv(csv_file_path=config.RAW_DATA_FILE_PATH)
    if raw_df is None:
        print("ERROR: Failed to load raw data. Exiting.")
        sys.exit(1)
    print(f"Raw data loaded. Shape: {raw_df.shape}")

    # 2. Perform Exploratory Data Analysis (EDA) - Optional
    if args.run_eda:
        print("\n--- 1.2 Performing Exploratory Data Analysis (EDA) ---")
        eda_plots_dir = (
            config.PLOTS_OUTPUT_DIR / "main_pipeline_eda_cpu"
        )  # Changed dir name
        helpers.ensure_directory_exists(eda_plots_dir)

        _ = eda.generate_descriptive_statistics(
            raw_df.copy(), df_name="Raw Data (Main CPU Pipeline)"
        )
        _ = eda.count_missing_values(
            raw_df.copy(), df_name="Raw Data (Main CPU Pipeline)"
        )

        existing_numerical_features_for_plot = [
            col for col in config.NUMERICAL_FEATURES if col in raw_df.columns
        ]
        if existing_numerical_features_for_plot:
            eda.plot_histograms(
                df=raw_df.copy(),
                columns_to_plot=existing_numerical_features_for_plot,
                plot_save_directory=eda_plots_dir,
            )
            eda.plot_correlation_heatmap(
                df=raw_df.copy(),
                columns_for_corr=existing_numerical_features_for_plot,
                plot_save_path=eda_plots_dir / "correlation_matrix_main_cpu.png",
            )
        else:
            print(
                "No numerical features (from config) found in the DataFrame to generate EDA plots."
            )
        print(f"EDA reports and plots (if any generated) saved to: {eda_plots_dir}")
    else:
        print("\n--- Skipping EDA as per --no-eda flag ---")

    # 3. Split Data
    print("\n--- 1.3 Splitting Data ---")
    X_train_raw_df, X_test_raw_df, y_train_raw_series, y_test_raw_series = (
        transformers.split_dataframe_into_train_test(
            dataframe=raw_df,
            target_column_name=config.TARGET_COLUMN,
            test_set_ratio=config.TEST_SET_SIZE,
            random_seed=config.RANDOM_SEED,
            stratify_by_target=True,
        )
    )

    # 4. Create Feature Preprocessor object
    print("\n--- 1.4 Creating Feature Preprocessor ---")
    feature_preprocessor_obj = transformers.create_feature_preprocessor(
        numerical_cols=config.NUMERICAL_FEATURES,
        categorical_cols=config.CATEGORICAL_FEATURES,
    )

    # 5. Apply Feature Preprocessing
    print("\n--- 1.5 Applying Feature Preprocessing (Output to NumPy) ---")
    X_train_np, X_test_np = transformers.apply_feature_preprocessing_to_numpy(
        X_train_df=X_train_raw_df.copy(),
        X_test_df=X_test_raw_df.copy(),
        preprocessor_object=feature_preprocessor_obj,
        fit_preprocessor_on_train=True,
    )
    if helpers.save_object_as_pickle(
        feature_preprocessor_obj, config.PREPROCESSOR_SAVE_PATH
    ):
        print(f"Fitted feature preprocessor saved to: {config.PREPROCESSOR_SAVE_PATH}")
    else:
        print(
            f"ERROR: Failed to save feature preprocessor to {config.PREPROCESSOR_SAVE_PATH}"
        )

    # 6. Prepare Target Variable
    print("\n--- 1.6 Preparing Target Variable (Output to NumPy) ---")
    y_train_np = transformers.prepare_target_to_numpy(y_train_raw_series.copy())
    y_test_np = transformers.prepare_target_to_numpy(y_test_raw_series.copy())
    print(
        f"Data prepared for modeling. X_train_np shape: {X_train_np.shape}, y_train_np shape: {y_train_np.shape}"
    )

    # === PART 2: DNN Modeling, Training & Basic Evaluation ===
    print("\n>>> PART 2: DNN Modeling, Training & Basic Evaluation <<<")

    # 1. Build DNN Model
    print("\n--- 2.1 Building DNN Model ---")
    input_shape_for_model = (X_train_np.shape[1],)
    dnn_model = dnn_builder.build_dynamic_dnn_model(
        input_features_shape=input_shape_for_model,
        model_arch_params=config.DNN_MODEL_PARAMS["architecture"],
        compilation_params=config.DNN_MODEL_PARAMS["compilation"],
    )

    # 2. Train DNN Model
    print("\n--- 2.2 Training DNN Model (Attempting CPU Only) ---")
    main_model_save_path = (
        config.MODEL_OUTPUT_DIR / f"main_dnn_model_multiclass_cpu.keras"
    )
    main_history_save_path = (
        config.PROCESSED_DATA_DIR / f"main_dnn_training_history_multiclass_cpu.pkl"
    )
    main_tensorboard_log_dir = (
        config.PROJECT_ROOT / "outputs" / "logs" / "main_dnn_multiclass_cpu_run"
    )

    validation_data_for_training = (X_test_np, y_test_np)

    # --- Simplify Callbacks for this CPU test to minimize variables ---
    print("--- Using minimal callbacks for CPU test (EarlyStopping only) ---")
    training_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        )
    ]
    # training_callbacks = trainer.create_standard_callbacks(
    #     early_stopping_params={"monitor": "val_accuracy", "patience": 10, "restore_best_weights": True},
    #     model_checkpoint_filepath=main_model_save_path, # Keep None if ModelCheckpoint is suspect
    #     model_checkpoint_params={"monitor": "val_accuracy", "save_best_only": True},
    #     tensorboard_logdir=main_tensorboard_log_dir # Keep None if Tensorboard is suspect
    # )

    trained_model, training_history = trainer.train_keras_model(
        model_to_train=dnn_model,
        X_train_data=X_train_np,
        y_train_data=y_train_np,
        training_params=config.DNN_MODEL_PARAMS["training"],
        validation_data_tuple=validation_data_for_training,
        callbacks_to_use=training_callbacks,
        history_log_path=main_history_save_path,
    )
    # Check if model was saved by ModelCheckpoint (if it was enabled)
    # if main_model_save_path.exists():
    #     print(f"Model training complete. Best model potentially saved to: {main_model_save_path}")
    # else:
    #     print(f"Model training complete. ModelCheckpoint was not used or did not save a model to {main_model_save_path}.")
    print("Model training completed.")

    # 3. Plot Training History
    print("\n--- 2.3 Plotting Training History ---")
    if training_history and training_history.history:
        history_plot_path = (
            config.PLOTS_OUTPUT_DIR / "main_dnn_training_history_multiclass_cpu.png"
        )
        plots.plot_training_history(
            history_data=training_history.history,
            plot_title="DNN Model Training History (CPU Run from main.py)",
            save_plot_path=history_plot_path,
        )
    else:
        print(
            "Skipping plotting training history as history object or data is not available."
        )

    print("\n--- Full DNN Training Pipeline (CPU ONLY ATTEMPT) COMPLETED ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the DNN Training Pipeline on CPU."
    )
    parser.add_argument(
        "--no-eda",
        action="store_false",
        dest="run_eda",
        default=True,
        help="Skip the Exploratory Data Analysis (EDA) step.",
    )
    cli_args = parser.parse_args()
    run_dnn_pipeline(cli_args)
