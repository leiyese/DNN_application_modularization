# src/modeling/trainer.py

import tensorflow as tf

print(f"--- Trainer.py: TensorFlow Version: {tf.__version__} ---")
# Uncomment the next line to force CPU execution for debugging
# tf.config.set_visible_devices([], "GPU")
# if not tf.config.list_physical_devices("GPU"):
#     print("--- Trainer.py: Running on CPU (GPU not available or disabled) ---")
# else:
#     print("--- Trainer.py: GPU is available ---")

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import pandas as pd
import numpy as np
import pathlib
from typing import Tuple, Optional, Dict, Any, Union, List

# Import configurations and helper functions
from src import config
from src.utils import helpers


def train_keras_model(
    model_to_train: tf.keras.Model,
    X_train_data: Union[pd.DataFrame, np.ndarray],
    y_train_data: Union[pd.Series, np.ndarray],
    training_params: Dict[str, Any],  # e.g., epochs, batch_size
    validation_data_tuple: Optional[
        Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]
    ] = None,
    callbacks_to_use: Optional[List[tf.keras.callbacks.Callback]] = None,
    history_log_path: Optional[pathlib.Path] = None,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trains a given Keras model with the provided data and parameters.

    Args:
        model_to_train (tf.keras.Model): The compiled Keras model.
        X_train_data (Union[pd.DataFrame, np.ndarray]): Training features.
        y_train_data (Union[pd.Series, np.ndarray]): Training target labels.
        training_params (Dict[str, Any]): Dictionary containing training parameters
                                          like 'epochs' and 'batch_size'.
        validation_data_tuple (Optional[Tuple]): Tuple containing (X_val, y_val)
                                                 for validation during training.
        callbacks_to_use (Optional[List[tf.keras.callbacks.Callback]]): A list of
                            Keras callbacks to use during training.
        history_log_path (Optional[pathlib.Path]): Path to save the training
                                                   history object (as a pickle file).

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: The trained Keras model
                                                           and the training history object.
    """
    print("--- Starting Model Training ---")

    epochs = training_params.get("epochs", 10)  # Default to 10 epochs if not specified
    batch_size = training_params.get("batch_size", 32)  # Default to 32 batch size

    if validation_data_tuple:
        print(
            f"  Using validation data: X_val shape {validation_data_tuple[0].shape}, y_val shape {validation_data_tuple[1].shape}"
        )
    else:
        print("  No validation data provided.")
        if callbacks_to_use:
            print(
                "  Warning: Callbacks monitoring validation metrics (e.g., 'val_loss') might not work as expected."
            )

    print(f"  Training for {epochs} epochs with batch_size {batch_size}.")

    # Fit the model
    training_history = model_to_train.fit(
        X_train_data,
        y_train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data_tuple,
        callbacks=(
            callbacks_to_use if callbacks_to_use else []
        ),  # Pass empty list if None
        verbose=1,  # 0=silent, 1=progress bar, 2=one line per epoch
    )

    print("\n--- Model Training Completed ---")

    # Save the training history dictionary
    if history_log_path:
        if helpers.save_object_as_pickle(training_history.history, history_log_path):
            print(f"Training history successfully saved to: {history_log_path}")
        else:
            print(f"Failed to save training history to: {history_log_path}")

    return model_to_train, training_history


# --- Helper function to create standard callbacks (can be expanded) ---
def create_standard_callbacks(
    early_stopping_params: Optional[Dict[str, Any]] = None,
    model_checkpoint_filepath: Optional[pathlib.Path] = None,
    model_checkpoint_params: Optional[Dict[str, Any]] = None,
    tensorboard_logdir: Optional[pathlib.Path] = None,
) -> List[tf.keras.callbacks.Callback]:
    """
    Creates a list of standard Keras callbacks.
    """
    active_callbacks = []

    # 1. Early Stopping
    if early_stopping_params:
        es_monitor = early_stopping_params.get("monitor", "val_loss")
        es_patience = early_stopping_params.get("patience", 10)
        es_restore_best = early_stopping_params.get("restore_best_weights", True)
        early_stopping = EarlyStopping(
            monitor=es_monitor,
            patience=es_patience,
            verbose=1,
            restore_best_weights=es_restore_best,
        )
        active_callbacks.append(early_stopping)
        print(
            f"Callback: EarlyStopping enabled (monitor='{es_monitor}', patience={es_patience})."
        )

    # 2. Model Checkpoint
    if model_checkpoint_filepath:
        model_checkpoint_filepath.parent.mkdir(parents=True, exist_ok=True)
        mc_monitor = (
            model_checkpoint_params.get("monitor", "val_loss")
            if model_checkpoint_params
            else "val_loss"
        )
        mc_save_best = (
            model_checkpoint_params.get("save_best_only", True)
            if model_checkpoint_params
            else True
        )
        mc_save_weights = (
            model_checkpoint_params.get("save_weights_only", False)
            if model_checkpoint_params
            else False
        )
        model_checkpoint = ModelCheckpoint(
            filepath=str(model_checkpoint_filepath),  # Needs string path
            monitor=mc_monitor,
            save_best_only=mc_save_best,
            save_weights_only=mc_save_weights,
            verbose=1,
        )
        active_callbacks.append(model_checkpoint)
        print(
            f"Callback: ModelCheckpoint enabled (saving to '{model_checkpoint_filepath}', monitor='{mc_monitor}')."
        )

    # 3. TensorBoard
    if tensorboard_logdir:
        tensorboard_logdir.mkdir(parents=True, exist_ok=True)
        # Basic TensorBoard setup
        tensorboard_callback = TensorBoard(
            log_dir=str(tensorboard_logdir), histogram_freq=1
        )
        active_callbacks.append(tensorboard_callback)
        print(
            f"Callback: TensorBoard logging enabled (logs at '{tensorboard_logdir}')."
        )

    return active_callbacks


if __name__ == "__main__":

    # This block allows you to test this module directly.
    # Test run: python -m src.modeling.trainer
    print("--- Testing Model Trainer Module (Multi-Class Focus) ---")

    # 1. Build a dummy model using our dnn_builder
    # Ensure dnn_builder is imported correctly
    from src.modeling import dnn_builder

    dummy_input_shape = (15,)  # Example: 15 features after preprocessing

    # Use model params from config, which are set for multi-class
    # config.DNN_MODEL_PARAMS["architecture"]["output_layer_units"] should be config.NUM_CLASSES
    dummy_model_for_training = dnn_builder.build_dynamic_dnn_model(
        input_features_shape=dummy_input_shape,
        model_arch_params=config.DNN_MODEL_PARAMS["architecture"],
        compilation_params=config.DNN_MODEL_PARAMS["compilation"],
    )

    # 2. Create dummy training and validation data
    num_train_samples = 200
    num_val_samples = 50
    num_features = dummy_input_shape[0]
    num_classes = config.NUM_CLASSES  # Should be 5 from our config

    X_train_dummy_np = np.random.rand(num_train_samples, num_features).astype(
        np.float32
    )
    # For sparse_categorical_crossentropy, y should be integers 0 to NUM_CLASSES-1
    y_train_dummy_np = np.random.randint(0, num_classes, num_train_samples).astype(
        np.int64
    )

    X_val_dummy_np = np.random.rand(num_val_samples, num_features).astype(np.float32)
    y_val_dummy_np = np.random.randint(0, num_classes, num_val_samples).astype(np.int64)
    dummy_validation_data = (X_val_dummy_np, y_val_dummy_np)

    # 3. Define paths for test outputs
    test_model_save_path = config.MODEL_OUTPUT_DIR / "test_trained_model.keras"
    test_history_save_path = config.PROCESSED_DATA_DIR / "test_training_history.pkl"
    test_tensorboard_logs_dir = (
        config.PROJECT_ROOT / "outputs" / "logs" / "test_trainer_run"
    )

    # 4. Create callbacks for testing
    test_early_stopping_params = {"monitor": "val_accuracy", "patience": 3}
    test_model_checkpoint_params = {"monitor": "val_accuracy", "save_best_only": True}

    test_callbacks = create_standard_callbacks(
        early_stopping_params=test_early_stopping_params,
        model_checkpoint_filepath=test_model_save_path,
        model_checkpoint_params=test_model_checkpoint_params,
        tensorboard_logdir=test_tensorboard_logs_dir,
    )

    # 5. Get training parameters from config
    # Using a sub-dictionary for training params from config makes it cleaner
    training_parameters_from_config = config.DNN_MODEL_PARAMS["training"].copy()
    training_parameters_from_config["epochs"] = 5  # Override epochs for quick test

    # 6. Train the dummy model
    print("\nStarting dummy model training for test...")
    trained_dummy_model, history_object = train_keras_model(
        model_to_train=dummy_model_for_training,
        X_train_data=X_train_dummy_np,
        y_train_data=y_train_dummy_np,
        training_params=training_parameters_from_config,
        validation_data_tuple=dummy_validation_data,
        callbacks_to_use=test_callbacks,
        history_log_path=test_history_save_path,
    )

    print("\n--- Trainer Test Output ---")
    print(f"Trained model type: {type(trained_dummy_model)}")
    if history_object:
        print(f"Training history keys: {history_object.history.keys()}")

    if test_model_save_path.exists():
        print(f"Best model during test run saved at: {test_model_save_path}")
    if test_history_save_path.exists():
        print(f"Training history for test run saved at: {test_history_save_path}")
    if test_tensorboard_logs_dir.exists() and any(test_tensorboard_logs_dir.iterdir()):
        print(
            f"TensorBoard logs for test run generated at: {test_tensorboard_logs_dir}"
        )
        print("  To view: tensorboard --logdir outputs/logs")

    print("\nModel Trainer module test completed.")
