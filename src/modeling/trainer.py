# src/modeling/trainer.py

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import pandas as pd  # For X_train, y_train types
import numpy as np  # For X_train, y_train types
import pathlib
from typing import Tuple, Optional, Dict, Any, Union

from src import config  # For default paths or training params
from src.utils import helpers  # For saving history


def train_model(
    model: tf.keras.Model,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    epochs: int = 50,
    batch_size: int = 32,
    callbacks_config: Optional[Dict[str, Any]] = None,
    model_checkpoint_path: Optional[pathlib.Path] = None,
    history_save_path: Optional[pathlib.Path] = None,
    tensorboard_log_dir: Optional[pathlib.Path] = None,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trains the Keras model with specified data and parameters.

    Args:
        model (tf.keras.Model): The compiled Keras model to train.
        X_train (Union[pd.DataFrame, np.ndarray]): Training features.
        y_train (Union[pd.Series, np.ndarray]): Training target.
        X_val (Optional[Union[pd.DataFrame, np.ndarray]]): Validation features.
        y_val (Optional[Union[pd.Series, np.ndarray]]): Validation target.
        epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.
        callbacks_config (Optional[Dict[str, Any]]): Configuration for callbacks like
            EarlyStopping. Example: {"early_stopping": {"monitor": "val_loss", "patience": 5}}.
        model_checkpoint_path (Optional[pathlib.Path]): Path to save the best model.
            If None, model checkpointing is skipped.
        history_save_path (Optional[pathlib.Path]): Path to save the training history object.
            If None, history is not saved to disk.
        tensorboard_log_dir (Optional[pathlib.Path]): Directory to save TensorBoard logs.
            If None, TensorBoard logging is skipped.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: The trained model and the training history.
    """
    print("--- Starting Model Training ---")

    # Prepare callbacks
    active_callbacks = []

    # Early Stopping
    if callbacks_config and "early_stopping" in callbacks_config:
        es_params = callbacks_config["early_stopping"]
        early_stopping = EarlyStopping(
            monitor=es_params.get("monitor", "val_loss"),
            patience=es_params.get("patience", 10),
            verbose=1,
            restore_best_weights=es_params.get("restore_best_weights", True),
        )
        active_callbacks.append(early_stopping)
        print(
            f"EarlyStopping enabled: monitor='{early_stopping.monitor}', patience={early_stopping.patience}"
        )

    # Model Checkpoint
    if model_checkpoint_path:
        model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model_checkpoint = ModelCheckpoint(
            filepath=str(model_checkpoint_path),  # ModelCheckpoint expects string path
            monitor=callbacks_config.get("model_checkpoint", {}).get(
                "monitor", "val_loss"
            ),
            save_best_only=callbacks_config.get("model_checkpoint", {}).get(
                "save_best_only", True
            ),
            save_weights_only=callbacks_config.get("model_checkpoint", {}).get(
                "save_weights_only", False
            ),
            verbose=1,
        )
        active_callbacks.append(model_checkpoint)
        print(f"ModelCheckpoint enabled: saving best model to {model_checkpoint_path}")

    # TensorBoard
    if tensorboard_log_dir:
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_callback = TensorBoard(
            log_dir=str(tensorboard_log_dir), histogram_freq=1
        )
        active_callbacks.append(tensorboard_callback)
        print(f"TensorBoard logging enabled: logs at {tensorboard_log_dir}")

    # Prepare validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        print(
            f"Using validation data: X_val shape {X_val.shape}, y_val shape {y_val.shape}"
        )
    else:
        print(
            "No validation data provided. EarlyStopping and ModelCheckpoint (if monitoring val_*) might not work as expected."
        )

    # Train the model
    print(f"\nTraining for {epochs} epochs with batch_size {batch_size}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=active_callbacks if active_callbacks else None,
        verbose=1,  # 0 = silent, 1 = progress bar, 2 = one line per epoch
    )

    print("\n--- Model Training Completed ---")

    # Save training history
    if history_save_path:
        history_save_path.parent.mkdir(parents=True, exist_ok=True)
        helpers.save_pickle(history.history, history_save_path)  # Save the history dict
        print(f"Training history saved to: {history_save_path}")

    return model, history


if __name__ == "__main__":
    # This block is for testing the trainer module directly.
    # To run this: python -m src.modeling.trainer
    print("--- Testing Model Trainer ---")

    # 1. Create a dummy model using our builder
    from src.modeling import dnn_builder

    dummy_input_shape = (5,)  # 5 features
    dummy_layers_config = [
        {"units": 8, "activation": "relu"},
        {"units": 4, "activation": "relu"},
    ]
    dummy_model = dnn_builder.build_dnn_model(
        input_shape=dummy_input_shape,
        layers_config=dummy_layers_config,
        output_layer_activation="sigmoid",
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 2. Create dummy data
    num_samples = 100
    X_train_dummy = np.random.rand(num_samples, dummy_input_shape[0])
    y_train_dummy = np.random.randint(0, 2, num_samples)
    X_val_dummy = np.random.rand(num_samples // 4, dummy_input_shape[0])
    y_val_dummy = np.random.randint(0, 2, num_samples // 4)

    # 3. Define paths for outputs from this test
    test_model_path = (
        config.MODEL_OUTPUT_DIR / "test_trainer_model.keras"
    )  # Keras native format
    test_history_path = config.PROCESSED_DATA_DIR / "test_trainer_history.pkl"
    test_tensorboard_dir = (
        config.PROJECT_ROOT / "outputs" / "logs" / "test_trainer_logs"
    )

    # 4. Define callbacks config for testing
    test_callbacks_config = {
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 3,
            "restore_best_weights": True,
        },
        "model_checkpoint": {
            "monitor": "val_accuracy",
            "save_best_only": True,
        },  # monitor val_accuracy
    }

    # 5. Train the dummy model
    trained_model, training_history = train_model(
        model=dummy_model,
        X_train=X_train_dummy,
        y_train=y_train_dummy,
        X_val=X_val_dummy,
        y_val=y_val_dummy,
        epochs=5,  # Few epochs for quick testing
        batch_size=16,
        callbacks_config=test_callbacks_config,
        model_checkpoint_path=test_model_path,
        history_save_path=test_history_path,
        tensorboard_log_dir=test_tensorboard_dir,
    )

    print("\n--- Trainer Test Output ---")
    print(f"Trained model type: {type(trained_model)}")
    print(f"Training history keys: {training_history.history.keys()}")
    if test_model_path.exists():
        print(f"Best model saved at: {test_model_path}")
    if test_history_path.exists():
        print(f"Training history saved at: {test_history_path}")
    if test_tensorboard_dir.exists() and any(test_tensorboard_dir.iterdir()):
        print(f"TensorBoard logs generated at: {test_tensorboard_dir}")
        print("You can view them by running: tensorboard --logdir outputs/logs")

    print("\nModel Trainer test completed.")
