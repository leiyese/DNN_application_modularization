# src/modeling/dnn_builder.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from typing import List, Dict, Any, Union

from src import config  # To get default architecture if not provided


def build_dnn_model(
    input_shape: tuple,
    layers_config: List[Dict[str, Any]],
    output_layer_activation: str,
    optimizer: Union[str, tf.keras.optimizers.Optimizer],
    loss: str,
    metrics: List[str],
) -> tf.keras.Model:
    """
    Builds and compiles a sequential DNN model based on the provided configuration.

    Args:
        input_shape (tuple): The shape of the input data (e.g., (num_features,)).
        layers_config (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary defines a hidden layer with 'units', 'activation', and 'dropout'.
        output_layer_activation (str): The activation function for the output layer
            (e.g., 'sigmoid' for binary classification, 'softmax' for multi-class).
        optimizer (Union[str, tf.keras.optimizers.Optimizer]): The optimizer to use.
        loss (str): The loss function to use.
        metrics (List[str]): A list of metrics to monitor during training.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    print("--- Building DNN Model ---")
    model = Sequential()

    # Add the input layer explicitly
    model.add(Input(shape=input_shape))
    print(f"Input Layer: shape={input_shape}")

    # Add hidden layers from the configuration
    for i, layer in enumerate(layers_config):
        units = layer.get("units")
        activation = layer.get("activation")
        dropout_rate = layer.get("dropout")

        if not units or not activation:
            raise ValueError(f"Layer {i} config must have 'units' and 'activation'.")

        model.add(Dense(units=units, activation=activation, name=f"hidden_layer_{i+1}"))
        print(f"Added Dense Layer: units={units}, activation='{activation}'")

        if dropout_rate and 0 < dropout_rate < 1:
            model.add(Dropout(rate=dropout_rate, name=f"dropout_{i+1}"))
            print(f"Added Dropout Layer: rate={dropout_rate}")

    # Add the output layer
    # For binary classification, this will be a single unit.
    model.add(Dense(1, activation=output_layer_activation, name="output_layer"))
    print(f"Output Layer: units=1, activation='{output_layer_activation}'")

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("\n--- Model Compilation ---")
    print(f"Optimizer: {optimizer}")
    print(f"Loss Function: {loss}")
    print(f"Metrics: {metrics}")

    print("\n--- Model Summary ---")
    model.summary()

    return model


if __name__ == "__main__":
    # This block is for testing the builder module directly.
    # To run this: python -m src.modeling.dnn_builder
    print("--- Testing DNN Builder ---")

    # Define a dummy input shape for testing
    dummy_input_shape = (10,)  # e.g., 10 features

    # Use the architecture defined in our config file
    test_model = build_dnn_model(
        input_shape=dummy_input_shape,
        layers_config=config.DNN_ARCHITECTURE["layers"],
        output_layer_activation=config.DNN_ARCHITECTURE["output_layer_activation"],
        optimizer=config.DNN_OPTIMIZER,
        loss=config.DNN_LOSS,
        metrics=config.DNN_METRICS,
    )

    print("\nDNN Builder test completed successfully. Model was built and compiled.")
    print(f"Is model a Keras Model? {isinstance(test_model, tf.keras.Model)}")
