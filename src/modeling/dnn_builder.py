# src/modeling/dnn_builder.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from typing import Tuple, List, Dict, Any, Union  # For type hinting

# Import our configuration to access model parameters
from src import config


def build_dynamic_dnn_model(
    input_features_shape: Tuple[int, ...],
    model_arch_params: Dict[str, Any],
    compilation_params: Dict[str, Any],
) -> tf.keras.Model:
    """
    Builds and compiles a sequential Deep Neural Network (DNN) model
    based on the provided architecture and compilation parameters.

    Args:
        input_features_shape (Tuple[int, ...]): The shape of the input features
            (e.g., (number_of_features,) for tabular data).
        model_arch_params (Dict[str, Any]): Dictionary containing architecture parameters:
            - "layers": List of hidden layer definitions. Each dict in the list
                        should have "units", "activation", and optionally "dropout_rate".
            - "output_layer_units": Number of units for the output layer.
            - "output_layer_activation": Activation function for the output layer.
        compilation_params (Dict[str, Any]): Dictionary containing compilation parameters:
            - "optimizer": Name of the optimizer or an optimizer instance.
            - "loss": Name of the loss function.
            - "metrics": List of metrics to evaluate.

    Returns:
        tf.keras.Model: The compiled Keras DNN model.
    """
    print("--- Building Dynamic DNN Model ---")
    model = Sequential(name="DynamicDNN")

    # Add the Input layer explicitly, using the provided shape
    model.add(Input(shape=input_features_shape, name="input_layer"))
    print(f"Input Layer: shape={input_features_shape}")

    # Add hidden layers based on the configuration
    hidden_layers_config = model_arch_params.get("layers", [])
    for i, layer_conf in enumerate(hidden_layers_config):
        units = layer_conf.get("units")
        activation = layer_conf.get("activation")
        dropout_rate = layer_conf.get("dropout_rate")

        if not units or not activation:
            raise ValueError(
                f"Hidden layer {i+1} configuration is missing 'units' or 'activation'."
            )

        model.add(Dense(units=units, activation=activation, name=f"hidden_layer_{i+1}"))
        print(f"  Added Dense Layer {i+1}: units={units}, activation='{activation}'")

        if dropout_rate and 0 < dropout_rate < 1:
            model.add(Dropout(rate=dropout_rate, name=f"dropout_{i+1}"))
            print(f"    Added Dropout Layer {i+1}: rate={dropout_rate}")

    # Add the output layer
    output_units = model_arch_params.get("output_layer_units")
    output_activation = model_arch_params.get("output_layer_activation")

    if output_units is None or output_activation is None:
        raise ValueError(
            "Model architecture parameters must include 'output_layer_units' and 'output_layer_activation'."
        )

    model.add(
        Dense(units=output_units, activation=output_activation, name="output_layer")
    )
    print(f"Output Layer: units={output_units}, activation='{output_activation}'")

    # Compile the model
    optimizer = compilation_params.get("optimizer", "adam")
    loss = compilation_params.get("loss")
    metrics = compilation_params.get("metrics", ["accuracy"])

    if not loss:
        raise ValueError("Compilation parameters must include a 'loss' function.")

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("\n--- Model Compilation ---")
    print(f"  Optimizer: {optimizer}")
    print(f"  Loss Function: {loss}")
    print(f"  Metrics: {metrics}")

    print("\n--- Model Summary ---")
    model.summary(line_length=100)  # Print a summary of the model

    return model


if __name__ == "__main__":
    # This block allows you to test this module directly.
    print("--- Testing DNN Builder Module (Multi-Class Focus) ---")

    input_shape_test = (20,)

    # Use the DNN params from config

    try:
        test_dnn_model = build_dynamic_dnn_model(
            input_features_shape=input_shape_test,
            model_arch_params=config.DNN_MODEL_PARAMS["architecture"],
            compilation_params=config.DNN_MODEL_PARAMS["compilation"],
        )
        print("\nDNN Buildinger was successfully compiled.")
        print(
            f"Is the model a keras model? {isinstance(test_dnn_model, tf.keras.Model)}"
        )
    except Exception as e:
        print("There was an error compiling the DNN builder test: {e}")
        import traceback

        traceback.print_exc()
