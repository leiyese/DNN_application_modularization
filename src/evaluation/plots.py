# src/evaluation/plots.py

import matplotlib.pyplot as plt
import seaborn as sns  # For a nicer default plot style
import pathlib
from typing import Dict, Optional, List

# Import config for default save locations if needed, though functions should take paths
from src import config
from src.utils import helpers  # For ensure_directory_exists


def plot_training_history(
    history_data: Dict[str, List[float]],
    plot_title: str = "Model Training History",
    loss_metric_name: str = "loss",
    accuracy_metric_name: str = "accuracy",  # Or whatever accuracy metric key is in history
    save_plot_path: Optional[pathlib.Path] = None,
) -> None:
    """
    Plots the training and validation loss and accuracy from Keras training history.

    Args:
        history_data (Dict[str, List[float]]): The dictionary obtained from
                                               Keras `history.history`.
        plot_title (str): The main title for the plots.
        loss_metric_name (str): The key for training loss in history_data (e.g., 'loss').
        accuracy_metric_name (str): The key for training accuracy in history_data (e.g., 'accuracy').
        save_plot_path (Optional[pathlib.Path]): If provided, the plot will be saved to this
                                                 path (e.g., "plots/training_curves.png").
                                                 Otherwise, the plot will be shown.
    """
    sns.set_style("whitegrid")  # Apply a nice seaborn style

    # Determine if validation data was used by checking for 'val_loss' or 'val_accuracy'
    has_validation_loss = f"val_{loss_metric_name}" in history_data
    has_validation_accuracy = f"val_{accuracy_metric_name}" in history_data

    num_epochs = len(history_data[loss_metric_name])
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(14, 6))

    # --- Plot Loss ---
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(
        epochs_range,
        history_data[loss_metric_name],
        label=f"Training {loss_metric_name.capitalize()}",
        marker="o",
        linestyle="-",
    )
    if has_validation_loss:
        plt.plot(
            epochs_range,
            history_data[f"val_{loss_metric_name}"],
            label=f"Validation {loss_metric_name.capitalize()}",
            marker="x",
            linestyle="--",
        )
    plt.title(f"Training and Validation {loss_metric_name.capitalize()}")
    plt.xlabel("Epoch")
    plt.ylabel(loss_metric_name.capitalize())
    plt.legend(loc="upper right")
    plt.grid(True)

    # --- Plot Accuracy ---
    # Check if the accuracy metric is actually in the history data
    if accuracy_metric_name in history_data:
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        plt.plot(
            epochs_range,
            history_data[accuracy_metric_name],
            label=f"Training {accuracy_metric_name.capitalize()}",
            marker="o",
            linestyle="-",
        )
        if has_validation_accuracy:
            plt.plot(
                epochs_range,
                history_data[f"val_{accuracy_metric_name}"],
                label=f"Validation {accuracy_metric_name.capitalize()}",
                marker="x",
                linestyle="--",
            )
        plt.title(f"Training and Validation {accuracy_metric_name.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel(accuracy_metric_name.capitalize())
        plt.legend(loc="lower right")
        plt.grid(True)
    else:
        print(
            f"Warning: Accuracy metric '{accuracy_metric_name}' not found in history data. Skipping accuracy plot."
        )

    plt.suptitle(plot_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle

    if save_plot_path:
        # Ensure the directory for saving the plot exists
        helpers.ensure_directory_exists(save_plot_path.parent)
        plt.savefig(save_plot_path)
        print(f"Training history plot saved to: {save_plot_path}")
        plt.close()  # Close the figure to free memory if saving
    else:
        plt.show()  # Display the plot


if __name__ == "__main__":
    # This block allows you to test this module directly.
    print("--- Testing Plotting Module ---")

    # 1. Create dummy history data (simulating Keras history.history dict)
    dummy_history = {
        "loss": [1.5, 1.0, 0.8, 0.6, 0.5],
        "accuracy": [0.50, 0.60, 0.70, 0.75, 0.80],
        "val_loss": [1.6, 1.2, 0.9, 0.75, 0.65],
        "val_accuracy": [0.45, 0.55, 0.65, 0.72, 0.78],
    }
    num_epochs_test = len(dummy_history["loss"])

    print(f"Dummy history data created for {num_epochs_test} epochs.")

    # 2. Define a path to save the test plot
    # Using PLOTS_OUTPUT_DIR from config for consistency
    test_plot_save_dir = config.PLOTS_OUTPUT_DIR / "test_plots"
    test_plot_filename = "test_training_history.png"
    full_test_plot_path = test_plot_save_dir / test_plot_filename

    # 3. Plot the dummy history and save it
    print(f"\nAttempting to plot and save to: {full_test_plot_path}")
    plot_training_history(
        history_data=dummy_history,
        plot_title="Dummy Model Training Progress",
        save_plot_path=full_test_plot_path,
    )

    # 4. Test plotting without validation data
    dummy_history_no_val = {
        "loss": [1.5, 1.0, 0.8, 0.6, 0.5],
        "accuracy": [0.50, 0.60, 0.70, 0.75, 0.80],
    }
    test_plot_no_val_filename = "test_training_history_no_val.png"
    full_test_plot_no_val_path = test_plot_save_dir / test_plot_no_val_filename
    print(
        f"\nAttempting to plot (no validation data) and save to: {full_test_plot_no_val_path}"
    )
    plot_training_history(
        history_data=dummy_history_no_val,
        plot_title="Dummy Model Training (No Validation)",
        save_plot_path=full_test_plot_no_val_path,
    )

    if full_test_plot_path.exists():
        print(
            f"\nTest plot with validation data should be saved at: {full_test_plot_path}"
        )
    else:
        print(
            f"\nERROR: Test plot with validation data was NOT saved to {full_test_plot_path}"
        )

    if full_test_plot_no_val_path.exists():
        print(
            f"Test plot without validation data should be saved at: {full_test_plot_no_val_path}"
        )
    else:
        print(
            f"ERROR: Test plot without validation data was NOT saved to {full_test_plot_no_val_path}"
        )

    print(
        "\nPlotting module test completed. Check the 'outputs/plots/test_plots/' directory."
    )
