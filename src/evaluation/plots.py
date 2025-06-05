import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from typing import List, Optional, Dict, Any


from src import config  # For default paths or feature lists if needed


def plot_training_history(
    history_data: Dict[str, List[float]],
    loss_plot_path: Optional[pathlib.Path] = None,
    accuracy_plot_path: Optional[pathlib.Path] = None,
    custom_title_prefix: str = "",
) -> None:
    """
    Plots the training and validation loss and accuracy from Keras history.

    Args:
        history_data (Dict[str, List[float]]): The history.history dictionary
            from a Keras model training.
        loss_plot_path (Optional[pathlib.Path]): Path to save the loss plot.
            If None, plot is shown.
        accuracy_plot_path (Optional[pathlib.Path]): Path to save the accuracy plot.
            If None, plot is shown.
        custom_title_prefix (str): A prefix to add to the plot titles.
    """

    if not history_data:
        print("Warning: History data is empty. Skipping plotting.")
        return

    num_epochs = len(history_data.get("loss", []))
    if num_epochs == 0:
        print("Warning: No epochs found in history data. Skipping plotting.")
        return

    epochs_range = range(1, num_epochs + 1)

    plt.style.use("seaborn-v0_8-whitegrid")

    if "loss" in history_data:
        plt.figure(figsize=(10, 6))
        plt.plot(
            epochs_range,
            history_data["loss"],
            label="Training Loss",
            color="royalblue",
            marker="o",
            linestyle="-",
        )
        if "val_loss" in history_data:
            plt.plot(
                epochs_range,
                history_data["val_loss"],
                label="Validation Loss",
                color="orangered",
                marker="x",
                linestyle="--",
            )

        plt.title(f"{custom_title_prefix}Training and Validation Loss".strip())
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.7)  # Ensure grid is on

        if loss_plot_path:
            loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(loss_plot_path)
            print(f"Saved training loss plot to: {loss_plot_path}")
            plt.close()
        else:
            plt.show()
    else:
        print("Warning: 'loss' key not found in history_data.")

    # --- Plot Training & Validation Accuracy ---
    # Determine the accuracy key (could be 'accuracy', 'acc', 'binary_accuracy', etc.)
    acc_key = None
    val_acc_key = None
    possible_acc_keys = [
        "accuracy",
        "acc",
        "binary_accuracy",
        "categorical_accuracy",
    ]  # Add more if needed

    for key in possible_acc_keys:
        if key in history_data:
            acc_key = key
            break

    if acc_key:  # If an accuracy key was found
        for key in possible_acc_keys:  # Search for corresponding validation key
            val_key_candidate = f"val_{key}"
            if val_key_candidate in history_data:
                val_acc_key = val_key_candidate
                break

        plt.figure(figsize=(10, 6))
        plt.plot(
            epochs_range,
            history_data[acc_key],
            label=f"Training {acc_key.capitalize()}",
            color="forestgreen",
            marker="o",
            linestyle="-",
        )
        if val_acc_key and val_acc_key in history_data:
            plt.plot(
                epochs_range,
                history_data[val_acc_key],
                label=f"Validation {acc_key.capitalize()}",
                color="gold",
                marker="x",
                linestyle="--",
            )

        plt.title(
            f"{custom_title_prefix}Training and Validation {acc_key.capitalize()}".strip()
        )
        plt.xlabel("Epoch")
        plt.ylabel(acc_key.capitalize())
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.7)  # Ensure grid is on

        if accuracy_plot_path:
            accuracy_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(accuracy_plot_path)
            print(f"Saved training {acc_key} plot to: {accuracy_plot_path}")
            plt.close()
        else:
            plt.show()
    else:
        print(
            "Warning: No standard 'accuracy' key found in history_data. Skipping accuracy plot."
        )


if __name__ == "__main__":
    # This block is for testing the plotting functions directly.
    # To run this: python -m src.evaluation.plots
    print("--- Testing Evaluation Plotting Functions ---")

    # --- Test plot_training_history ---
    print("\n--- Testing plot_training_history ---")
    # Create dummy history data similar to what Keras produces
    dummy_history = {
        "loss": [0.6, 0.4, 0.3, 0.25, 0.2],
        "accuracy": [0.70, 0.80, 0.85, 0.88, 0.90],
        "val_loss": [0.55, 0.42, 0.35, 0.30, 0.28],
        "val_accuracy": [0.72, 0.79, 0.83, 0.86, 0.87],
    }

    test_plots_dir = config.PLOTS_OUTPUT_DIR / "trainer_plots_test"
    test_loss_plot_path = test_plots_dir / "test_training_loss.png"
    test_acc_plot_path = test_plots_dir / "test_training_accuracy.png"

    plot_training_history(
        history_data=dummy_history,
        loss_plot_path=test_loss_plot_path,
        accuracy_plot_path=test_acc_plot_path,
        custom_title_prefix="Test Run: ",
    )

    if test_loss_plot_path.exists() and test_acc_plot_path.exists():
        print(f"Training history plots saved successfully to {test_plots_dir}")
    else:
        print(f"Failed to save training history plots to {test_plots_dir}")

    # You can add calls to test your EDA plots here as well if you moved them
    # from src.eda.exploratory_analysis to this file.
    # For now, assuming they are still in src.eda.exploratory_analysis.py
    # and this file (src.evaluation.plots) focuses on model evaluation plots.

    print("\nEvaluation plotting tests completed.")
