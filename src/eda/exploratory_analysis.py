# src/eda/exploratory_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from typing import List, Optional

from src import config  # For default paths or feature lists if needed


def generate_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates descriptive statistics for the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Descriptive statistics.
    """
    print("--- Descriptive Statistics ---")
    desc_stats = df.describe(include="all")
    print(desc_stats)
    return desc_stats


def get_null_counts(df: pd.DataFrame) -> pd.Series:
    """
    Gets the count of null values for each column.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: Series with null counts per column.
    """
    print("\n--- Null Value Counts ---")
    null_counts = df.isnull().sum()
    print(
        null_counts[null_counts > 0]
        if (null_counts > 0).any()
        else "No null values found."
    )
    return null_counts


def plot_histograms_for_numerical_features(
    df: pd.DataFrame,
    numerical_features: List[str],
    save_dir: Optional[pathlib.Path] = None,
) -> None:
    """
    Plots histograms for specified numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_features (List[str]): List of numerical column names to plot.
        save_dir (Optional[pathlib.Path]): Directory to save the plots. If None, plots are shown.
    """
    print("\n--- Plotting Histograms for Numerical Features ---")
    if save_dir:
        config.PLOTS_OUTPUT_DIR.mkdir(
            parents=True, exist_ok=True
        )  # Ensure base plots dir exists
        save_dir.mkdir(parents=True, exist_ok=True)  # Ensure specific save_dir exists

    for col in numerical_features:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            if save_dir:
                plot_path = save_dir / f"histogram_{col}.png"
                plt.savefig(plot_path)
                print(f"Saved histogram for {col} to {plot_path}")
                plt.close()  # Close the plot to free memory when saving
            else:
                plt.show()
        else:
            print(
                f"Warning: Column '{col}' not found in DataFrame for histogram plotting."
            )


def plot_correlation_matrix(
    df: pd.DataFrame,
    numerical_features: List[
        str
    ],  # Typically on numerical features, or appropriately encoded categoricals
    save_path: Optional[pathlib.Path] = None,
) -> None:
    """
    Plots the correlation matrix for specified numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_features (List[str]): List of numerical column names for correlation.
        save_path (Optional[pathlib.Path]): Path to save the plot. If None, plot is shown.
    """
    print("\n--- Plotting Correlation Matrix ---")
    # Select only the numerical features for the correlation matrix
    df_numerical = df[numerical_features]
    corr_matrix = df_numerical.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix of Numerical Features")

    if save_path:
        config.PLOTS_OUTPUT_DIR.mkdir(
            parents=True, exist_ok=True
        )  # Ensure base plots dir exists
        save_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure specific save_dir exists
        plt.savefig(save_path)
        print(f"Saved correlation matrix to {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # This block is for testing the EDA module directly.
    # To run this:
    # 1. Make sure your conda environment is activated.
    # 2. Navigate to the 'dynamic_dnn_trainer' project root directory in your terminal.
    # 3. Run: python -m src.eda.exploratory_analysis

    print("--- Testing EDA functions ---")

    # Load raw data using our loader
    from src.data_ingestion import loader

    raw_df = None
    if config.RAW_DATA_FILE.exists():
        raw_df = loader.load_csv_data(file_path=config.RAW_DATA_FILE)
    else:
        print(
            f"ERROR: Raw data file not found at {config.RAW_DATA_FILE}. Cannot run test."
        )
        exit()

    # Test descriptive stats
    desc_stats_df = generate_descriptive_stats(raw_df)

    # Test null counts
    nulls = get_null_counts(raw_df)

    # Test plotting histograms
    # Define a subdirectory within PLOTS_OUTPUT_DIR for EDA plots from this test run
    eda_test_plots_dir = config.PLOTS_OUTPUT_DIR / "eda_module_test"
    plot_histograms_for_numerical_features(
        df=raw_df,
        numerical_features=config.NUMERICAL_FEATURES,
        save_dir=eda_test_plots_dir,
    )

    # Test plotting correlation matrix
    correlation_plot_path = eda_test_plots_dir / "correlation_matrix.png"
    plot_correlation_matrix(
        df=raw_df,
        numerical_features=config.NUMERICAL_FEATURES,
        save_path=correlation_plot_path,
    )

    print(
        f"\nEDA tests completed. Check the console output and the directory: {eda_test_plots_dir}"
    )
