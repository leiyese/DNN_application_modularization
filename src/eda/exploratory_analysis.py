# src/eda/exploratory_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from typing import List, Optional, Union

# Import config for default paths or feature lists if needed by specific EDA functions
from src import config
from src.utils import helpers  # For ensure_directory_exists


def display_dataframe_info(df: pd.DataFrame, df_name: str = "DataFrame") -> None:
    """
    Displays basic information about the DataFrame: head, info, shape.
    """
    print(f"\n--- Basic Info for {df_name} ---")
    print("First 5 rows:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()
    print(f"\nShape of {df_name}: {df.shape}")


def generate_descriptive_statistics(
    df: pd.DataFrame, df_name: str = "DataFrame"
) -> pd.DataFrame:
    """
    Generates and prints descriptive statistics for the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        df_name (str): A name for the DataFrame for print statements.

    Returns:
        pd.DataFrame: A DataFrame containing descriptive statistics.
    """
    print(f"\n--- Descriptive Statistics for {df_name} ---")
    # include='all' provides stats for object/categorical columns too
    desc_stats = df.describe(include="all").transpose()
    print(desc_stats)
    return desc_stats


def count_missing_values(df: pd.DataFrame, df_name: str = "DataFrame") -> pd.Series:
    """
    Counts and prints the number of missing (NaN) values for each column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        df_name (str): A name for the DataFrame for print statements.

    Returns:
        pd.Series: A Series with column names as index and count of NaNs as values.
    """
    print(f"\n--- Missing Value Counts for {df_name} ---")
    missing_counts = df.isnull().sum()
    missing_counts_filtered = missing_counts[
        missing_counts > 0
    ]  # Show only columns with missing values

    if not missing_counts_filtered.empty:
        print(missing_counts_filtered)
    else:
        print("No missing values found in any column.")
    return missing_counts


def plot_histograms(
    df: pd.DataFrame,
    columns_to_plot: List[str],
    plot_save_directory: Optional[pathlib.Path] = None,
    bins: int = 30,
    kde: bool = True,
) -> None:
    """
    Plots histograms for the specified numerical columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_plot (List[str]): A list of column names for which to plot histograms.
        plot_save_directory (Optional[pathlib.Path]): Directory to save the plots.
                                                       If None, plots are shown interactively.
        bins (int): Number of bins for the histogram.
        kde (bool): Whether to plot a Kernel Density Estimate.
    """
    print(f"\n--- Plotting Histograms for: {', '.join(columns_to_plot)} ---")
    sns.set_style("whitegrid")

    if plot_save_directory:
        helpers.ensure_directory_exists(plot_save_directory)

    for col_name in columns_to_plot:
        if col_name in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col_name], bins=bins, kde=kde, color="darkblue")
            plt.title(f"Distribution of {col_name}", fontsize=15)
            plt.xlabel(col_name, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)

            if plot_save_directory:
                save_path = plot_save_directory / f"histogram_{col_name}.png"
                plt.savefig(save_path)
                print(f"  Histogram for '{col_name}' saved to: {save_path}")
                plt.close()  # Close plot to free memory when saving in loop
            else:
                plt.show()
        else:
            print(
                f"  Warning: Column '{col_name}' not found in DataFrame. Skipping histogram."
            )


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns_for_corr: List[str],  # Typically numerical features
    plot_save_path: Optional[pathlib.Path] = None,
    method: str = "pearson",
) -> None:
    """
    Plots a heatmap of the correlation matrix for specified columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_for_corr (List[str]): List of column names to include in the correlation matrix.
        plot_save_path (Optional[pathlib.Path]): Path to save the heatmap image.
                                                  If None, plot is shown interactively.
        method (str): Method of correlation ('pearson', 'kendall', 'spearman').
    """
    print(
        f"\n--- Plotting Correlation Heatmap (Method: {method}) for: {', '.join(columns_for_corr)} ---"
    )
    sns.set_style("whitegrid")

    # Select only the specified columns for correlation
    df_subset = df[columns_for_corr]
    correlation_matrix = df_subset.corr(method=method)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar=True,
    )
    plt.title(f"Correlation Matrix ({method.capitalize()})", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if plot_save_path:
        helpers.ensure_directory_exists(plot_save_path.parent)
        plt.savefig(plot_save_path)
        print(f"  Correlation heatmap saved to: {plot_save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # This block allows you to test this module directly.
    print("--- Testing EDA Module ---")

    # 1. Load raw data using our loader
    from src.data_ingestion import (
        loader,
    )  # Assuming loader.py is in the same src structure

    # config is already imported at the top

    raw_df_test = loader.load_data_from_csv(csv_file_path=config.RAW_DATA_FILE_PATH)

    if raw_df_test is not None:
        # 2. Test basic info display
        display_dataframe_info(raw_df_test, "Raw Test DataFrame")

        # 3. Test descriptive statistics
        stats_df = generate_descriptive_statistics(raw_df_test, "Raw Test DataFrame")

        # 4. Test missing value counts
        missing_vals = count_missing_values(raw_df_test, "Raw Test DataFrame")

        # 5. Define where to save test plots
        eda_test_plots_output_dir = config.PLOTS_OUTPUT_DIR / "eda_module_tests"

        # 6. Test plotting histograms for numerical features
        # Ensure numerical features from config actually exist in the DataFrame
        existing_numerical_features = [
            col for col in config.NUMERICAL_FEATURES if col in raw_df_test.columns
        ]
        if existing_numerical_features:
            plot_histograms(
                df=raw_df_test,
                columns_to_plot=existing_numerical_features,
                plot_save_directory=eda_test_plots_output_dir,
            )
        else:
            print(
                "No numerical features (from config) found in the DataFrame to plot histograms."
            )

        # 7. Test plotting correlation heatmap
        # Ensure numerical features for correlation exist
        if existing_numerical_features:
            correlation_heatmap_save_path = (
                eda_test_plots_output_dir / "correlation_heatmap_test.png"
            )
            plot_correlation_heatmap(
                df=raw_df_test,
                columns_for_corr=existing_numerical_features,
                plot_save_path=correlation_heatmap_save_path,
            )
        else:
            print(
                "No numerical features (from config) found in the DataFrame for correlation heatmap."
            )

        print(
            f"\nEDA module tests completed. Check console output and plots in: {eda_test_plots_output_dir}"
        )
    else:
        print("EDA module tests aborted: Raw data could not be loaded.")
