# src/preprocessing/transformers.py

import pandas as pd
import numpy as np  # For data type checks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pathlib
from typing import List, Tuple, Any, Optional

# Import configurations and helper functions
from src import config
from src.utils import helpers


def split_dataframe_into_train_test(
    dataframe: pd.DataFrame,
    target_column_name: str,
    test_set_ratio: float,
    stratify_by_target: bool = True,
    random_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits a DataFrame into training and testing sets for features (X) and target (y).

    Args:
        dataframe (pd.DataFrame): The input DataFrame to split.
        target_column_name (str): The name of the column to be used as the target variable.
        test_set_ratio (float): The proportion of the dataset to allocate to the test set.
        stratify_by_target (bool): If True, performs stratified sampling based on the target
                                   variable, which is useful for classification tasks.
        random_seed (Optional[int]): Seed for the random number generator for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    print(
        f"Splitting data. Target column: '{target_column_name}', Test set ratio: {test_set_ratio}"
    )

    X = dataframe.drop(columns=[target_column_name])
    y = dataframe[target_column_name]

    stratify_option = y if stratify_by_target else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_set_ratio,
        random_state=random_seed,
        stratify=stratify_option,
    )

    print("Data splitting complete:")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape},  y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def create_feature_preprocessor(
    numerical_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for preprocessing features.
    - Numerical features: Median imputation then StandardScaler.
    - Categorical features: Most frequent imputation then OneHotEncoder.

    Args:
        numerical_cols (List[str]): List of numerical column names.
        categorical_cols (List[str]): List of categorical column names.

    Returns:
        ColumnTransformer: The configured preprocessor object.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer_num", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer_cat", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical_transforms", numerical_pipeline, numerical_cols),
            ("categorical_transforms", categorical_pipeline, categorical_cols),
        ],
        remainder="passthrough",  # Keeps columns not specified, if any
    )
    return preprocessor


def apply_feature_preprocessing(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    preprocessor_object: ColumnTransformer,
    fit_preprocessor: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the preprocessor to training and testing feature sets.
    Fits the preprocessor on training data if fit_preprocessor is True.

    Args:
        X_train_df (pd.DataFrame): Training features.
        X_test_df (pd.DataFrame): Testing features.
        preprocessor_object (ColumnTransformer): The preprocessor to apply.
        fit_preprocessor (bool): If True, fit the preprocessor on X_train_df.
                                 If False, assume it's already fitted.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed X_train and X_test as DataFrames.
    """
    if fit_preprocessor:
        print("Fitting preprocessor on training data and transforming X_train...")
        X_train_processed_np = preprocessor_object.fit_transform(X_train_df)
    else:
        print("Preprocessor already fitted. Transforming X_train...")
        X_train_processed_np = preprocessor_object.transform(X_train_df)

    print("Transforming X_test data...")
    X_test_processed_np = preprocessor_object.transform(X_test_df)

    # Attempt to get feature names after transformation for DataFrame creation
    try:
        # Get feature names from the 'numerical_transforms' part
        num_feature_names = preprocessor_object.transformers_[0][
            2
        ]  # Original numerical col names

        # Get feature names from the 'categorical_transforms' (OneHotEncoder) part
        ohe_step = preprocessor_object.named_transformers_[
            "categorical_transforms"
        ].named_steps["onehot"]
        cat_feature_names_ohe = list(
            ohe_step.get_feature_names_out(config.CATEGORICAL_FEATURES)
        )

        # Handle 'remainder' columns if any
        # This part can be complex if 'remainder' is not 'drop' or 'passthrough'
        # or if column order changes significantly.
        # For 'passthrough', original names of remainder columns are used.
        remainder_cols = []
        if preprocessor_object.remainder == "passthrough":
            # Identify columns that were neither numerical nor categorical
            processed_cols_flat = set(num_feature_names + config.CATEGORICAL_FEATURES)
            original_cols = set(X_train_df.columns)
            remainder_cols = sorted(list(original_cols - processed_cols_flat))

        all_processed_feature_names = (
            num_feature_names + cat_feature_names_ohe + remainder_cols
        )

        X_train_processed_df = pd.DataFrame(
            X_train_processed_np,
            columns=all_processed_feature_names,
            index=X_train_df.index,
        )
        X_test_processed_df = pd.DataFrame(
            X_test_processed_np,
            columns=all_processed_feature_names,
            index=X_test_df.index,
        )
        print(
            "Successfully created DataFrames for processed features with column names."
        )

    except Exception as e:
        print(
            f"Warning: Could not reconstruct feature names for processed DataFrames: {e}"
        )
        print("Returning processed data as NumPy arrays.")
        # If creating DataFrame with names fails, return NumPy arrays
        # The calling code will need to handle this (e.g. model input shape)
        return X_train_processed_np, X_test_processed_np  # type: ignore

    return X_train_processed_df, X_test_processed_df


def prepare_target_variable(y_series: pd.Series) -> pd.Series:
    """
    Prepares the target variable. For multi-class with sparse_categorical_crossentropy,
    we need integer labels (0, 1, 2, ...). This function ensures the dtype is integer.
    If extensive encoding (like string to int) was needed, a LabelEncoder would be used here.

    Args:
        y_series (pd.Series): The target variable series.

    Returns:
        pd.Series: The prepared target variable series (ensured as integer type).
    """
    print(f"Preparing target variable. Original dtype: {y_series.dtype}")
    # Ensure target is integer type, as expected by sparse_categorical_crossentropy
    if not pd.api.types.is_integer_dtype(y_series):
        try:
            y_prepared = y_series.astype(int)
            print(f"Target variable converted to int. New dtype: {y_prepared.dtype}")
        except ValueError as e:
            print(f"Error: Could not convert target variable to int: {e}")
            print(
                "Please ensure the target column contains values convertible to integers."
            )
            raise
    else:
        y_prepared = y_series.copy()  # Make a copy if already int

    # For multi-class, check if values are sequential from 0 (optional, but good sanity check)
    # unique_targets = sorted(y_prepared.unique())
    # if unique_targets != list(range(len(unique_targets))):
    #     print(f"Warning: Target values are {unique_targets}, not strictly 0 to N-1 sequential.")
    #     print("LabelEncoder might be needed if classes are not 0-indexed or have gaps.")

    return y_prepared


if __name__ == "__main__":
    # This block allows you to test this module directly.
    print("--- Testing Preprocessing Transformers Module (Multi-Class Focus) ---")

    # 1. Load raw data
    from src.data_ingestion import (
        loader,
    )  # Assuming loader.py is in the same src structure

    raw_dataframe = loader.load_data_from_csv(csv_file_path=config.RAW_DATA_FILE_PATH)

    if raw_dataframe is None:
        print("Test aborted: Raw data could not be loaded.")
        exit()

    # 2. Split data
    print("\n--- Step 1: Splitting data ---")
    X_train, X_test, y_train_raw, y_test_raw = split_dataframe_into_train_test(
        dataframe=raw_dataframe,
        target_column_name=config.TARGET_COLUMN,
        test_set_ratio=config.TEST_SET_SIZE,
        random_seed=config.RANDOM_SEED,
        stratify_by_target=True,
    )

    # 3. Create and fit/load feature preprocessor
    print("\n--- Step 2: Feature Preprocessing ---")
    # Create the preprocessor
    feature_preprocessor = create_feature_preprocessor(
        numerical_cols=config.NUMERICAL_FEATURES,
        categorical_cols=config.CATEGORICAL_FEATURES,
    )

    # Apply preprocessing (fit on train, transform train and test)
    X_train_processed, X_test_processed = apply_feature_preprocessing(
        X_train_df=X_train.copy(),  # Use .copy() to avoid SettingWithCopyWarning on potential imputation
        X_test_df=X_test.copy(),
        preprocessor_object=feature_preprocessor,
        fit_preprocessor=True,  # Fit it now
    )

    # Save the fitted preprocessor
    if helpers.save_object_as_pickle(
        feature_preprocessor, config.PREPROCESSOR_SAVE_PATH
    ):
        print(f"Fitted feature preprocessor saved to: {config.PREPROCESSOR_SAVE_PATH}")

    print("\nProcessed training features (X_train_processed) head:")
    # X_train_processed might be DataFrame or NumPy array
    if isinstance(X_train_processed, pd.DataFrame):
        print(X_train_processed.head())
        print("X_train_processed is a DataFrame")
    else:
        print(X_train_processed[:5])  # Print first 5 rows if NumPy
        print("X_train_processed is a Numpy array")

    # 4. Prepare target variable
    print("\n--- Step 3: Preparing Target Variable ---")
    y_train_prepared = prepare_target_variable(y_train_raw.copy())
    y_test_prepared = prepare_target_variable(y_test_raw.copy())

    print("\nPrepared training target (y_train_prepared) head:")
    print(y_train_prepared.head())
    print(
        f"Unique values in prepared training target: {sorted(y_train_prepared.unique())}"
    )

    print("\n--- Preprocessing Module Test Completed ---")
