# src/preprocessing/transformers.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pathlib
from typing import List, Tuple, Any, Optional

from src import (
    config,
)  # For default feature lists if needed by create_feature_preprocessor

# from src.utils import helpers # helpers will be used in main.py for saving preprocessor


# This function returns DataFrames/Series as it's just splitting
def split_dataframe_into_train_test(
    dataframe: pd.DataFrame,
    target_column_name: str,
    test_set_ratio: float,
    stratify_by_target: bool = True,
    random_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits a DataFrame into training and testing sets for features (X) and target (y).
    Returns raw splits as Pandas DataFrames and Series.
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
    print(f"  X_train_raw shape: {X_train.shape}, y_train_raw shape: {y_train.shape}")
    print(f"  X_test_raw shape: {X_test.shape},  y_test_raw shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test


# This function defines the preprocessor
def create_feature_preprocessor(
    numerical_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for preprocessing features.
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
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),  # dense output
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical_transforms", numerical_pipeline, numerical_cols),
            ("categorical_transforms", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    print("Feature preprocessor object created.")
    return preprocessor


# This function applies the preprocessor and returns NumPy arrays
def apply_feature_preprocessing_to_numpy(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    preprocessor_object: ColumnTransformer,  # This object will be fitted or is already fitted
    fit_preprocessor_on_train: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the preprocessor to training and testing feature sets.
    Fits the preprocessor on X_train_df if fit_preprocessor_on_train is True.
    Returns processed X_train and X_test as NumPy arrays (float32).
    """
    X_train_transformed_np = None
    if fit_preprocessor_on_train:
        print("Fitting preprocessor on training data and transforming X_train...")
        X_train_transformed_np = preprocessor_object.fit_transform(X_train_df)
    else:
        print("Preprocessor already fitted. Transforming X_train...")
        X_train_transformed_np = preprocessor_object.transform(X_train_df)

    print("Transforming X_test data...")
    X_test_transformed_np = preprocessor_object.transform(X_test_df)

    # Ensure float32 for TensorFlow
    X_train_final_np = X_train_transformed_np.astype(np.float32)
    X_test_final_np = X_test_transformed_np.astype(np.float32)

    print(
        f"  Processed X_train_np shape: {X_train_final_np.shape}, dtype: {X_train_final_np.dtype}"
    )
    print(
        f"  Processed X_test_np shape: {X_test_final_np.shape}, dtype: {X_test_final_np.dtype}"
    )
    return X_train_final_np, X_test_final_np


# This function prepares the target and returns a NumPy array
def prepare_target_to_numpy(y_series: pd.Series) -> np.ndarray:
    """
    Prepares the target variable (ensures integer type) and returns it as a NumPy array (int64).
    """
    print(f"Preparing target variable. Original dtype: {y_series.dtype}")
    y_prepared_series = None
    if not pd.api.types.is_integer_dtype(y_series):
        try:
            y_prepared_series = y_series.astype(int)
            print(
                f"Target variable intermediate conversion to int. New dtype: {y_prepared_series.dtype}"
            )
        except ValueError as e:
            print(
                f"Error: Could not convert target variable to int: {e}. Ensure target is numeric."
            )
            raise
    else:
        y_prepared_series = (
            y_series.copy()
        )  # Use copy to avoid modifying original if it was passed around

    y_final_np = y_prepared_series.to_numpy(dtype=np.int64)
    print(
        f"  Prepared target y_np shape: {y_final_np.shape}, dtype: {y_final_np.dtype}, unique values: {np.unique(y_final_np)}"
    )
    return y_final_np


if __name__ == "__main__":
    print(
        "--- Testing Preprocessing Transformers Module (Granular Functions, NumPy Output) ---"
    )
    from src.data_ingestion import loader
    from src.utils import helpers  # For saving in test

    # config is already imported at the top of the file

    raw_dataframe = loader.load_data_from_csv(csv_file_path=config.RAW_DATA_FILE_PATH)
    if raw_dataframe is None:
        print("Test aborted: Raw data could not be loaded.")
        exit()  # Use exit() in __main__ block for scripts

    # 1. Split data
    print("\n--- Testing data splitting ---")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_dataframe_into_train_test(
        dataframe=raw_dataframe,
        target_column_name=config.TARGET_COLUMN,  # Corrected: Use the full arg name
        test_set_ratio=config.TEST_SET_SIZE,  # Corrected: Use the full arg name
        random_seed=config.RANDOM_SEED,
        stratify_by_target=True,  # Explicitly set, though it's the default
    )

    # 2. Create feature preprocessor
    print("\n--- Testing preprocessor creation ---")
    feature_preprocessor = create_feature_preprocessor(
        numerical_cols=config.NUMERICAL_FEATURES,
        categorical_cols=config.CATEGORICAL_FEATURES,
    )

    # 3. Apply feature preprocessing (fit and transform, output NumPy)
    print("\n--- Testing feature preprocessing to NumPy ---")
    X_train_np, X_test_np = apply_feature_preprocessing_to_numpy(
        X_train_df=X_train_raw.copy(),  # Use .copy() to avoid potential SettingWithCopyWarning
        X_test_df=X_test_raw.copy(),
        preprocessor_object=feature_preprocessor,
        fit_preprocessor_on_train=True,
    )
    # Save the fitted preprocessor (as main.py would do)
    if helpers.save_object_as_pickle(
        feature_preprocessor, config.PREPROCESSOR_SAVE_PATH
    ):
        print(
            f"Fitted feature preprocessor saved to (for test): {config.PREPROCESSOR_SAVE_PATH}"
        )

    # 4. Prepare target variables to NumPy
    print("\n--- Testing target preparation to NumPy ---")
    y_train_np = prepare_target_to_numpy(y_train_raw.copy())
    y_test_np = prepare_target_to_numpy(y_test_raw.copy())

    print("\n--- Test Output Shapes and Types from Granular Functions ---")
    print(f"X_train_np: shape={X_train_np.shape}, dtype={X_train_np.dtype}")
    print(
        f"y_train_np: shape={y_train_np.shape}, dtype={y_train_np.dtype}"
    )  # No unique print here, was in prepare_target
    print(f"X_test_np: shape={X_test_np.shape}, dtype={X_test_np.dtype}")
    print(f"y_test_np: shape={y_test_np.shape}, dtype={y_test_np.dtype}")
    print(
        f"Fitted Preprocessor Object: {type(feature_preprocessor)}"
    )  # Check type of the saved object
    if config.PREPROCESSOR_SAVE_PATH.exists():
        print(f"Preprocessor pickle file exists at: {config.PREPROCESSOR_SAVE_PATH}")
    else:
        print(
            f"ERROR: Preprocessor pickle file NOT found at: {config.PREPROCESSOR_SAVE_PATH}"
        )
    print("\n--- Preprocessing Module Test with Granular Functions Completed ---")
