# src/preprocessing/transformers.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pathlib
from typing import List, Tuple, Any, Optional

from src import config  # To access feature lists, target column, etc.
from src.utils import helpers  # For saving/loading scalers and encoders


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: Optional[int] = None,
    stratify_col: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target variable column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (Optional[int]): Controls the shuffling applied to the data before splitting.
        stratify_col (Optional[pd.Series]): If not None, data is split in a stratified fashion,
                                            using this as the class labels.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=(
            stratify_col if stratify_col is not None else y
        ),  # Stratify by target by default if classification
    )
    print(f"Data split into training and testing sets.")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    preprocessor_save_path: Optional[pathlib.Path] = None,
    fit_preprocessor: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[ColumnTransformer]]:
    """
    Applies preprocessing (imputation, scaling for numerical; imputation, OHE for categorical).
    Can fit a new preprocessor or load an existing one.

    Args:
        X_train (pd.DataFrame): Training feature DataFrame.
        X_test (pd.DataFrame): Testing feature DataFrame.
        numerical_features (List[str]): List of numerical feature column names.
        categorical_features (List[str]): List of categorical feature column names.
        preprocessor_save_path (Optional[pathlib.Path]): Path to save/load the fitted ColumnTransformer.
        fit_preprocessor (bool): If True, fits a new preprocessor. If False, loads from preprocessor_save_path.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Optional[ColumnTransformer]]:
            X_train_processed, X_test_processed, fitted_preprocessor (or loaded)
    """
    # Define preprocessing steps for numerical features
    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # Impute missing values with median
            ("scaler", StandardScaler()),  # Scale features
        ]
    )

    # Define preprocessing steps for categorical features
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),  # Impute missing values with most frequent
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),  # OHE
        ]
    )

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",  # Keep other columns (if any) not specified
    )

    if fit_preprocessor:
        print("Fitting new preprocessor...")
        X_train_processed = preprocessor.fit_transform(X_train)
        if preprocessor_save_path:
            helpers.save_pickle(preprocessor, preprocessor_save_path)
            print(f"Preprocessor saved to {preprocessor_save_path}")
    else:
        if not preprocessor_save_path or not preprocessor_save_path.exists():
            raise ValueError(
                "preprocessor_save_path must be provided and exist if fit_preprocessor is False."
            )
        print(f"Loading preprocessor from {preprocessor_save_path}...")
        preprocessor = helpers.load_pickle(preprocessor_save_path)
        X_train_processed = preprocessor.transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after OHE for creating DataFrames (optional, but good for inspection)
    # This can be complex if you have many OHE features.
    # For simplicity, we'll return NumPy arrays first, can convert to DF later if needed.
    # If you need DataFrames with proper column names:
    try:
        # Get feature names after transformation
        # For 'passthrough' columns, their names are just the column names.
        # For 'num' transformer, names are the original numerical feature names.
        # For 'cat' transformer (OneHotEncoder), names are generated.
        ohe_feature_names = (
            preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_features)
        )
        processed_feature_names = numerical_features + list(ohe_feature_names)

        # Handle 'remainder' columns if any were passed through
        if preprocessor.remainder == "passthrough" and hasattr(
            preprocessor, "feature_names_in_"
        ):
            input_features = X_train.columns.tolist()
            transformed_features_flat = []
            if "num" in preprocessor.named_transformers_:
                transformed_features_flat.extend(numerical_features)
            if "cat" in preprocessor.named_transformers_:
                transformed_features_flat.extend(list(ohe_feature_names))

            remainder_features = [
                f
                for f in input_features
                if f not in numerical_features and f not in categorical_features
            ]
            processed_feature_names.extend(remainder_features)

        X_train_processed_df = pd.DataFrame(
            X_train_processed, columns=processed_feature_names, index=X_train.index
        )
        X_test_processed_df = pd.DataFrame(
            X_test_processed, columns=processed_feature_names, index=X_test.index
        )
        print("Processed features returned as DataFrames with inferred column names.")
    except Exception as e:
        print(
            f"Could not construct DataFrame with feature names: {e}. Returning NumPy arrays."
        )
        X_train_processed_df = X_train_processed  # type: ignore
        X_test_processed_df = X_test_processed  # type: ignore

    print(f"X_train_processed shape: {X_train_processed_df.shape}")  # type: ignore
    print(f"X_test_processed shape: {X_test_processed_df.shape}")  # type: ignore

    return X_train_processed_df, X_test_processed_df, preprocessor


def encode_target(
    y_train: pd.Series,
    y_test: pd.Series,
    encoder_save_path: Optional[pathlib.Path] = None,
    fit_encoder: bool = True,
) -> Tuple[pd.Series, pd.Series, Optional[LabelEncoder]]:
    """
    Encodes the target variable using LabelEncoder.
    Can fit a new encoder or load an existing one.

    Args:
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
        encoder_save_path (Optional[pathlib.Path]): Path to save/load the fitted LabelEncoder.
        fit_encoder (bool): If True, fits a new encoder. If False, loads from encoder_save_path.

    Returns:
        Tuple[pd.Series, pd.Series, Optional[LabelEncoder]]:
            y_train_encoded, y_test_encoded, fitted_encoder (or loaded)
    """
    if fit_encoder:
        print("Fitting new LabelEncoder for target...")
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        if encoder_save_path:
            helpers.save_pickle(encoder, encoder_save_path)
            print(f"LabelEncoder saved to {encoder_save_path}")
    else:
        if not encoder_save_path or not encoder_save_path.exists():
            raise ValueError(
                "encoder_save_path must be provided and exist if fit_encoder is False."
            )
        print(f"Loading LabelEncoder from {encoder_save_path}...")
        encoder = helpers.load_pickle(encoder_save_path)
        y_train_encoded = encoder.transform(y_train)

    y_test_encoded = encoder.transform(y_test)

    print(
        f"Target variable encoded. Example mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}"
    )
    return (
        pd.Series(y_train_encoded, name=y_train.name, index=y_train.index),
        pd.Series(y_test_encoded, name=y_test.name, index=y_test.index),
        encoder,
    )


if __name__ == "__main__":
    # This block is for testing the preprocessing module directly.
    # To run this:
    # 1. Make sure your conda environment is activated.
    # 2. Navigate to the 'dynamic_dnn_trainer' project root directory in your terminal.
    # 3. Run: python -m src.preprocessing.transformers

    print("--- Testing preprocessing functions ---")

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

    # --- Test data splitting ---
    print("\n--- Testing data splitting ---")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_data(
        df=raw_df,
        target_column=config.TARGET_COLUMN,
        test_size=config.TEST_SPLIT_SIZE,
        random_state=config.RANDOM_STATE,
        stratify_col=raw_df[config.TARGET_COLUMN],  # Stratify for classification
    )
    print(f"X_train_raw head:\n{X_train_raw.head()}")

    # --- Test feature preprocessing ---
    print("\n--- Testing feature preprocessing (fitting new) ---")
    preprocessor_path = config.PROCESSED_DATA_DIR / "test_preprocessor.pkl"
    X_train_proc, X_test_proc, fitted_preprocessor = preprocess_features(
        X_train=X_train_raw.copy(),  # Use copy to avoid SettingWithCopyWarning on potential imputation
        X_test=X_test_raw.copy(),
        numerical_features=config.NUMERICAL_FEATURES,
        categorical_features=config.CATEGORICAL_FEATURES,
        preprocessor_save_path=preprocessor_path,
        fit_preprocessor=True,
    )
    print(f"X_train_proc head:\n{X_train_proc.head()}")  # type: ignore
    print(f"Fitted preprocessor: {fitted_preprocessor}")

    # --- Test loading existing preprocessor ---
    if preprocessor_path.exists():
        print("\n--- Testing feature preprocessing (loading existing) ---")
        X_train_proc_load, X_test_proc_load, loaded_preprocessor = preprocess_features(
            X_train=X_train_raw.copy(),
            X_test=X_test_raw.copy(),
            numerical_features=config.NUMERICAL_FEATURES,
            categorical_features=config.CATEGORICAL_FEATURES,
            preprocessor_save_path=preprocessor_path,
            fit_preprocessor=False,  # Load existing
        )
        # Basic check: compare shapes or a few values if complex
        if X_train_proc.shape == X_train_proc_load.shape:  # type: ignore
            print("Shapes match for loaded preprocessor output. Test likely PASSED.")
        else:
            print("Shapes DO NOT match for loaded preprocessor output. Test FAILED.")
        print(f"Loaded preprocessor: {loaded_preprocessor}")

    # --- Test target encoding ---
    print("\n--- Testing target encoding (fitting new) ---")
    target_encoder_path = config.PROCESSED_DATA_DIR / "test_target_encoder.pkl"
    y_train_enc, y_test_enc, fitted_target_encoder = encode_target(
        y_train=y_train_raw.copy(),
        y_test=y_test_raw.copy(),
        encoder_save_path=target_encoder_path,
        fit_encoder=True,
    )
    print(f"y_train_enc head:\n{y_train_enc.head()}")
    print(f"Fitted target encoder classes: {fitted_target_encoder.classes_}")  # type: ignore

    # --- Test loading existing target encoder ---
    if target_encoder_path.exists():
        print("\n--- Testing target encoding (loading existing) ---")
        y_train_enc_load, y_test_enc_load, loaded_target_encoder = encode_target(
            y_train=y_train_raw.copy(),
            y_test=y_test_raw.copy(),
            encoder_save_path=target_encoder_path,
            fit_encoder=False,  # Load existing
        )
        if y_train_enc.equals(y_train_enc_load):
            print("Loaded target encoder output matches. Test PASSED.")
        else:
            print("Loaded target encoder output DOES NOT match. Test FAILED.")
        print(f"Loaded target encoder classes: {loaded_target_encoder.classes_}")  # type: ignore

    print("\nPreprocessing tests completed.")
    print(
        f"Check {config.PROCESSED_DATA_DIR} for 'test_preprocessor.pkl' and 'test_target_encoder.pkl'."
    )
