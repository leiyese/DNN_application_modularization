

**Overall Project Structure (Reminder):**
```
dynamic_dnn_trainer/
├── notebooks/
│   └── dnn_training_workflow.ipynb  # (and later cnn_training_workflow.ipynb)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── transformers.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   └── dnn_builder.py         # (will become model_builder.py in Part 5)
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics_suite.py
│   │   └── plots.py
│   ├── tuning/
│   │   ├── __init__.py
│   │   └── hyperparameter_search.py
│   ├── eda/
│   │   ├── __init__.py
│   │   └── exploratory_analysis.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── main.py
├── requirements.txt
├── environment.yml (for Conda)
├── data/
│   └── raw/
│   └── processed/
└── outputs/
    ├── models/
    ├── plots/
    └── reports/
```

**Part 1: Foundation & Core Data Pipeline (DNN Focus)**

*   **Goal:** Establish the project structure, configuration, and a data loading/preprocessing pipeline (within `src/`) for tabular data suitable for a DNN.
*   **Specific Tasks:**
    1.  **Project Setup:**
        *   Create main directory `dynamic_dnn_trainer/`. Initialize Git.
        *   Set up Conda environment (`dynamic_dnn_env`), `requirements.txt`, `environment.yml`.
        *   Create the directory structure: `notebooks/`, `src/`, `data/`, `outputs/`.
        *   Inside `src/`, create `__init__.py`.
        *   Create subdirectories within `src/`: `data_ingestion/`, `preprocessing/`, `modeling/`, `evaluation/`, `tuning/`, `eda/`, `utils/`. Add `__init__.py` to each of these subdirectories.
    2.  **Configuration Module:**
        *   Create `src/config.py`.
        *   Define initial configurations: data paths, sample tabular dataset parameters, basic preprocessing flags.
    3.  **Data Ingestion Module:**
        *   Create `src/data_ingestion/loader.py`.
        *   Implement `load_csv(filepath, **kwargs)` function.
    4.  **Utilities Module (Basic):**
        *   Create `src/utils/helpers.py`.
        *   Implement `save_pickle(obj, filepath)` and `load_pickle(filepath)`.
    5.  **Preprocessing Module:**
        *   Create `src/preprocessing/transformers.py`.
        *   Implement:
            *   `split_data(X, y, test_size, random_state)`
            *   `scale_numerical_features(df_train, df_test, columns_to_scale, scaler_path)` (saves/loads scaler using `src/utils/helpers.py`)
            *   `encode_categorical_features(df_train, df_test, columns_to_encode, encoder_path, target_column=None)` (handles features and optionally target; saves/loads encoder(s) using `src/utils/helpers.py`)
    6.  **Basic EDA Module:**
        *   Create `src/eda/exploratory_analysis.py`.
        *   Implement `generate_descriptive_stats(df)`, `get_null_counts(df)`.
    7.  **Initial Colab Notebook:**
        *   Create `notebooks/dnn_training_workflow.ipynb`.
        *   Include cells for: cloning repo (if needed), `cd` into project, `sys.path.append('src')`.
        *   Test importing from `src/` modules (e.g., `from config import ...`, `from data_ingestion.loader import load_csv`).
        *   Orchestrate data loading, EDA, and preprocessing using the created `src/` modules.
    8.  **Initial `main.py`:**
        *   Create `main.py` at the project root.
        *   Implement basic CLI argument parsing (e.g., for config file path).
        *   Orchestrate data loading, EDA, and preprocessing by importing and calling functions from `src/` modules.
*   **Deliverable:** A structured project with a functional pipeline (all code in `src/`) to load, perform basic EDA on, and preprocess tabular data.

**Part 2: DNN Modeling, Training & Basic Evaluation**

*   **Goal:** Implement DNN model building, training, and initial evaluation plots (loss/accuracy) using modules within `src/`.
*   **Specific Tasks:**
    1.  **DNN Builder Module:**
        *   Create `src/modeling/dnn_builder.py`.
        *   Implement `build_dnn_model(input_shape, layers_config, optimizer_config, loss, metrics)` based on parameters from `src/config.py`.
    2.  **Trainer Module:**
        *   Create `src/modeling/trainer.py`.
        *   Implement `train_model(model, X_train, y_train, X_val, y_val, training_params, model_save_path, history_save_path)`:
            *   Sets up `ModelCheckpoint`, `EarlyStopping`, `TensorBoard` (optional) callbacks.
            *   Calls `model.fit()`.
            *   Saves the best model and the training history (using `src/utils/helpers.py` for history).
    3.  **Basic Evaluation Plotting Module:**
        *   Create `src/evaluation/plots.py`.
        *   Implement `plot_training_history(history, loss_plot_path, acc_plot_path)`.
    4.  **Integration:**
        *   Update `notebooks/dnn_training_workflow.ipynb` and `main.py` to:
            *   Import from `src/modeling/dnn_builder.py` and `src/modeling/trainer.py`.
            *   Build, train, and save the DNN.
            *   Import from `src/evaluation/plots.py` to plot training history.
*   **Deliverable:** Ability to dynamically define (via `src/modeling/dnn_builder.py`), train (via `src/modeling/trainer.py`), save a DNN, and visualize its training progress (via `src/evaluation/plots.py`).

**Part 3: Comprehensive DNN Evaluation Suite**

*   **Goal:** Implement the full suite of evaluation metrics and visualizations for DNN classification tasks, all within `src/evaluation/`.
*   **Specific Tasks:**
    1.  **Metrics Suite Module:**
        *   Create `src/evaluation/metrics_suite.py`.
        *   Implement functions:
            *   `get_predictions(model, X, model_type='dnn')` (handles different output shapes if needed later)
            *   `calculate_auc_roc_data(y_true, y_pred_proba)`
            *   `generate_confusion_matrix_data(y_true, y_pred_classes)`
            *   `generate_classification_report_data(y_true, y_pred_classes)`
            *   `calculate_precision_recall_data(y_true, y_pred_proba)`
            *   `calculate_permutation_importance_data(model, X, y, scoring, n_repeats, random_state)`
    2.  **Plotting Module (Extended):**
        *   Extend `src/evaluation/plots.py` with:
            *   `plot_roc_curve(fpr, tpr, roc_auc, save_path)`
            *   `plot_confusion_matrix(cm_data, class_names, save_path)`
            *   `plot_precision_recall_curve(precision, recall, average_precision, save_path)`
            *   `plot_feature_importance(importances_df, save_path)`
    3.  **Integration:**
        *   Update `notebooks/dnn_training_workflow.ipynb` and `main.py` to load the trained model, make predictions, and use the new functions from `src/evaluation/metrics_suite.py` and `src/evaluation/plots.py` for comprehensive evaluation.
*   **Deliverable:** A comprehensive evaluation framework within `src/evaluation/` for assessing DNN classification performance.

**Part 4: DNN Hyperparameter Tuning & Refinements**

*   **Goal:** Add DNN hyperparameter tuning using `src/tuning/` and refine the overall application (configuration, utilities, EDA within `src/`).
*   **Specific Tasks:**
    1.  **Hyperparameter Tuning Module:**
        *   Create `src/tuning/hyperparameter_search.py`.
        *   Implement `perform_grid_search(X_train, y_train, X_val, y_val, model_build_fn_ref, param_grid, cv_params, fit_params)`:
            *   `model_build_fn_ref` will point to `src/modeling/dnn_builder.build_dnn_model`.
            *   Uses `GridSearchCV` with a Keras wrapper.
            *   Parameter grid for DNNs defined in `src/config.py`.
    2.  **EDA Module Enhancement:**
        *   Extend `src/eda/exploratory_analysis.py` with `plot_correlation_matrix(df, save_path)`, `plot_histograms_for_columns(df, columns, save_path_prefix)`.
    3.  **Configuration Enhancement:**
        *   Refine `src/config.py`. Consider adding support for loading parts of the config from YAML/JSON (e.g., using a helper in `src/utils/helpers.py`).
    4.  **Utilities Module Enhancement:**
        *   Extend `src/utils/helpers.py` with `setup_logger()` and functions to ensure output directories exist.
    5.  **Integration:**
        *   Incorporate grid search from `src/tuning/hyperparameter_search.py` into `notebooks/dnn_training_workflow.ipynb` and `main.py`.
        *   Improve output organization in `outputs/` (e.g., subfolders for experiments).
    6.  **Documentation & Testing:** Add docstrings to all functions/modules in `src/`. Consider writing basic unit tests (e.g., in a separate `tests/` directory) for critical components in `src/`.
*   **Deliverable:** Ability to optimize DNN hyperparameters using `src/tuning/`, and a more polished, robust, and well-documented pipeline with all core logic in `src/`.

**Part 5: Adding CNN Compatibility**

*   **Goal:** Extend the framework (all modules in `src/`) to support Convolutional Neural Networks (CNNs), primarily for image classification.
*   **Specific Tasks:**
    1.  **Configuration (`src/config.py`):**
        *   Extend `src/config.py` to include `model_type: "cnn"`, CNN-specific layer parameters, input image dimensions, image preprocessing/augmentation flags and parameters.
    2.  **Preprocessing (Image Specific):**
        *   Extend `src/preprocessing/transformers.py` (or create `src/preprocessing/image_processor.py`):
            *   Functions for loading images from paths.
            *   Image resizing, normalization.
            *   Wrapper/helper for Keras `ImageDataGenerator` or `tf.data` pipelines for augmentation and batching.
            *   Function to prepare data for CNN input shape.
    3.  **Model Builder (Refactor):**
        *   Rename `src/modeling/dnn_builder.py` to `src/modeling/model_builder.py`.
        *   Inside `src/modeling/model_builder.py`:
            *   Keep `build_dnn_model(...)`.
            *   Add `build_cnn_model(input_shape, cnn_layers_config, dense_layers_config, optimizer_config, loss, metrics)`.
            *   Create a top-level function like `get_model(config)` that calls the appropriate builder based on `config['model_type']`.
    4.  **Data Ingestion (`src/data_ingestion/loader.py`):**
        *   Ensure `src/data_ingestion/loader.py` can load data suitable for CNNs (e.g., a CSV with image file paths and labels).
    5.  **EDA (`src/eda/exploratory_analysis.py`):**
        *   Add functions to `src/eda/exploratory_analysis.py` like `display_sample_images(image_paths, labels, n_samples)`.
    6.  **Training & Evaluation:**
        *   The `src/modeling/trainer.py`, `src/evaluation/metrics_suite.py`, and `src/evaluation/plots.py` should largely work as is for classification tasks, provided the model output is consistent. Minor adjustments might be needed in `get_predictions` if CNN output shapes differ significantly.
    7.  **New Colab Notebook / `main.py` updates:**
        *   Create `notebooks/cnn_training_workflow.ipynb` (or add sections to the existing one).
        *   Update `main.py` to handle `model_type: "cnn"` configurations.
        *   Demonstrate the full pipeline for an image dataset using the extended `src/` modules.
*   **Deliverable:** The framework, with all core logic in `src/`, can now dynamically build, train, and evaluate both DNNs (for tabular data) and CNNs (for image data) based on the provided configuration.




**Virtual Environment Recommendation:**

*   **Conda:**
    *   **Pros:**
        *   Manages Python versions *and* Python packages.
        *   Excels at handling complex binary dependencies, especially for scientific computing and ML libraries (like MKL, CUDA, cuDNN). This is crucial if you plan to use GPU acceleration locally in VSCode.
        *   Creates truly isolated environments.
        *   Widely used in the data science/ML community, so good support and many packages are readily available through conda-forge.
    *   **Cons:**
        *   Can be a bit heavier than `venv`.
        *   Sometimes `conda install` might lag behind `pip` for the very latest package versions (though `pip` can be used within a conda environment).



**Recommendation:** **Use Conda for local development in VSCode.** It will provide the most robust and hassle-free experience, especially if you intend to leverage local GPUs. You can still use `pip install -r requirements.txt` *inside* your activated conda environment if some packages are only on PyPI or you prefer pip's resolution for certain things.

For **Google Colab**, you don't manage the environment in the same way. Colab provides a pre-configured environment. You'll primarily use `!pip install` for any packages not already included or if you need specific versions. Your `requirements.txt` will be key for ensuring consistency in package versions between your local Conda env and Colab.
