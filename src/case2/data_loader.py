# src/hr_data_loader.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def load_hr_data(
    file_path: Path
) -> pd.DataFrame:
    """
    Load the aggregated HR_data.csv file containing extracted features and response labels.

    Parameters
    ----------
    file_path : Path
        Path to the HR_data.csv file.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by (Individual, Round, Phase) with all feature and response columns.
    """
    # Read CSV, treating the first unnamed column as a generic index
    df = pd.read_csv(file_path, index_col=0)

    # Rename the index for clarity (optional)
    df.index.name = 'Entry'

    # Convert Round and Phase to consistent types
    if 'Round' in df.columns:
        df['Round'] = df['Round'].astype(str)
    if 'Phase' in df.columns:
        df['Phase'] = df['Phase'].astype(str)

    df.set_index(['Individual', 'Round', 'Phase'], inplace=True)
    return df


def split_features_and_labels(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the loaded DataFrame into features (X) and labels/responses (y).

    Features include biosignal-derived columns; labels include questionnaire & metadata.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_hr_data.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        Labels/responses matrix.
    """
    label_cols = [
        'Puzzler', 'Frustrated', 'Cohort',
        'upset','hostile','alert','ashamed','inspired',
        'nervous','attentive','afraid','active','determined'
    ]

    # All other columns
    feature_cols = [c for c in df.columns if c not in label_cols]

    X = df[feature_cols].copy()
    y = df[label_cols].copy()
    return X, y

def preprocess_data_for_subspace_methods(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data by centering and normalizing.

    Parameters
    ----------
    X : pd.DataFrame
        Physiological input data.
    y : pd.DataFrame
        Labels/responses matrix.

    Returns
    -------
    X_preprocessed : pd.DataFrame
        Preprocessed physiological input data.
    y_preprocessed : pd.DataFrame
        Preprocessed labels/responses matrix.
    """
    X_centered = X - X.mean(axis=0)

    S = np.linalg.norm(X_centered, axis=0)
    X_preprocessed = (X_centered / S)

    cohort_mapping = {"D1_1": 1, "D1_2": 2, "D1_3": 3, "D1_4": 4, "D1_5": 5, "D1_6": 6}
    y_mapped = y.copy()
    if "Cohort" in y.columns:
        y_mapped["Cohort"] = y_mapped["Cohort"].map(cohort_mapping)

    y_preprocessed = y_mapped.fillna(-1)
    return X_preprocessed, y_preprocessed

def run_data_loading():
    # Determine directories
    base = Path(__file__).parent.parent.parent  # project root
    raw = base / 'data' / 'raw'

    print("Loading datasets from:", raw)
    hr_file = raw / 'HR_data.csv'
    hr_df = load_hr_data(hr_file)
    X, y = split_features_and_labels(hr_df)

    # Impute missing values in X, float cast numeric columns
    X = X.fillna(X.median())
    X = X.copy().astype(float)

    return hr_df, X, y