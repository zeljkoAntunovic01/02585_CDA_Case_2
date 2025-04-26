# src/hr_data_loader.py

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

def run_data_loading():
    # base = Path(__file__).parent.parent.parent / 'data' / 'raw' 
    # print("Loading datasets from:", base)
    # hr_file = base / 'HR_data.csv'
    # hr_df = load_hr_data(hr_file)
    # X, y = split_features_and_labels(hr_df)

    # print("Loaded HR_data.csv with shape:", hr_df.shape)
    # print(hr_df.head)
    # print("Feature matrix X shape:", X.shape)
    # print(X.head())
    # print("Labels matrix y shape:", y.shape)
    # print(y.head())
    # return hr_df, X, y
    # Determine directories
    base = Path(__file__).parent.parent.parent  # project root
    raw = base / 'data' / 'raw'
    processed = base / 'data' / 'processed'

    # Ensure processed directory exists
    processed.mkdir(parents=True, exist_ok=True)

    print("Loading datasets from:", raw)
    hr_file = raw / 'HR_data.csv'
    hr_df = load_hr_data(hr_file)
    X, y = split_features_and_labels(hr_df)

    # Save processed outputs
    X.to_csv(processed / 'X.csv')
    y.to_csv(processed / 'y.csv')
    hr_df.to_csv(processed / 'HR_data_processed.csv')

    print("Saved processed data to:", processed)
    # print("HR_data shape:", hr_df.shape)
    # print(hr_df.head())
    # print("Feature matrix X shape:", X.shape)
    # print(X.head())
    # print("Labels matrix y shape:", y.shape)
    # print(y.head())

    return hr_df, X, y