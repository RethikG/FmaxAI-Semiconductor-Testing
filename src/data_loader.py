"""
Data Loading and Validation Module for FmaxAI
Handles safe data ingestion, split separation (train/calibration/test),
and guarantees zero target/downstream leakage.
"""

import os
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import PRE_TEST_FEATURES, DOWNSTREAM_LEAKAGE_COLUMNS


def load_raw_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the synthetic Fmax dataset from CSV.
    """
    if filepath is None:
        # Default to synthetic_fmax_dataset.csv in workspace root or data folder
        candidates = [
            "synthetic_fmax_dataset.csv",
            os.path.join("data", "synthetic_fmax_dataset.csv"),
            os.path.join(os.path.dirname(__file__), "..", "synthetic_fmax_dataset.csv")
        ]
        for c in candidates:
            if os.path.exists(c):
                filepath = c
                break
        if filepath is None:
            raise FileNotFoundError("Could not locate synthetic_fmax_dataset.csv")

    df = pd.read_csv(filepath)
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validates dataset integrity, checks for nulls, verifies column availability,
    and returns a summary status dictionary.
    """
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    
    expected_splits = {'train', 'calibration', 'test'}
    actual_splits = set(df['split'].unique()) if 'split' in df.columns else set()
    split_valid = expected_splits.issubset(actual_splits)
    
    feature_check = all(feat in df.columns for feat in PRE_TEST_FEATURES)
    target_check = 'fmax_true_mhz' in df.columns
    
    return {
        "is_valid": total_missing == 0 and split_valid and feature_check and target_check,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_missing": total_missing,
        "splits_present": list(actual_splits),
        "feature_check_passed": feature_check,
        "target_check_passed": target_check
    }


def prepare_ml_splits(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame
]:
    """
    Extracts strictly pre-test features and targets according to the documented split:
    - Train (1,500 units)
    - Calibration (500 units)
    - Test (4,000 units)
    
    Returns:
    (X_train, y_train, X_cal, y_cal, X_test, y_test, full_df)
    """
    # Guard against accidental target leakage:
    for col in PRE_TEST_FEATURES:
        if col in DOWNSTREAM_LEAKAGE_COLUMNS:
            raise ValueError(f"CRITICAL: Leakage detected. {col} is a downstream column!")

    train_mask = df['split'] == 'train'
    cal_mask = df['split'] == 'calibration'
    test_mask = df['split'] == 'test'
    
    X_train = df.loc[train_mask, PRE_TEST_FEATURES].copy()
    y_train = df.loc[train_mask, 'fmax_true_mhz'].copy()
    
    X_cal = df.loc[cal_mask, PRE_TEST_FEATURES].copy()
    y_cal = df.loc[cal_mask, 'fmax_true_mhz'].copy()
    
    X_test = df.loc[test_mask, PRE_TEST_FEATURES].copy()
    y_test = df.loc[test_mask, 'fmax_true_mhz'].copy()
    
    return X_train, y_train, X_cal, y_cal, X_test, y_test, df

