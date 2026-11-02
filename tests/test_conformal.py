"""
Unit tests for data loader, model training, and conformal calibration.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_raw_dataset, validate_dataset, prepare_ml_splits
from src.models.train import FmaxModelTrainer
from src.models.conformal import CalibratedConformalPredictor
from src.quality.ood_detector import OODDetector


def test_data_loader_integrity():
    df = load_raw_dataset()
    val_status = validate_dataset(df)
    
    assert val_status["is_valid"]
    assert val_status["total_rows"] == 6000
    assert val_status["total_missing"] == 0


def test_ml_pipeline_and_conformal():
    df = load_raw_dataset()
    X_train, y_train, X_cal, y_cal, X_test, y_test, _ = prepare_ml_splits(df)
    
    # Train Ridge baseline for fast test execution
    trainer = FmaxModelTrainer()
    trainer.models = {"Ridge": trainer.models["Ridge Regression"]}
    summary_df = trainer.train_all(X_train, y_train, X_test, y_test)
    
    assert "RMSE (MHz)" in summary_df.columns
    best_name, best_model = trainer.get_best_model()
    
    # Calibrate conformal predictor
    conformal = CalibratedConformalPredictor(best_model)
    conformal.calibrate(X_cal, y_cal)
    
    assert conformal.is_calibrated
    assert conformal.sigma_residual > 0.0
    
    # Check coverage on test
    cov_metrics = conformal.evaluate_coverage(X_test, y_test, k=1.5)
    assert cov_metrics["coverage_pct"] > 70.0  # nominal 1.5 sigma interval


def test_ood_detector():
    df = load_raw_dataset()
    X_train, _, _, _, X_test, _, _ = prepare_ml_splits(df)
    
    detector = OODDetector(quantile_threshold=0.99)
    detector.fit(X_train)
    
    # In-distribution sample
    is_ood, reason, dist = detector.check_unit(X_train.iloc[0])
    assert not is_ood
    
    # Artificially extreme sample (OOD excursion)
    ood_sample = X_train.iloc[0].copy()
    ood_sample["ro_01"] = 999.0
    is_ood_bad, reason_bad, _ = detector.check_unit(ood_sample)
    assert is_ood_bad

