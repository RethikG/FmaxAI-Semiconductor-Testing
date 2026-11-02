"""
Unit tests for physical tester environment and search algorithms.
"""

import pytest
import numpy as np
from src.config import TesterConfig
from src.simulation.tester_env import SiliconTester
from src.simulation.search_algorithms import (
    run_linear_sweep,
    run_binary_search,
    run_ml_seeded_walk,
    run_prediction_windowed_binary,
    run_probabilistic_bisection,
    run_confirm_only_binning
)


@pytest.fixture
def test_config():
    return TesterConfig(sigma_test_mhz=0.0)  # Deterministic test for exact assertions


def test_linear_sweep_exact(test_config):
    true_fmax = 1964.0
    tester = SiliconTester(true_fmax, test_config)
    res = run_linear_sweep(tester, test_config)
    
    # 1964 with 10 MHz grid -> highest passing point is 1960.0
    assert res["measured_fmax"] == 1960.0
    assert res["steps"] > 30


def test_binary_search_exact(test_config):
    true_fmax = 1964.0
    tester = SiliconTester(true_fmax, test_config)
    res = run_binary_search(tester, test_config)
    
    assert res["measured_fmax"] == 1960.0
    assert res["steps"] in [6, 7]


def test_windowed_binary_search_inside_window(test_config):
    true_fmax = 1964.0
    y_hat = 1960.0
    L = 1920.0
    U = 2000.0
    
    tester = SiliconTester(true_fmax, test_config)
    res = run_prediction_windowed_binary(tester, y_hat, L, U, test_config)
    
    assert res["measured_fmax"] == 1960.0
    # Window width is 80 MHz (8 steps), binary search inside takes ~3-4 steps!
    assert res["steps"] <= 5
    assert not res["edge_hit"]


def test_windowed_binary_search_upper_edge_recovery(test_config):
    # True Fmax exceeds upper bound of window
    true_fmax = 2080.0
    y_hat = 1960.0
    L = 1920.0
    U = 2000.0
    
    tester = SiliconTester(true_fmax, test_config)
    res = run_prediction_windowed_binary(tester, y_hat, L, U, test_config)
    
    # Must recover and find 2080.0 despite window under-prediction
    assert res["measured_fmax"] == 2080.0
    assert res["edge_hit"]


def test_confirm_only_binning(test_config):
    true_fmax = 2045.0
    y_hat = 2040.0
    sigma = 25.0
    
    tester = SiliconTester(true_fmax, test_config)
    res = run_confirm_only_binning(tester, y_hat, sigma, config=test_config)
    
    # With 8 MHz guard band, 2045 passes 2000 + 8 = 2008, but fails 2100 + 8 = 2108
    assert res["assigned_bin"] == "2000"
    assert res["steps"] <= 3

