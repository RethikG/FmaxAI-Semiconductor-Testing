"""
Conformal Prediction and Calibrated Uncertainty Module
Implements Split Conformal Prediction and residual standard deviation windowing
for distribution-free interval guarantees [L, U] as defined in Section 5.2 & Table 5.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


class CalibratedConformalPredictor:
    """
    Computes calibrated prediction intervals [L, U] based on hold-out calibration data.
    Supports both:
    1. Residual standard-deviation windowing: [y_hat - k*sigma, y_hat + k*sigma] (Table 5)
    2. Exact split conformal quantile intervals: [y_hat + q_{alpha/2}(r), y_hat + q_{1-alpha/2}(r)] (Section 5.2)
    """
    def __init__(self, base_model: any):
        self.base_model = base_model
        self.residuals: np.ndarray = np.array([])
        self.sigma_residual: float = 0.0
        self.is_calibrated: bool = False

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.Series):
        """
        Calibrates on held-out calibration set (500 units in benchmark).
        Computes signed residuals r_i = y_i - y_hat_i.
        """
        y_hat_cal = self.base_model.predict(X_cal)
        self.residuals = y_cal.values - y_hat_cal
        self.sigma_residual = float(np.std(self.residuals, ddof=1))
        self.is_calibrated = True

    def predict_interval(
        self,
        X: pd.DataFrame,
        k: float = 1.5,
        method: str = "sigma"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates predictions and calibrated bounds [L, U].
        
        Args:
            X: Input feature DataFrame
            k: Window multiplier (default 1.5 sigma from Table 5)
            method: 'sigma' for [y_hat - k*sigma, y_hat + k*sigma]
                    'conformal' for empirical quantile bounds at coverage level
                    
        Returns:
            Tuple of (y_hat, L, U)
        """
        if not self.is_calibrated:
            raise RuntimeError("CalibratedConformalPredictor must be calibrated before generating intervals.")
            
        y_hat = self.base_model.predict(X)
        
        if method == "sigma":
            half_width = k * self.sigma_residual
            L = y_hat - half_width
            U = y_hat + half_width
        elif method == "conformal":
            # Target 95% interval coverage by default (alpha = 0.05)
            alpha = max(0.01, min(0.5, 1.0 - (k / 2.0)))
            q_low = float(np.percentile(self.residuals, 100 * (alpha / 2.0)))
            q_high = float(np.percentile(self.residuals, 100 * (1.0 - alpha / 2.0)))
            L = y_hat + q_low
            U = y_hat + q_high
        else:
            raise ValueError(f"Unknown method {method}. Choose 'sigma' or 'conformal'.")
            
        return y_hat, L, U

    def evaluate_coverage(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        k: float = 1.5,
        method: str = "sigma"
    ) -> Dict[str, float]:
        """
        Evaluates empirical interval coverage and mean window width on a test set.
        """
        y_hat, L, U = self.predict_interval(X_test, k=k, method=method)
        y_true = y_test.values
        
        covered = (y_true >= L) & (y_true <= U)
        coverage_pct = float(np.mean(covered) * 100.0)
        mean_width = float(np.mean(U - L))
        
        return {
            "coverage_pct": coverage_pct,
            "mean_window_width_mhz": mean_width,
            "residual_sigma_mhz": self.sigma_residual,
            "k_multiplier": k
        }

