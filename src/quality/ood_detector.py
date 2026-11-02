"""
Out-of-Distribution (OOD) Fallback Detector
Implements feature range checks and Mahalanobis distance scoring (Section 8.2)
to identify excursion lots and anomalous dies before trusting ML inferences.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis


class OODDetector:
    """
    Detects whether a die's pre-test sensor readings lie outside the training distribution.
    If an OOD condition is detected, the unit is routed to full binary search.
    """
    def __init__(self, quantile_threshold: float = 0.99):
        self.quantile_threshold = quantile_threshold
        self.feature_means: np.ndarray = np.array([])
        self.inv_cov_matrix: np.ndarray = np.array([])
        self.distance_threshold: float = 0.0
        self.min_bounds: pd.Series = pd.Series(dtype=float)
        self.max_bounds: pd.Series = pd.Series(dtype=float)
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame):
        """
        Fits training distribution statistics: feature means, covariance matrix,
        and empirical distance threshold.
        """
        self.feature_means = X_train.mean(axis=0).values
        cov_matrix = np.cov(X_train.values, rowvar=False)
        # Add small regularization to avoid singularity
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        self.inv_cov_matrix = np.linalg.pinv(cov_matrix)
        
        # Calculate training distances
        train_distances = self._compute_mahalanobis(X_train.values)
        self.distance_threshold = float(np.percentile(train_distances, 100 * self.quantile_threshold))
        
        # Feature range boundaries with 5% padding
        ranges = X_train.max(axis=0) - X_train.min(axis=0)
        self.min_bounds = X_train.min(axis=0) - 0.05 * ranges
        self.max_bounds = X_train.max(axis=0) + 0.05 * ranges
        self.is_fitted = True

    def _compute_mahalanobis(self, X_values: np.ndarray) -> np.ndarray:
        """Computes Mahalanobis distances for each row vector."""
        diff = X_values - self.feature_means
        # Vectorized Mahalanobis distance calculation: sqrt(diag(diff @ inv_cov @ diff.T))
        left = np.dot(diff, self.inv_cov_matrix)
        distances = np.sqrt(np.sum(left * diff, axis=1))
        return distances

    def check_unit(self, unit_features: pd.Series) -> Tuple[bool, str, float]:
        """
        Checks a single unit.
        Returns:
            (is_ood, reason, distance)
        """
        if not self.is_fitted:
            raise RuntimeError("OODDetector must be fitted before checking units.")
            
        # 1. Feature Range Check
        range_fail = (unit_features < self.min_bounds) | (unit_features > self.max_bounds)
        if range_fail.any():
            failing_cols = list(unit_features.index[range_fail])
            return True, f"Feature range boundary violation on: {', '.join(failing_cols[:3])}", 0.0
            
        # 2. Multivariate Mahalanobis Check
        dist = float(self._compute_mahalanobis(unit_features.values.reshape(1, -1))[0])
        if dist > self.distance_threshold:
            return True, f"Mahalanobis distance ({dist:.2f}) exceeds threshold ({self.distance_threshold:.2f})", dist
            
        return False, "Nominal In-Distribution", dist

    def evaluate_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates a batch of units and returns OOD flags and distances.
        """
        distances = self._compute_mahalanobis(X.values)
        is_ood_dist = distances > self.distance_threshold
        
        # Check range bounds
        out_of_min = (X < self.min_bounds).any(axis=1)
        out_of_max = (X > self.max_bounds).any(axis=1)
        is_ood_range = out_of_min | out_of_max
        
        is_ood = is_ood_dist | is_ood_range
        
        return pd.DataFrame({
            "mahalanobis_distance": distances,
            "is_ood": is_ood,
            "threshold": self.distance_threshold
        }, index=X.index)

