"""
Drift Monitoring and Statistical Process Control (SPC) Module
Implements EWMA and CUSUM control charts on prediction residuals (Section 8.4)
and Population Stability Index (PSI) to detect drift in manufacturing lots.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_psi(reference: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    """
    Computes the Population Stability Index (PSI) between a reference and target distribution.
    PSI < 0.10: No significant shift
    0.10 <= PSI < 0.25: Moderate shift
    PSI >= 0.25: Significant shift requiring model retraining
    """
    ref_clean = reference[~np.isnan(reference)]
    tgt_clean = target[~np.isnan(target)]
    
    if len(ref_clean) == 0 or len(tgt_clean) == 0:
        return 0.0
        
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    tgt_counts, _ = np.histogram(tgt_clean, bins=bin_edges)
    
    # Avoid zero division with smoothing
    ref_pct = (ref_counts + 1e-4) / (len(ref_clean) + 1e-4 * bins)
    tgt_pct = (tgt_counts + 1e-4) / (len(tgt_clean) + 1e-4 * bins)
    
    psi_value = np.sum((tgt_pct - ref_pct) * np.log(tgt_pct / ref_pct))
    return float(psi_value)


class DriftMonitor:
    """
    Monitors residuals of audited units over time using EWMA and CUSUM charts.
    """
    def __init__(self, lambda_ewma: float = 0.2, l_sigma: float = 3.0, k_cusum: float = 0.5, h_cusum: float = 4.0):
        self.lambda_ewma = lambda_ewma
        self.l_sigma = l_sigma
        self.k_cusum = k_cusum
        self.h_cusum = h_cusum
        
    def calculate_ewma(self, residuals: np.ndarray, sigma_r: float) -> pd.DataFrame:
        """
        Computes EWMA chart values, upper and lower control limits (UCL, LCL).
        """
        n = len(residuals)
        ewma_vals = np.zeros(n)
        ucl = np.zeros(n)
        lcl = np.zeros(n)
        
        current_ewma = 0.0
        for i in range(n):
            current_ewma = self.lambda_ewma * residuals[i] + (1 - self.lambda_ewma) * current_ewma
            ewma_vals[i] = current_ewma
            
            # Time-varying standard deviation for EWMA
            factor = np.sqrt((self.lambda_ewma / (2.0 - self.lambda_ewma)) * (1.0 - (1.0 - self.lambda_ewma) ** (2 * (i + 1))))
            limit = self.l_sigma * sigma_r * factor
            ucl[i] = limit
            lcl[i] = -limit
            
        return pd.DataFrame({
            "residual": residuals,
            "ewma": ewma_vals,
            "ucl": ucl,
            "lcl": lcl,
            "alarm": (ewma_vals > ucl) | (ewma_vals < lcl)
        })

    def calculate_cusum(self, residuals: np.ndarray, sigma_r: float) -> pd.DataFrame:
        """
        Computes two-sided tabular CUSUM chart for mean shift detection.
        """
        n = len(residuals)
        c_plus = np.zeros(n)
        c_minus = np.zeros(n)
        threshold = self.h_cusum * sigma_r
        k = self.k_cusum * sigma_r
        
        for i in range(1, n):
            c_plus[i] = max(0.0, c_plus[i - 1] + residuals[i] - k)
            c_minus[i] = min(0.0, c_minus[i - 1] + residuals[i] + k)
            
        return pd.DataFrame({
            "residual": residuals,
            "c_plus": c_plus,
            "c_minus": c_minus,
            "threshold_upper": threshold,
            "threshold_lower": -threshold,
            "alarm": (c_plus > threshold) | (c_minus < -threshold)
        })

