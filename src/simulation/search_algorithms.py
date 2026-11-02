"""
Fmax Search Algorithms Module
Implements:
- Conventional Linear Sweep
- Conventional Binary Search
- Naive ML-Seeded Walk
- Algorithm 1: Prediction-Windowed Binary Search (with dynamic edge recovery)
- Algorithm 2: Probabilistic Bisection with Learned Prior (Bayesian Adaptive Search)
- Algorithm 3: Confirm-Only Binning
- Algorithm 4: Multi-Corner (DVFS) Window Reuse
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.stats import norm

from src.simulation.tester_env import SiliconTester
from src.config import TesterConfig, DEFAULT_CONFIG


def binary_search_slice(
    tester: SiliconTester,
    subgrid: List[float]
) -> Optional[float]:
    """
    Standard binary search over a discrete grid slice.
    Returns the highest passing frequency in subgrid, or None if all fail.
    """
    if not subgrid:
        return None
        
    low = 0
    high = len(subgrid) - 1
    best = None
    
    while low <= high:
        mid = (low + high) // 2
        f = subgrid[mid]
        if tester.test(f):
            best = f
            low = mid + 1  # Passed: search upper half for higher frequency
        else:
            high = mid - 1  # Failed: search lower half
            
    return best


def run_linear_sweep(
    tester: SiliconTester,
    config: TesterConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Conventional Linear Sweep:
    Steps upward from grid minimum until the first failure.
    """
    grid = config.grid_points
    best = None
    trajectory = []
    
    for f in grid:
        passed = tester.test(f)
        trajectory.append((f, passed))
        if passed:
            best = f
        else:
            # First fail stops the linear walk
            break
            
    assigned_bin = tester.assign_bin(best)
    return {
        "method": "Linear sweep",
        "measured_fmax": best,
        "steps": tester.steps_executed,
        "time_ms": tester.total_test_time_ms,
        "assigned_bin": assigned_bin,
        "overbin_escape": tester.check_overbin_escape(assigned_bin),
        "underbin": tester.check_underbin(assigned_bin),
        "trajectory": tester.test_history.copy(),
        "edge_hit": False
    }


def run_binary_search(
    tester: SiliconTester,
    config: TesterConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Conventional Binary Search:
    Halves the full search window [1600..2400 MHz] at each step.
    """
    grid = config.grid_points
    best = binary_search_slice(tester, grid)
    
    assigned_bin = tester.assign_bin(best)
    return {
        "method": "Binary search",
        "measured_fmax": best,
        "steps": tester.steps_executed,
        "time_ms": tester.total_test_time_ms,
        "assigned_bin": assigned_bin,
        "overbin_escape": tester.check_overbin_escape(assigned_bin),
        "underbin": tester.check_underbin(assigned_bin),
        "trajectory": tester.test_history.copy(),
        "edge_hit": False
    }


def run_ml_seeded_walk(
    tester: SiliconTester,
    y_hat: float,
    sigma: float,
    config: TesterConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Naive ML-Seeded Walk (Section 7.2):
    Starts 1.5 sigma below the prediction and steps up one grid point at a time.
    """
    grid = config.grid_points
    start_f = max(grid[0], y_hat - 1.5 * sigma)
    # Snap start_f to grid
    start_idx = int(np.clip(np.round((start_f - grid[0]) / config.grid_step_mhz), 0, len(grid) - 1))
    
    best = None
    for idx in range(start_idx, len(grid)):
        f = grid[idx]
        passed = tester.test(f)
        if passed:
            best = f
        else:
            break
            
    # If the start failed, step downward
    if best is None:
        for idx in range(start_idx - 1, -1, -1):
            f = grid[idx]
            passed = tester.test(f)
            if passed:
                best = f
                break
                
    assigned_bin = tester.assign_bin(best)
    return {
        "method": "ML-seeded walk",
        "measured_fmax": best,
        "steps": tester.steps_executed,
        "time_ms": tester.total_test_time_ms,
        "assigned_bin": assigned_bin,
        "overbin_escape": tester.check_overbin_escape(assigned_bin),
        "underbin": tester.check_underbin(assigned_bin),
        "trajectory": tester.test_history.copy(),
        "edge_hit": False
    }


def run_prediction_windowed_binary(
    tester: SiliconTester,
    y_hat: float,
    L: float,
    U: float,
    config: TesterConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Algorithm 1: Prediction-windowed binary search (Recommended default)
    Runs binary search inside calibrated window [L, U].
    Extends upward dynamically if upper edge hit; searches below if all failed.
    """
    grid = config.grid_points
    step = config.grid_step_mhz
    grid_min = config.grid_min_mhz
    grid_max = config.grid_max_mhz
    
    # Map [L, U] to grid indices
    lo = int(np.clip(np.floor((L - grid_min) / step), 0, len(grid) - 1))
    hi = int(np.clip(np.ceil((U - grid_min) / step), 0, len(grid) - 1))
    
    if lo > hi:
        lo, hi = hi, lo
        
    edge_hit = False
    best = binary_search_slice(tester, grid[lo : hi + 1])
    
    # Edge handling (Section 6, Algorithm 1)
    if best == grid[hi] and hi < len(grid) - 1:
        # Hit upper edge: true Fmax may exceed the predicted window
        edge_hit = True
        stride = 1
        curr_hi = hi
        while curr_hi + stride < len(grid) and tester.test(grid[curr_hi + stride]):
            curr_hi += stride
            stride *= 2
            
        ext_end = min(len(grid) - 1, curr_hi + stride)
        ext_slice = grid[curr_hi : ext_end + 1]
        ext_best = binary_search_slice(tester, ext_slice)
        if ext_best is not None:
            best = ext_best
            
    elif best is None and lo > 0:
        # Everything in window failed: search below window
        edge_hit = True
        best = binary_search_slice(tester, grid[0:lo])
        
    assigned_bin = tester.assign_bin(best)
    return {
        "method": "ML-windowed binary",
        "measured_fmax": best,
        "steps": tester.steps_executed,
        "time_ms": tester.total_test_time_ms,
        "assigned_bin": assigned_bin,
        "overbin_escape": tester.check_overbin_escape(assigned_bin),
        "underbin": tester.check_underbin(assigned_bin),
        "trajectory": tester.test_history.copy(),
        "edge_hit": edge_hit,
        "window_low": grid[lo],
        "window_high": grid[hi]
    }


def run_probabilistic_bisection(
    tester: SiliconTester,
    prior_mean: float,
    prior_sigma: float,
    method_name: str = "ML + Bayesian",
    config: TesterConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Algorithm 2: Probabilistic bisection with a learned (or population) prior.
    Maintains a continuous probability distribution updated via likelihood modeling test noise.
    Returns the conservative 5th percentile of the posterior.
    """
    # Fine grid for posterior representation
    fine_grid = np.linspace(config.grid_min_mhz, config.grid_max_mhz, 401)
    # Initialize prior normal distribution
    p = norm.pdf(fine_grid, loc=prior_mean, scale=max(5.0, prior_sigma))
    p /= np.sum(p)
    
    target_width = config.bayesian_target_ci_width_mhz
    max_steps = config.bayesian_max_steps
    step_mhz = config.grid_step_mhz
    grid_min = config.grid_min_mhz
    
    for _ in range(max_steps):
        # Calculate CDF to find 5th, 50th (median), and 95th percentiles
        cdf = np.cumsum(p)
        idx_5 = int(np.searchsorted(cdf, 0.05))
        idx_50 = int(np.searchsorted(cdf, 0.50))
        idx_95 = int(np.searchsorted(cdf, 0.95))
        
        idx_5 = min(len(fine_grid) - 1, idx_5)
        idx_50 = min(len(fine_grid) - 1, idx_50)
        idx_95 = min(len(fine_grid) - 1, idx_95)
        
        q5 = fine_grid[idx_5]
        q50 = fine_grid[idx_50]
        q95 = fine_grid[idx_95]
        
        if (q95 - q5) <= target_width:
            break
            
        # Snap test frequency to physical tester grid
        f_test = grid_min + np.round((q50 - grid_min) / step_mhz) * step_mhz
        f_test = float(np.clip(f_test, config.grid_min_mhz, config.grid_max_mhz))
        
        outcome = tester.test(f_test)
        
        # Likelihood under test repeatability noise
        sigma_test = config.sigma_test_mhz
        likelihood = norm.cdf((fine_grid - f_test) / sigma_test)
        if not outcome:
            likelihood = 1.0 - likelihood
            
        p = p * likelihood
        p_sum = np.sum(p)
        if p_sum > 0:
            p /= p_sum
        else:
            # Fallback uniform if numerical underflow
            p = np.ones_like(p) / len(p)
            
    # Final conservative estimate: 5th percentile
    cdf = np.cumsum(p)
    idx_conservative = int(np.searchsorted(cdf, config.bayesian_conservative_quantile))
    idx_conservative = min(len(fine_grid) - 1, idx_conservative)
    conservative_fmax = fine_grid[idx_conservative]
    
    assigned_bin = tester.assign_bin(conservative_fmax)
    return {
        "method": method_name,
        "measured_fmax": conservative_fmax,
        "steps": tester.steps_executed,
        "time_ms": tester.total_test_time_ms,
        "assigned_bin": assigned_bin,
        "overbin_escape": tester.check_overbin_escape(assigned_bin),
        "underbin": tester.check_underbin(assigned_bin),
        "trajectory": tester.test_history.copy(),
        "edge_hit": False
    }


def run_confirm_only_binning(
    tester: SiliconTester,
    y_hat: float,
    sigma: float,
    k: float = 1.0,
    config: TesterConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Algorithm 3: Confirm-only binning (Section 6, Algorithm 3).
    Only tests near the predicted bin boundary. Does not measure continuous Fmax.
    """
    bins = list(config.bins_mhz)
    gb = config.measurement_guard_band_mhz
    
    # Identify candidate bin b where bins[b] <= y_hat - k*sigma
    candidate_b = -1
    for i, b_val in enumerate(bins):
        if b_val <= (y_hat - k * sigma):
            candidate_b = i
            
    assigned_bin = "reject"
    if candidate_b >= 0:
        b = candidate_b
        # Test candidate bin edge + guard band
        if tester.test(bins[b] + gb):
            # Pass: check if higher bin is achievable
            while (b + 1 < len(bins)) and (y_hat + 2.0 * sigma > bins[b + 1]) and tester.test(bins[b + 1] + gb):
                b += 1
            assigned_bin = str(int(bins[b]))
        else:
            # Fail: step down
            while b > 0:
                b -= 1
                if tester.test(bins[b] + gb):
                    assigned_bin = str(int(bins[b]))
                    break
    else:
        # Check lowest bin directly
        if tester.test(bins[0] + gb):
            assigned_bin = str(int(bins[0]))
            
    return {
        "method": "ML confirm-bin (binning only)",
        "measured_fmax": float(assigned_bin) if assigned_bin != "reject" else None,
        "steps": tester.steps_executed,
        "time_ms": tester.total_test_time_ms,
        "assigned_bin": assigned_bin,
        "overbin_escape": tester.check_overbin_escape(assigned_bin),
        "underbin": tester.check_underbin(assigned_bin),
        "trajectory": tester.test_history.copy(),
        "edge_hit": False
    }

