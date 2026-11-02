"""
Benchmark Coordinator Module
Executes all conventional and AI-guided search algorithms across test units.
Computes comprehensive statistical comparisons, test time reductions,
and binning quality metrics (Table 6).
"""

from typing import Dict, List, Any, Optional, Tuple 
import numpy as np
import pandas as pd

from src.config import TesterConfig, DEFAULT_CONFIG
from src.simulation.tester_env import SiliconTester
from src.simulation.search_algorithms import (
    run_linear_sweep,
    run_binary_search,
    run_ml_seeded_walk,
    run_prediction_windowed_binary,
    run_probabilistic_bisection,
    run_confirm_only_binning
)


def run_full_benchmark(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    residual_sigma: float,
    pop_mean: float = 2000.0,
    pop_sigma: float = 110.0,
    config: TesterConfig = DEFAULT_CONFIG,
    sample_size: Optional[int] = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs all 7 methods across test units and compiles comparative statistics.
    Returns:
        (summary_df, detailed_unit_results_df)
    """
    rng = np.random.RandomState(seed)
    
    if sample_size is not None and sample_size < len(df_test):
        indices = rng.choice(len(df_test), size=sample_size, replace=False)
        df_sub = df_test.iloc[indices].copy().reset_index(drop=True)
        y_p = y_pred[indices]
        b_low = bounds_low[indices]
        b_high = bounds_high[indices]
    else:
        df_sub = df_test.copy().reset_index(drop=True)
        y_p = y_pred
        b_low = bounds_low
        b_high = bounds_high

    methods = [
        "Linear sweep",
        "Binary search",
        "Bayesian (population prior)",
        "ML-seeded walk",
        "ML-windowed binary",
        "ML + Bayesian",
        "ML confirm-bin (binning only)"
    ]
    
    results = {m: [] for m in methods}
    
    for i in range(len(df_sub)):
        fmax_true = float(df_sub.loc[i, 'fmax_true_mhz'])
        y_hat = float(y_p[i])
        L = float(b_low[i])
        U = float(b_high[i])
        
        # Fresh tester instance per search method with identical seed sequence
        # to ensure fair comparison under identical silicon test noise
        unit_seed = int(rng.randint(0, 1_000_000))
        
        # 1. Linear Sweep
        t_lin = SiliconTester(fmax_true, config, np.random.RandomState(unit_seed))
        res_lin = run_linear_sweep(t_lin, config)
        results["Linear sweep"].append(res_lin)
        
        # 2. Binary Search
        t_bin = SiliconTester(fmax_true, config, np.random.RandomState(unit_seed))
        res_bin = run_binary_search(t_bin, config)
        results["Binary search"].append(res_bin)
        
        # 3. Bayesian (Population Prior)
        t_bpop = SiliconTester(fmax_true, config, np.random.RandomState(unit_seed))
        res_bpop = run_probabilistic_bisection(
            t_bpop, pop_mean, pop_sigma, method_name="Bayesian (population prior)", config=config
        )
        results["Bayesian (population prior)"].append(res_bpop)
        
        # 4. ML-Seeded Walk
        t_seed = SiliconTester(fmax_true, config, np.random.RandomState(unit_seed))
        res_seed = run_ml_seeded_walk(t_seed, y_hat, residual_sigma, config)
        results["ML-seeded walk"].append(res_seed)
        
        # 5. ML-Windowed Binary (Algorithm 1)
        t_win = SiliconTester(fmax_true, config, np.random.RandomState(unit_seed))
        res_win = run_prediction_windowed_binary(t_win, y_hat, L, U, config)
        results["ML-windowed binary"].append(res_win)
        
        # 6. ML + Bayesian
        t_mlb = SiliconTester(fmax_true, config, np.random.RandomState(unit_seed))
        res_mlb = run_probabilistic_bisection(
            t_mlb, y_hat, residual_sigma, method_name="ML + Bayesian", config=config
        )
        results["ML + Bayesian"].append(res_mlb)
        
        # 7. ML Confirm-Only Binning
        t_conf = SiliconTester(fmax_true, config, np.random.RandomState(unit_seed))
        res_conf = run_confirm_only_binning(t_conf, y_hat, residual_sigma, config=config)
        results["ML confirm-bin (binning only)"].append(res_conf)

    # Compute comparative summary table (Table 6 format)
    summary_rows = []
    baseline_bin_steps = np.mean([r["steps"] for r in results["Binary search"]])
    baseline_lin_steps = np.mean([r["steps"] for r in results["Linear sweep"]])
    
    for m in methods:
        m_res = results[m]
        steps = [r["steps"] for r in m_res]
        times = [r["time_ms"] for r in m_res]
        overbins = [r["overbin_escape"] for r in m_res]
        underbins = [r["underbin"] for r in m_res]
        
        # Error calculation (where continuous Fmax is measured)
        errors = []
        for i, r in enumerate(m_res):
            if r["measured_fmax"] is not None:
                errors.append(abs(r["measured_fmax"] - float(df_sub.loc[i, 'fmax_true_mhz'])))
                
        mean_steps = float(np.mean(steps))
        p95_steps = float(np.percentile(steps, 95))
        mean_time = float(np.mean(times))
        
        save_vs_bin = ((baseline_bin_steps - mean_steps) / baseline_bin_steps) * 100.0
        save_vs_lin = ((baseline_lin_steps - mean_steps) / baseline_lin_steps) * 100.0
        
        mean_err = float(np.mean(errors)) if errors else np.nan
        underbin_pct = float(np.mean(underbins) * 100.0)
        overbin_count = sum(overbins)
        overbin_ppm = float((overbin_count / len(m_res)) * 1_000_000)
        
        summary_rows.append({
            "Method": m,
            "Mean Tests": round(mean_steps, 2),
            "p95 Tests": int(round(p95_steps)),
            "Time (ms)": round(mean_time, 1),
            "Saving vs Binary": f"{save_vs_bin:+.1f}%",
            "Saving vs Linear": f"{save_vs_lin:+.1f}%",
            "|Error| (MHz)": round(mean_err, 1) if not np.isnan(mean_err) else "—",
            "Under-bin (%)": f"{underbin_pct:.1f}%",
            "Over-bin (ppm)": int(round(overbin_ppm))
        })
        
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, results

