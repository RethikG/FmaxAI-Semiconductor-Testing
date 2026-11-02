"""
FmaxAI Configuration Module
Defines physical tester specifications, frequency grid, timing constants,
speed bins, and quality parameters based on the Technical Document.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TesterConfig:
    __test__ = False  # Prevent pytest from collecting as test class
    # Frequency search grid
    grid_min_mhz: float = 1600.0
    grid_max_mhz: float = 2400.0
    grid_step_mhz: float = 10.0
    
    # Tester hardware timing (illustrative values from Table 5)
    t_switch_ms: float = 5.0    # Clock divider/PLL settling time
    t_apply_ms: float = 25.0    # Pattern application or workload execution time
    
    # Test repeatability noise (Gaussian noise standard deviation on pass/fail boundary)
    sigma_test_mhz: float = 4.0
    
    # Speed binning configuration
    bins_mhz: tuple = (1800.0, 1900.0, 2000.0, 2100.0, 2200.0)
    measurement_guard_band_mhz: float = 8.0
    
    # AI-guided windowing parameter (in units of calibration residual sigma)
    default_window_k: float = 1.5
    
    # Probabilistic (Bayesian) adaptive search parameters
    bayesian_target_ci_width_mhz: float = 15.0
    bayesian_ci_percentiles: tuple = (0.05, 0.95)
    bayesian_max_steps: int = 15
    bayesian_conservative_quantile: float = 0.05
    
    # Cost & production parameters (Section 9.3)
    default_cost_per_second: float = 0.004  # $0.004 per unit-second
    default_annual_units: int = 20_000_000  # 20M units/year
    default_corners: int = 3                # DVFS voltage corners

    @property
    def total_step_time_ms(self) -> float:
        return self.t_switch_ms + self.t_apply_ms

    @property
    def grid_points(self) -> List[float]:
        count = int(round((self.grid_max_mhz - self.grid_min_mhz) / self.grid_step_mhz)) + 1
        return [self.grid_min_mhz + i * self.grid_step_mhz for i in range(count)]


# Global default configuration instance
DEFAULT_CONFIG = TesterConfig()

# Legitimate input features available prior to Fmax testing
PRE_TEST_FEATURES = [
    'ro_01', 'ro_02', 'ro_03', 'ro_04', 'ro_05',
    'ro_06', 'ro_07', 'ro_08', 'ro_09', 'ro_10',
    'iddq_norm', 'vmin_norm',
    'die_x', 'die_y', 'radius_sq'
]

# Post-search simulation/leakage columns that MUST NEVER be used as model input features
DOWNSTREAM_LEAKAGE_COLUMNS = [
    'fmax_true_mhz',
    'fmax_pred_mhz',
    'window_low_mhz',
    'window_high_mhz',
    'true_bin_mhz',
    'tests_binary',
    'tests_windowed',
    'fmax_measured_windowed_mhz'
]
