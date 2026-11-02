"""
Silicon Tester Environment Module
Simulates Automatic Test Equipment (ATE) hardware, pass/fail evaluation
with Gaussian test noise, execution timing, and speed bin assignment.
"""

from typing import Tuple, List, Optional
import numpy as np
from scipy.stats import norm
from src.config import TesterConfig, DEFAULT_CONFIG


class SiliconTester:
    """
    Virtual ATE Tester environment for an individual silicon die.
    Evaluates pass/fail outcome at tested frequency f based on true Fmax and noise sigma.
    """
    def __init__(
        self,
        fmax_true_mhz: float,
        config: TesterConfig = DEFAULT_CONFIG,
        rng: Optional[np.random.RandomState] = None
    ):
        self.fmax_true = fmax_true_mhz
        self.config = config
        self.rng = rng if rng is not None else np.random.RandomState()
        
        # Test telemetry tracking
        self.test_history: List[Tuple[float, bool]] = []
        self.steps_executed: int = 0

    def test(self, frequency_mhz: float) -> bool:
        """
        Executes a test pattern burst at frequency_mhz.
        Pass probability: P(pass | f, Fmax) = Phi((Fmax - f) / sigma_test)
        """
        self.steps_executed += 1
        
        # If noise is zero, deterministic step function
        if self.config.sigma_test_mhz <= 0:
            outcome = frequency_mhz <= self.fmax_true
        else:
            # Gaussian pass/fail boundary noise (Section 3.3)
            prob_pass = float(norm.cdf((self.fmax_true - frequency_mhz) / self.config.sigma_test_mhz))
            rand_val = self.rng.uniform(0.0, 1.0)
            outcome = rand_val < prob_pass
            
        self.test_history.append((frequency_mhz, outcome))
        return outcome

    @property
    def total_test_time_ms(self) -> float:
        """Calculates total elapsed tester time in milliseconds."""
        return self.steps_executed * self.config.total_step_time_ms

    def assign_bin(self, measured_fmax_mhz: Optional[float]) -> str:
        """
        Assigns speed bin according to the measurement guard band rule:
        A die must pass B + guard_band to be assigned to bin B.
        """
        if measured_fmax_mhz is None:
            return "reject"
            
        gb = self.config.measurement_guard_band_mhz
        for b in reversed(self.config.bins_mhz):
            if measured_fmax_mhz >= (b + gb):
                return str(int(b))
                
        return "reject"

    def check_overbin_escape(self, assigned_bin_str: str) -> bool:
        """
        Checks for an over-bin escape: A critical quality failure where a part
        is assigned to a bin faster than its true physical Fmax supports.
        """
        if assigned_bin_str == "reject":
            return False
        assigned_bin_freq = float(assigned_bin_str)
        # Part cannot run at its rated bin frequency
        return self.fmax_true < assigned_bin_freq

    def check_underbin(self, assigned_bin_str: str) -> bool:
        """
        Checks for under-binning: Revenue loss where a part physically qualified
        for a higher bin, but was placed in a lower bin.
        """
        # Highest bin the true physical Fmax could have satisfied with guard band
        gb = self.config.measurement_guard_band_mhz
        achievable_bin = "reject"
        for b in reversed(self.config.bins_mhz):
            if self.fmax_true >= (b + gb):
                achievable_bin = str(int(b))
                break
                
        if achievable_bin == "reject":
            return False
            
        if assigned_bin_str == "reject":
            return True
            
        return float(assigned_bin_str) < float(achievable_bin)

