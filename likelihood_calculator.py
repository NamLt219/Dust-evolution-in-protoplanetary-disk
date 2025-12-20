"""
Likelihood Calculator & Chi-squared Computation
================================================
Tính toán likelihood cho MCMC dựa trên residuals giữa observed và model images.

Features:
- Chi-squared calculation với proper weighting
- Masking low SNR regions
- Prior probability evaluation
- Log-likelihood computation
- Multiple likelihood forms (chi2, Gaussian, etc.)

Author: Pipeline Builder  
Date: 2025-11-19
"""

import numpy as np
from typing import Dict, Optional, Tuple, Callable
from scipy import stats

try:
    from mcmc_pipeline_config import *
    from mcmc_logger import get_logger
except ImportError:
    print("WARNING: Could not import config")
    RMS_NOISE_JY = 1e-4
    MASK_THRESHOLD_SIGMA = 3.0


class LikelihoodCalculator:
    """
    Calculate log-likelihood for MCMC sampling.
    """
    
    def __init__(self,
                 obs_image: np.ndarray,
                 obs_uncertainty: Optional[np.ndarray] = None,
                 rms_noise: float = RMS_NOISE_JY,
                 mask_threshold: float = MASK_THRESHOLD_SIGMA,
                 use_mask: bool = True):
        """
        Initialize likelihood calculator.
        
        Parameters:
        -----------
        obs_image : ndarray
            Observed image
        obs_uncertainty : ndarray, optional
            Uncertainty map (if None, use constant RMS)
        rms_noise : float
            RMS noise level
        mask_threshold : float
            Mask pixels below N*sigma
        use_mask : bool
            Apply masking to low SNR regions
        """
        self.obs_image = obs_image
        self.rms_noise = rms_noise
        
        # Validate image is 2D
        if obs_image.ndim != 2:
            raise ValueError(f"Observation image must be 2D, got {obs_image.ndim}D: {obs_image.shape}")
        
        # Setup uncertainty map
        if obs_uncertainty is not None:
            self.uncertainty = obs_uncertainty
        else:
            self.uncertainty = np.ones_like(obs_image) * rms_noise
        
        # Create mask (only fit high SNR regions)
        if use_mask:
            self.mask = self.obs_image > (mask_threshold * rms_noise)
            n_pixels = np.sum(self.mask)
        else:
            self.mask = np.ones_like(obs_image, dtype=bool)
            n_pixels = obs_image.size
        
        self.n_data_points = n_pixels
        
        try:
            self.logger = get_logger()
            self.logger.info(f"LikelihoodCalculator initialized: "
                           f"{n_pixels} valid pixels out of {obs_image.size} total")
        except:
            self.logger = None
            print(f"[INFO] LikelihoodCalculator: {n_pixels} valid pixels")
    
    def log_likelihood(self, 
                      model_image: np.ndarray,
                      verbose: bool = False) -> float:
        """
        Compute log-likelihood (Gaussian likelihood = -0.5 * chi2).
        
        Parameters:
        -----------
        model_image : ndarray
            Model image
        verbose : bool
            Print debug info
        
        Returns:
        --------
        log_likelihood : float
            Log-likelihood value
        """
        # Validate shape match
        if model_image.shape != self.obs_image.shape:
            raise ValueError(
                f"Model shape {model_image.shape} doesn't match "
                f"observation shape {self.obs_image.shape}"
            )
        
        # Compute residuals
        residuals = self.obs_image - model_image
        
        # Apply mask
        residuals_masked = residuals[self.mask]
        uncertainty_masked = self.uncertainty[self.mask]
        
        # Chi-squared
        chi2_terms = (residuals_masked / uncertainty_masked) ** 2
        chi2 = np.sum(chi2_terms)
        
        # Log-likelihood (Gaussian)
        # ln(L) = -0.5 * [chi2 + N*ln(2*pi) + sum(ln(sigma^2))]
        log_like = -0.5 * (
            chi2 + 
            self.n_data_points * np.log(2 * np.pi) + 
            np.sum(np.log(uncertainty_masked**2))
        )
        
        if verbose:
            print(f"  Chi2: {chi2:.2f}")
            print(f"  Reduced chi2: {chi2 / self.n_data_points:.2f}")
            print(f"  Log-likelihood: {log_like:.2f}")
        
        return log_like
    
    def chi_squared(self, model_image: np.ndarray) -> float:
        """
        Compute chi-squared statistic.
        
        Parameters:
        -----------
        model_image : ndarray
            Model image
        
        Returns:
        --------
        chi2 : float
            Chi-squared value
        """
        residuals = self.obs_image - model_image
        residuals_masked = residuals[self.mask]
        uncertainty_masked = self.uncertainty[self.mask]
        
        chi2 = np.sum((residuals_masked / uncertainty_masked) ** 2)
        
        return chi2
    
    def reduced_chi_squared(self, 
                           model_image: np.ndarray,
                           n_params: int) -> float:
        """
        Compute reduced chi-squared.
        
        Parameters:
        -----------
        model_image : ndarray
            Model image
        n_params : int
            Number of fitted parameters
        
        Returns:
        --------
        chi2_red : float
            Reduced chi-squared
        """
        chi2 = self.chi_squared(model_image)
        dof = self.n_data_points - n_params
        
        return chi2 / dof if dof > 0 else np.inf
    
    def residuals(self, model_image: np.ndarray) -> np.ndarray:
        """Get residuals (obs - model)."""
        return self.obs_image - model_image
    
    def normalized_residuals(self, model_image: np.ndarray) -> np.ndarray:
        """Get normalized residuals ((obs - model) / uncertainty)."""
        return (self.obs_image - model_image) / self.uncertainty
    
    def __getstate__(self):
        """Prepare object for pickling (multiprocessing)."""
        state = self.__dict__.copy()
        state['logger'] = None  # Remove unpicklable logger
        return state
    
    def __setstate__(self, state):
        """Restore object after unpickling."""
        self.__dict__.update(state)
        try:
            self.logger = get_logger()
        except:
            self.logger = None


class PriorEvaluator:
    """
    Evaluate prior probabilities for parameters.
    """
    
    def __init__(self, param_config: list):
        """
        Initialize prior evaluator.
        
        Parameters:
        -----------
        param_config : list of dict
            Parameter configuration from config file
        """
        self.param_config = param_config
        self.param_names = [p["name"] for p in param_config]
        self.bounds = {p["name"]: (p["min"], p["max"]) for p in param_config}
        
        try:
            self.logger = get_logger()
        except:
            self.logger = None
    
    def log_prior(self, params: Dict[str, float]) -> float:
        """
        Compute log-prior probability.
        
        Parameters:
        -----------
        params : dict
            Parameter values
        
        Returns:
        --------
        log_prior : float
            Log-prior probability (0 if within bounds, -inf if outside)
        """
        # Check bounds (uniform prior)
        for name, value in params.items():
            if name in self.bounds:
                pmin, pmax = self.bounds[name]
                if not (pmin <= value <= pmax):
                    return -np.inf  # Outside bounds
        
        # All parameters within bounds - uniform prior
        return 0.0
    
    def log_prior_gaussian(self, 
                          params: Dict[str, float],
                          means: Dict[str, float],
                          stds: Dict[str, float]) -> float:
        """
        Gaussian prior (optional).
        
        Parameters:
        -----------
        params : dict
            Parameter values
        means : dict
            Prior means
        stds : dict
            Prior standard deviations
        
        Returns:
        --------
        log_prior : float
            Log-prior probability
        """
        # First check bounds
        if not np.isfinite(self.log_prior(params)):
            return -np.inf
        
        # Gaussian prior
        log_prior = 0.0
        for name, value in params.items():
            if name in means:
                mean = means[name]
                std = stds[name]
                log_prior += stats.norm.logpdf(value, loc=mean, scale=std)
        
        return log_prior
    
    def __getstate__(self):
        """Prepare object for pickling (multiprocessing)."""
        state = self.__dict__.copy()
        state['logger'] = None  # Remove unpicklable logger
        return state
    
    def __setstate__(self, state):
        """Restore object after unpickling."""
        self.__dict__.update(state)
        try:
            self.logger = get_logger()
        except:
            self.logger = None


class MCMCProbability:
    """
    Combined prior + likelihood for MCMC.
    """
    
    def __init__(self,
                 likelihood_calc: LikelihoodCalculator,
                 prior_eval: PriorEvaluator,
                 forward_simulator,
                 param_names: list,
                 fixed_params: Optional[Dict[str, float]] = None):
        """
        Initialize MCMC probability calculator.
        
        Parameters:
        -----------
        likelihood_calc : LikelihoodCalculator
            Likelihood calculator
        prior_eval : PriorEvaluator
            Prior evaluator
        forward_simulator : ForwardModelSimulator
            Forward model simulator
        param_names : list
            Parameter names in order (only fitted parameters)
        fixed_params : dict, optional
            Fixed parameters not being fitted
        """
        self.likelihood_calc = likelihood_calc
        self.prior_eval = prior_eval
        self.forward_simulator = forward_simulator
        self.param_names = param_names
        self.fixed_params = fixed_params or {}
        
        # Call counter for diagnostics
        self.n_calls = 0
        self.n_accepts = 0
        self.n_rejects = 0
        
        try:
            self.logger = get_logger()
            if self.fixed_params:
                self.logger.info(f"Fixed parameters: {self.fixed_params}")
        except:
            self.logger = None
    
    def __call__(self, theta: np.ndarray) -> float:
        """
        Compute log-probability for MCMC.
        
        This is the function called by emcee sampler.
        
        Parameters:
        -----------
        theta : ndarray
            Parameter vector
        
        Returns:
        --------
        log_prob : float
            Log-probability (log-prior + log-likelihood)
        """
        self.n_calls += 1
        
        # Convert to dict (only fitted parameters)
        params = {name: val for name, val in zip(self.param_names, theta)}
        
        # Add fixed parameters
        params.update(self.fixed_params)
        
        # Check prior (only on fitted parameters)
        fitted_params = {name: val for name, val in zip(self.param_names, theta)}
        log_prior = self.prior_eval.log_prior(fitted_params)
        
        if not np.isfinite(log_prior):
            self.n_rejects += 1
            return -np.inf
        
        # Run forward model
        try:
            success, model_image, metadata = self.forward_simulator.simulate(params)
            
            if not success:
                self.n_rejects += 1
                return -np.inf
            
            # Compute likelihood
            log_like = self.likelihood_calc.log_likelihood(model_image)
            
            # Total probability
            log_prob = log_prior + log_like
            
            if np.isfinite(log_prob):
                self.n_accepts += 1
            else:
                self.n_rejects += 1
            
            return log_prob
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in probability calculation", exception=e)
            self.n_rejects += 1
            return -np.inf
    
    def get_acceptance_rate(self) -> float:
        """Get current acceptance rate."""
        if self.n_calls == 0:
            return 0.0
        return self.n_accepts / self.n_calls
    
    def reset_counters(self):
        """Reset call counters."""
        self.n_calls = 0
        self.n_accepts = 0
        self.n_rejects = 0
    
    def __getstate__(self):
        """
        Prepare object for pickling (multiprocessing).
        Remove unpicklable logger objects before serialization.
        """
        state = self.__dict__.copy()
        # Remove logger (has thread locks - not picklable)
        state['logger'] = None
        # Also remove logger from nested objects if they exist
        if hasattr(self.likelihood_calc, 'logger'):
            # Don't modify original, create shallow copy
            import copy
            state['likelihood_calc'] = copy.copy(self.likelihood_calc)
            state['likelihood_calc'].logger = None
        if hasattr(self.forward_simulator, 'logger'):
            import copy
            state['forward_simulator'] = copy.copy(self.forward_simulator)
            state['forward_simulator'].logger = None
        return state
    
    def __setstate__(self, state):
        """
        Restore object after unpickling.
        Recreate logger in the new process.
        """
        self.__dict__.update(state)
        # Recreate logger in the new process
        try:
            self.logger = get_logger()
        except:
            self.logger = None
        # Restore loggers in nested objects
        if hasattr(self.likelihood_calc, 'logger') and self.likelihood_calc.logger is None:
            try:
                self.likelihood_calc.logger = get_logger()
            except:
                pass
        if hasattr(self.forward_simulator, 'logger') and self.forward_simulator.logger is None:
            try:
                self.forward_simulator.logger = get_logger()
            except:
                pass


if __name__ == "__main__":
    # Test likelihood calculator
    print("Testing LikelihoodCalculator...")
    
    # Create fake observation
    np.random.seed(42)
    obs = np.random.randn(100, 100) * 0.001 + 0.01
    
    # Create fake model
    model = obs + np.random.randn(100, 100) * 0.0005
    
    # Test likelihood
    calc = LikelihoodCalculator(
        obs_image=obs,
        rms_noise=0.001,
        use_mask=False
    )
    
    log_like = calc.log_likelihood(model, verbose=True)
    chi2 = calc.chi_squared(model)
    
    print(f"\nTest results:")
    print(f"  Log-likelihood: {log_like:.2f}")
    print(f"  Chi-squared: {chi2:.2f}")
    print(f"  Reduced chi-squared: {chi2 / obs.size:.4f}")
