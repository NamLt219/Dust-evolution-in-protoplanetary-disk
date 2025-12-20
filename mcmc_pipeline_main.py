"""
Main MCMC Pipeline Runner
==========================
Orchestrate toàn bộ MCMC fitting pipeline:
1. Load observed data
2. Setup forward simulator
3. Initialize MCMC sampler  
4. Run MCMC
5. Analyze results
6. Generate plots

Author: Pipeline Builder
Date: 2025-11-19
"""

import numpy as np
import os
import sys
from pathlib import Path
import time
from astropy.io import fits
import matplotlib.pyplot as plt

# Import pipeline components
from mcmc_pipeline_config import *
from mcmc_logger import setup_logger, get_logger
from forward_simulator import ForwardModelSimulatorV2 as ForwardModelSimulator
from likelihood_calculator import (
    LikelihoodCalculator, 
    PriorEvaluator, 
    MCMCProbability
)
from mcmc_sampler import MCMCSampler


class MCMCPipeline:
    """
    Main pipeline orchestrator.
    """
    
    def __init__(self, config_override: dict = None):
        """
        Initialize pipeline.
        
        Parameters:
        -----------
        config_override : dict, optional
            Override default config values
        """
        # Setup logger first
        self.logger = setup_logger(
            log_dir=LOG_DIR,
            console_level=LOG_LEVEL_CONSOLE,
            file_level=LOG_LEVEL_FILE
        )
        
        self.logger.info("="*80)
        self.logger.info("MCMC PIPELINE INITIALIZATION")
        self.logger.info("="*80)
        
        # Apply config overrides
        if config_override:
            self.logger.info(f"Applying config overrides: {config_override}")
            # Apply overrides to global config
            for key, value in config_override.items():
                if hasattr(sys.modules['mcmc_pipeline_config'], key):
                    setattr(sys.modules['mcmc_pipeline_config'], key, value)
                    self.logger.info(f"  Set {key} = {value}")
        
        # Print config summary
        print_config_summary()
        
        # Components (will be initialized)
        self.forward_simulator = None
        self.likelihood_calc = None
        self.prior_eval = None
        self.mcmc_prob = None
        self.sampler = None
        
        # Data
        self.obs_image = None
        self.obs_uncertainty = None
        
        # Results
        self.chain = None
        self.log_prob = None
        self.best_params = None
    
    def load_observation(self, fits_path: str = OBS_FITS_PATH):
        """
        Load observed FITS file.
        
        Parameters:
        -----------
        fits_path : str
            Path to FITS file
        """
        self.logger.info(f"Loading observation from: {fits_path}")
        
        if not os.path.exists(fits_path):
            raise FileNotFoundError(f"FITS file not found: {fits_path}")
        
        # Load FITS
        with fits.open(fits_path) as hdul:
            # Typically image is in primary HDU or first extension
            if len(hdul) > 1 and hdul[1].data is not None:
                self.obs_image = hdul[1].data
            else:
                self.obs_image = hdul[0].data
            
            # Get header info for beam
            header = hdul[0].header
            
            # Try to extract beam info from header
            if 'BMAJ' in header and 'BMIN' in header:
                bmaj = header['BMAJ'] * 3600  # degrees to arcsec
                bmin = header['BMIN'] * 3600
                bpa = header.get('BPA', 0.0)
                
                self.logger.info(f"Beam from FITS header: "
                               f"{bmaj:.3f}\" x {bmin:.3f}\", PA={bpa:.1f}°")
        
        # Handle data dimensions
        # FITS can have extra dimensions (frequency, Stokes)
        while self.obs_image.ndim > 2:
            self.obs_image = self.obs_image[0]
        
        # Keep observation in Jy/beam (CASA standard, matches paper)
        # Model will also output Jy/beam after beam convolution
        # Setup uncertainty map (constant RMS for now)
        self.obs_uncertainty = np.ones_like(self.obs_image) * RMS_NOISE_JY
        
        self.logger.info(f"Observation loaded: shape={self.obs_image.shape}, "
                        f"range=[{np.nanmin(self.obs_image):.2e}, {np.nanmax(self.obs_image):.2e}]")
        
        return self.obs_image
    
    def setup_forward_simulator(self):
        """Setup forward model simulator."""
        self.logger.info("Setting up forward simulator...")
        
        # ForwardModelSimulatorV2 uses config from mcmc_pipeline_config.py
        # No need to pass work_dir, radmc3d_exec - reads from global config
        self.forward_simulator = ForwardModelSimulator(
            config_dict=None,  # Use defaults from config
            cleanup=CLEANUP_TEMP_FILES
        )
        
        self.logger.info("Forward simulator ready")
    
    def setup_likelihood(self):
        """Setup likelihood calculator."""
        if self.obs_image is None:
            raise RuntimeError("Must load observation first")
        
        self.logger.info("Setting up likelihood calculator...")
        
        self.likelihood_calc = LikelihoodCalculator(
            obs_image=self.obs_image,
            obs_uncertainty=self.obs_uncertainty,
            rms_noise=RMS_NOISE_JY,
            mask_threshold=MASK_THRESHOLD_SIGMA,
            use_mask=USE_WEIGHTS
        )
        
        self.logger.info(f"Likelihood calculator ready: "
                        f"{self.likelihood_calc.n_data_points} data points")
    
    def setup_prior(self):
        """Setup prior evaluator."""
        self.logger.info("Setting up prior evaluator...")
        
        self.prior_eval = PriorEvaluator(param_config=MCMC_PARAMETERS)
        
        self.logger.info(f"Prior evaluator ready: {len(MCMC_PARAMETERS)} parameters")
    
    def setup_mcmc_probability(self):
        """Setup combined MCMC probability."""
        if self.forward_simulator is None:
            self.setup_forward_simulator()
        if self.likelihood_calc is None:
            self.setup_likelihood()
        if self.prior_eval is None:
            self.setup_prior()
        
        self.logger.info("Setting up MCMC probability function...")
        
        # Get fixed params from config
        fixed_params = getattr(sys.modules['mcmc_pipeline_config'], 'FIXED_PARAMS', {})
        
        self.mcmc_prob = MCMCProbability(
            likelihood_calc=self.likelihood_calc,
            prior_eval=self.prior_eval,
            forward_simulator=self.forward_simulator,
            param_names=PARAM_NAMES,
            fixed_params=fixed_params
        )
        
        self.logger.info("MCMC probability function ready")
    
    def run_mcmc(self, 
                n_steps: int = N_STEPS_TOTAL,
                n_burn_in: int = N_STEPS_BURNIN,
                resume_from: str = None):
        """
        Run MCMC sampling.
        
        Parameters:
        -----------
        n_steps : int
            Total number of steps
        n_burn_in : int
            Burn-in steps
        resume_from : str, optional
            Path to checkpoint to resume from
        """
        if self.mcmc_prob is None:
            self.setup_mcmc_probability()
        
        self.logger.info("="*80)
        self.logger.info("STARTING MCMC SAMPLING")
        self.logger.info("="*80)
        
        # Initialize sampler
        self.sampler = MCMCSampler(
            log_prob_fn=self.mcmc_prob,
            n_params=N_PARAMS,
            n_walkers=N_WALKERS,
            param_names=PARAM_NAMES,
            checkpoint_dir=CHECKPOINT_DIR,
            use_parallel=USE_MULTIPROCESSING,
            n_processes=N_PROCESSES
        )
        
        # Resume or start fresh - check if HDF5 backend exists
        backend_path = os.path.join(CHECKPOINT_DIR, "mcmc_chain.h5")
        backend_exists = os.path.exists(backend_path)
        
        if resume_from or backend_exists:
            if backend_exists:
                # Check backend iterations
                try:
                    import emcee
                    backend = emcee.backends.HDFBackend(backend_path, read_only=True)
                    saved_steps = backend.iteration
                    if saved_steps > 0:
                        self.logger.info(f"🔄 Found existing backend with {saved_steps} steps")
                        self.logger.info(f"   Will resume from step {saved_steps + 1}")
                        resume_mode = True
                    else:
                        self.logger.info("Backend exists but empty - starting fresh")
                        resume_mode = False
                except:
                    self.logger.info("Backend file exists but cannot read - starting fresh")
                    resume_mode = False
            else:
                self.logger.info(f"Resuming from checkpoint: {resume_from}")
                resume_mode = True
        else:
            # Generate initial positions
            self.logger.info("Generating initial walker positions...")
            resume_mode = False
        
        initial_pos = get_initial_positions(n_walkers=N_WALKERS)
        
        # Log phase transition to burn-in
        if n_burn_in > 0:
            self.logger.log_phase_transition("burn-in", 0)
        
        # Run MCMC with HDF5 Backend (burn-in handled later with discard)
        self.chain, self.log_prob = self.sampler.run(
            initial_positions=initial_pos,
            n_steps=n_steps,
            resume=resume_mode,
            show_progress=SHOW_PROGRESS
        )
        
        # Get best fit (HDF5 version stores in sampler attributes)
        self.best_params = self.sampler.best_params
        best_log_prob = self.sampler.best_log_prob
        
        self.logger.info("="*80)
        self.logger.info("MCMC SAMPLING COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"Best fit parameters:")
        for name, val in zip(PARAM_NAMES, self.best_params):
            self.logger.info(f"  {name:20s} = {val:.6f}")
        self.logger.info(f"  Log-likelihood = {best_log_prob:.4f}")
    
    def save_results(self):
        """Save all results."""
        self.logger.info("Saving results...")
        
        # Save chain and log-prob
        self.sampler.save_results(MCMC_OUTPUT_DIR)
        
        # Save best-fit model
        self.logger.info("Computing best-fit model...")
        best_params_dict = params_to_dict(self.best_params)
        success, best_image, metadata = self.forward_simulator.simulate(best_params_dict)
        
        if success:
            # Save as FITS
            os.makedirs(BEST_FIT_DIR, exist_ok=True)
            hdu = fits.PrimaryHDU(best_image)
            hdu.writeto(BEST_FIT_IMAGE_FILE, overwrite=True)
            self.logger.info(f"Best-fit image saved: {BEST_FIT_IMAGE_FILE}")
        else:
            self.logger.error("Failed to compute best-fit model")
        
        self.logger.info("Results saved successfully")
    
    def analyze_results(self):
        """Analyze MCMC results."""
        self.logger.info("Analyzing results...")
        
        # Get samples (discard burn-in)
        samples, log_prob_samples = self.sampler.get_samples(
            discard=N_STEPS_BURNIN,
            thin=THIN_BY,
            flat=True
        )
        
        # Compute statistics
        self.logger.info("Parameter statistics:")
        for i, name in enumerate(PARAM_NAMES):
            median = np.median(samples[:, i])
            std = np.std(samples[:, i])
            q16, q84 = np.percentile(samples[:, i], [16, 84])
            self.logger.info(f"  {name:20s}: {median:.6f} ± {std:.6f} "
                           f"[{q16:.6f}, {q84:.6f}]")
        
        # Convergence diagnostics
        tau = self.sampler.compute_autocorr_time()
        if tau is not None:
            self.logger.info("Autocorrelation times:")
            for name, t in zip(PARAM_NAMES, tau):
                self.logger.info(f"  {name:20s}: τ = {t:.2f}")
        
        r_hat = self.sampler.compute_gelman_rubin(discard=N_STEPS_BURNIN)
        if r_hat is not None:
            self.logger.info("Gelman-Rubin R-hat:")
            for name, r in zip(PARAM_NAMES, r_hat):
                status = "✓" if r < 1.1 else "✗"
                self.logger.info(f"  {name:20s}: R-hat = {r:.4f} {status}")
    
    def plot_results(self):
        """Generate diagnostic plots."""
        self.logger.info("Generating plots...")
        
        try:
            import corner
            
            # Get samples
            samples, _ = self.sampler.get_samples(
                discard=N_STEPS_BURNIN,
                thin=THIN_BY,
                flat=True
            )

            # Corner plot
            fig = corner.corner(
                samples,
                labels=PARAM_LABELS,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12}
            )
            fig.savefig(CORNER_PLOT_FILE, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Corner plot saved: {CORNER_PLOT_FILE}")
            
        except ImportError:
            self.logger.warning("corner package not installed, skipping corner plot")
        
        # Trace plot
        fig, axes = plt.subplots(N_PARAMS, 1, figsize=(10, 2*N_PARAMS))
        if N_PARAMS == 1:
            axes = [axes]
        
        for i in range(N_PARAMS):
            for walker in range(min(10, N_WALKERS)):  # Plot first 10 walkers
                axes[i].plot(self.chain[walker, :, i], alpha=0.3, lw=0.5)
            axes[i].axvline(N_STEPS_BURNIN, color='r', ls='--', lw=1, label='End of burn-in')
            axes[i].set_ylabel(PARAM_LABELS[i])
            if i == 0:
                axes[i].legend()
        axes[-1].set_xlabel('Step')
        fig.tight_layout()
        fig.savefig(TRACE_PLOT_FILE, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Trace plot saved: {TRACE_PLOT_FILE}")
    
    def run_full_pipeline(self):
        """Run complete pipeline from start to finish."""
        try:
            # Load data
            self.load_observation()
            
            # Setup components
            self.setup_forward_simulator()
            self.setup_likelihood()
            self.setup_prior()
            self.setup_mcmc_probability()
            
            # Run MCMC
            self.run_mcmc()
            
            # Analyze and save
            self.analyze_results()
            self.save_results()
            self.plot_results()
            
            self.logger.info("="*80)
            self.logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.critical("Pipeline failed", exception=e)
            raise
        finally:
            self.logger.close()


def main():
    """Main entry point."""
    pipeline = MCMCPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
