import numpy as np
import emcee
import time
import os
import sys
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict
import json
from tqdm import tqdm
import multiprocessing as mp


try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

try:
    import mcmc_pipeline_config as config
    from mcmc_logger import get_logger
except ImportError:
    print("WARNING: Config not imported")
    class config:
        N_WALKERS = 32
        N_STEPS_TOTAL = 1000
        CHECKPOINT_INTERVAL = 50


class MCMCSampler:

    
    def __init__(self,
                 log_prob_fn: Callable,
                 n_params: int,
                 n_walkers: int = None,
                 param_names: Optional[list] = None,
                 checkpoint_dir: str = "./checkpoints",
                 use_parallel: bool = False,
                 n_processes: Optional[int] = None,
                 backend_filename: Optional[str] = None):

        self.log_prob_fn = log_prob_fn
        self.n_params = n_params
        self.n_walkers = n_walkers if n_walkers is not None else config.N_WALKERS
        self.param_names = param_names or [f"param_{i}" for i in range(n_params)]
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # HDF5 Backend setup
        self.backend_filename = backend_filename or "mcmc_chain.h5"
        self.backend_path = self.checkpoint_dir / self.backend_filename
        self.backend = None
        
        # Parallel processing setup
        self.use_parallel = use_parallel
        self.n_processes = n_processes or mp.cpu_count()
        self.pool = None
        
        # Sampler will be initialized when run() is called
        self.sampler = None
        
        # Best fit tracking (lightweight - only current best)
        self.best_log_prob = -np.inf
        self.best_params = None
        self.best_iteration = 0
        
        try:
            self.logger = get_logger()
            self.logger.info(f"MCMCSampler initialized with HDF5 Backend")
            self.logger.info(f"  • Backend file: {self.backend_path}")
            self.logger.info(f"  • {n_params} params, {n_walkers} walkers")
            self.logger.info(f"  • Parallel: {use_parallel} ({n_processes} processes)")
        except:
            self.logger = None
            print(f"[INFO] MCMCSampler: {n_params} params, {n_walkers} walkers")
            print(f"[INFO] HDF5 Backend: {self.backend_path}")
    
    def _init_backend(self, reset: bool = False):
 
        if reset and self.backend_path.exists():
            # Delete old file to ensure clean start
            self.backend_path.unlink()
            if self.logger:
                self.logger.info(f"🗑️  Deleted old backend file: {self.backend_path}")
        
        self.backend = emcee.backends.HDFBackend(str(self.backend_path))
        
        if reset or not self.backend_path.exists():
            # Start fresh
            self.backend.reset(self.n_walkers, self.n_params)
            if self.logger:
                self.logger.info(f"✅ HDF5 Backend initialized (FRESH START)")
                self.logger.info(f"   File: {self.backend_path}")
                self.logger.info(f"   Walkers: {self.n_walkers}, Parameters: {self.n_params}")
        else:
            # Resume from existing
            saved_steps = self.backend.iteration
            if self.logger:
                self.logger.info(f"")
                self.logger.info(f"═══════════════════════════════════════════════════════")
                self.logger.info(f"🔄 DISASTER RECOVERY MODE ACTIVATED")
                self.logger.info(f"═══════════════════════════════════════════════════════")
                self.logger.info(f"✅ Existing HDF5 backend found with {saved_steps} steps")
                self.logger.info(f"   File: {self.backend_path}")
                self.logger.info(f"   📦 Resuming run from step {saved_steps + 1}")
                self.logger.info(f"   Walkers: {self.n_walkers}, Parameters: {self.n_params}")
                self.logger.info(f"═══════════════════════════════════════════════════════")
                self.logger.info(f"")
    
    def _init_sampler(self):
    
        moves = [
            (emcee.moves.DEMove(),        0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ]
        if self.logger:
            self.logger.info("Move strategy: 80% DEMove + 20% DESnookerMove (ter Braak & Vrugt 2008)")

        if self.use_parallel:
            if self.pool is None:
                ctx = mp.get_context('spawn')
                self.pool = ctx.Pool(self.n_processes, maxtasksperchild=1)
                if self.logger:
                    self.logger.info(f"Initialized multiprocessing pool: {self.n_processes} workers (spawn context, maxtasksperchild=1)")

            self.sampler = emcee.EnsembleSampler(
                self.n_walkers,
                self.n_params,
                self.log_prob_fn,
                moves=moves,
                pool=self.pool,
                backend=self.backend
            )
        else:
            self.sampler = emcee.EnsembleSampler(
                self.n_walkers,
                self.n_params,
                self.log_prob_fn,
                moves=moves,
                backend=self.backend
            )
    
    def run(self,
            initial_positions: np.ndarray,
            n_steps: int,
            resume: bool = False,
            show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    
        if self.logger:
            self.logger.info("="*80)
            self.logger.info(f"STARTING MCMC RUN")
            self.logger.info("="*80)
            self.logger.info(f"  • Target steps: {n_steps}")
            self.logger.info(f"  • Resume mode: {resume}")
            self.logger.info(f"  • Backend: {self.backend_path}")
        
        # Initialize backend
        self._init_backend(reset=not resume)
        
        # Initialize sampler
        if self.sampler is None:
            self._init_sampler()
        
        # Determine starting point
        if resume and self.backend.iteration > 0:
            # Resume from saved state
            start_iteration = self.backend.iteration
            # Get last state from backend: chain shape is (n_steps, n_walkers, n_params)
            # We need last step for all walkers: (n_walkers, n_params)
            last_chain = self.backend.get_chain()[-1, :, :]  # Last step, all walkers
            state = last_chain
            remaining_steps = n_steps - start_iteration
            
            if remaining_steps <= 0:
                if self.logger:
                    self.logger.info(f"✅ Already completed {start_iteration}/{n_steps} steps. Nothing to run.")
                return self.get_chain(), self.get_log_prob()
            
            if self.logger:
                self.logger.info(f"📦 Resuming from step {start_iteration}")
                self.logger.info(f"   Remaining: {remaining_steps} steps")
            
            n_steps_to_run = remaining_steps
        else:
            # Fresh start
            start_iteration = 0
            state = initial_positions
            n_steps_to_run = n_steps
            
            if self.logger:
                self.logger.info(f"🆕 Fresh start: running {n_steps_to_run} steps")
        
        # Progress bar with unbuffered output
        if show_progress:
            pbar = tqdm(
                total=n_steps_to_run, 
                desc="MCMC Sampling",
                initial=0,
                unit="step",
                file=sys.stdout,
                mininterval=1.0,
                smoothing=0.1
            )
        
        start_time = time.time()
        last_log_time = start_time
        last_step_time = start_time  # Track time of last completed step
        
        # ═══════════════════════════════════════════════════════════
        # MAIN SAMPLING LOOP - HDF5 Backend handles all storage!
        # ═══════════════════════════════════════════════════════════
        
        step_count = 0
        step_timeout = 18000  # 5 hour timeout per step (very generous for slow VMs)
        
        for sample in self.sampler.sample(state, iterations=n_steps_to_run, progress=False):
            step_count += 1
            current_step_time = time.time()
            
            # Watchdog: Detect stuck iterations
            step_duration = current_step_time - last_step_time
            if step_duration > step_timeout:
                if self.logger:
                    self.logger.error(f"⚠️ WATCHDOG: Step took {step_duration/60:.1f} min (>{step_timeout/60:.0f} min timeout)")
                    self.logger.error(f"   This indicates a deadlock. Attempting graceful exit...")
                # Save what we have and exit
                if show_progress:
                    pbar.close()
                if self.pool:
                    self.pool.terminate()
                    self.pool.join()
                raise RuntimeError(f"MCMC step timeout after {step_duration/60:.1f} minutes")
            
            last_step_time = current_step_time
            
            # Track best fit (lightweight - only read last step's log_prob)
            # Fixed indexing: [-1, :] = last step, all walkers (not all steps, last walker)
            current_log_probs = self.backend.get_log_prob()[-1, :]  # Last step, all walkers
            max_log_prob = np.max(current_log_probs)
            
            if max_log_prob > self.best_log_prob:
                best_walker = np.argmax(current_log_probs)
                self.best_log_prob = max_log_prob
                # Correct indexing: [last_step, best_walker, all_params]
                self.best_params = self.backend.get_chain()[-1, best_walker, :].copy()
                self.best_iteration = self.backend.iteration
                
                if self.logger:
                    params_dict = {name: val for name, val in zip(self.param_names, self.best_params)}
                    self.logger.log_best_fit_update(
                        self.best_iteration, 
                        params_dict, 
                        self.best_log_prob
                    )
            
            # Periodic logging (every 100 steps or 30 seconds)
            current_time = time.time()
            if step_count % 100 == 0 or (current_time - last_log_time) > 30:
                self._log_diagnostics()
                last_log_time = current_time
            
            # Update progress bar with flush
            if show_progress:
                pbar.update(1)
                acc_rate = np.mean(self.sampler.acceptance_fraction)
                pbar.set_postfix({
                    "acc_rate": f"{acc_rate:.3f}", 
                    "best_lnL": f"{self.best_log_prob:.2f}",
                    "saved": self.backend.iteration
                })
                sys.stdout.flush()  # Force immediate output
        
        if show_progress:
            pbar.close()
        
        # ═══════════════════════════════════════════════════════════
        
        total_time = time.time() - start_time
        final_iteration = self.backend.iteration
        
        if self.logger:
            self.logger.info("="*80)
            self.logger.info(f"✅ MCMC COMPLETED")
            self.logger.info("="*80)
            self.logger.info(f"  • Total steps saved: {final_iteration}")
            self.logger.info(f"  • Runtime: {total_time:.1f}s ({total_time/n_steps_to_run:.2f}s per step)")
            self.logger.info(f"  • Final acceptance rate: {np.mean(self.sampler.acceptance_fraction):.3f}")
            self.logger.info(f"  • Best log-prob: {self.best_log_prob:.4f} at iteration {self.best_iteration}")
            self.logger.info(f"  • Backend file: {self.backend_path} ({self.backend_path.stat().st_size / 1024**2:.1f} MB)")
        
        self._cleanup_pool()
        
        # Return chain from backend (no RAM accumulation!)
        return self.get_chain(), self.get_log_prob()
    
    def get_chain(self, discard: int = 0, thin: int = 1, flat: bool = False) -> np.ndarray:

        if self.backend is None:
            raise RuntimeError("Backend not initialized. Run MCMC first or load existing backend.")
        
        return self.backend.get_chain(discard=discard, thin=thin, flat=flat)
    
    def get_log_prob(self, discard: int = 0, thin: int = 1, flat: bool = False) -> np.ndarray:
        if self.backend is None:
            raise RuntimeError("Backend not initialized. Run MCMC first or load existing backend.")
        
        return self.backend.get_log_prob(discard=discard, thin=thin, flat=flat)
    
    def get_samples(self, 
                   discard: int = 0, 
                   thin: int = 1,
                   flat: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        chain = self.get_chain(discard=discard, thin=thin, flat=flat)
        log_prob = self.get_log_prob(discard=discard, thin=thin, flat=flat)
        
        return chain, log_prob
    
    def load_backend(self, backend_path: Optional[str] = None):

        if backend_path is not None:
            self.backend_path = Path(backend_path)
        
        if not self.backend_path.exists():
            raise FileNotFoundError(f"Backend file not found: {self.backend_path}")
        
        self.backend = emcee.backends.HDFBackend(str(self.backend_path))
        
        if self.logger:
            self.logger.info(f"Loaded HDF5 backend: {self.backend_path}")
            self.logger.info(f"  • Saved steps: {self.backend.iteration}")
            self.logger.info(f"  • Walkers: {self.backend.shape[0]}")
            self.logger.info(f"  • Params: {self.backend.shape[1]}")
    
    def get_backend_info(self) -> Dict:

        if self.backend is None:
            return {"status": "not_initialized"}
        
        return {
            "backend_path": str(self.backend_path),
            "iteration": self.backend.iteration,
            "shape": self.backend.shape,
            "file_size_mb": self.backend_path.stat().st_size / 1024**2 if self.backend_path.exists() else 0,
            "best_log_prob": self.best_log_prob,
            "best_iteration": self.best_iteration,
        }
    
    def _log_diagnostics(self):
        """Log convergence diagnostics (lightweight version)."""
        if self.logger is None or self.backend is None:
            return
        
        try:
            # Acceptance rate
            acc_rate = np.mean(self.sampler.acceptance_fraction)
            
            # Get only last 100 steps for stats (avoid loading full chain)
            n_saved = self.backend.iteration
            start_idx = max(0, n_saved - 100)
            recent_log_prob = self.backend.get_log_prob()[start_idx:, :]
            
            mean_lnL = np.mean(recent_log_prob)
            std_lnL = np.std(recent_log_prob)
            
            # Try autocorrelation (only if enough steps)
            if n_saved > 50:
                try:
                    # Sample last 200 steps for autocorr (avoid memory issues)
                    sample_size = min(200, n_saved)
                    chain_sample = self.backend.get_chain()[-sample_size:, :, :]
                    tau = emcee.autocorr.integrated_time(chain_sample, quiet=True)
                    mean_tau = np.mean(tau)
                except:
                    mean_tau = None
            else:
                mean_tau = None
            
            self.logger.log_chain_stats(
                iteration=n_saved,
                acceptance_rate=acc_rate,
                mean_likelihood=mean_lnL,
                std_likelihood=std_lnL,
                autocorr_time=mean_tau
            )
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Diagnostics logging failed: {e}")
    
    def save_metadata(self, filename: Optional[str] = None, **kwargs):
       
        if filename is None:
            filename = self.checkpoint_dir / "mcmc_metadata.json"
        
        metadata = {
            "n_params": self.n_params,
            "n_walkers": self.n_walkers,
            "param_names": self.param_names,
            "backend_file": str(self.backend_path),
            "best_log_prob": float(self.best_log_prob),
            "best_params": self.best_params.tolist() if self.best_params is not None else None,
            "best_iteration": int(self.best_iteration),
            **kwargs
        }
        
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Metadata saved: {filename}")
    
    def compute_autocorr_time(self) -> Optional[np.ndarray]:

        if self.backend is None:
            return None
        
        try:
            # Get chain from backend
            chain = self.backend.get_chain()  # (n_steps, n_walkers, n_params)
            
            # Need at least 50 steps for meaningful autocorr
            if chain.shape[0] < 50:
                if self.logger:
                    self.logger.debug(f"Too few steps ({chain.shape[0]}) for autocorrelation")
                return None
            
            # emcee expects (n_steps, n_walkers, n_params) - no transpose needed
            tau = emcee.autocorr.integrated_time(chain, quiet=True)
            return tau
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Could not compute autocorr time: {e}")
            return None
    
    def compute_gelman_rubin(self, discard: int = 0) -> Optional[np.ndarray]:

        if self.backend is None:
            return None
        
        try:
            # Get chain from backend
            chain = self.backend.get_chain()  # (n_steps, n_walkers, n_params)
            
            if chain.shape[0] <= discard:
                return None
            
            # Discard burn-in
            chain = chain[discard:, :, :]  # (n_steps_post_burnin, n_walkers, n_params)
            n_steps, n_walkers, n_params = chain.shape
            
            if n_steps < 2:
                return None
            
            # Transpose to (n_walkers, n_steps, n_params)
            chain = chain.transpose(1, 0, 2)
            
            # Split each chain in half
            n_half = n_steps // 2
            chain_split = np.concatenate([
                chain[:, :n_half, :],
                chain[:, n_half:2*n_half, :]
            ], axis=0)
            
            n_chains = chain_split.shape[0]
            
            # Compute R-hat for each parameter
            r_hat = np.zeros(n_params)
            
            for i in range(n_params):
                # Within-chain variance
                W = np.mean(np.var(chain_split[:, :, i], axis=1, ddof=1))
                
                # Between-chain variance
                chain_means = np.mean(chain_split[:, :, i], axis=1)
                B = n_half * np.var(chain_means, ddof=1)
                
                # Variance estimate
                var_estimate = ((n_half - 1) / n_half) * W + (1 / n_half) * B
                
                # R-hat
                r_hat[i] = np.sqrt(var_estimate / W) if W > 0 else 1.0
            
            return r_hat
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Could not compute Gelman-Rubin: {e}")
            return None
    
    def _cleanup_pool(self):
        """Clean up multiprocessing pool."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
            if self.logger:
                self.logger.info("Multiprocessing pool closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        self._cleanup_pool()


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def load_chain_from_backend(backend_path: str, 
                           discard: int = 0, 
                           thin: int = 1, 
                           flat: bool = True) -> Tuple[np.ndarray, np.ndarray]:
  
    backend = emcee.backends.HDFBackend(backend_path, read_only=True)
    
    chain = backend.get_chain(discard=discard, thin=thin, flat=flat)
    log_prob = backend.get_log_prob(discard=discard, thin=thin, flat=flat)
    
    return chain, log_prob


def get_backend_info(backend_path: str) -> Dict:

    backend = emcee.backends.HDFBackend(backend_path, read_only=True)
    
    return {
        "iterations": backend.iteration,
        "shape": backend.shape,  # (n_walkers, n_params)
        "file_size_mb": Path(backend_path).stat().st_size / 1024**2,
    }


if __name__ == "__main__":
    # Test HDF5 backend
    print("="*80)
    print("MCMC Sampler with HDF5 Backend - Test")
    print("="*80)
    
    # Simple test log-probability
    def log_prob(x):
        return -0.5 * np.sum(x**2)
    
    # Initialize sampler
    sampler = MCMCSampler(
        log_prob_fn=log_prob,
        n_params=3,
        n_walkers=10,
        param_names=["x", "y", "z"],
        checkpoint_dir="./test_checkpoints"
    )
    
    # Initial positions
    initial = np.random.randn(10, 3)
    
    # Run
    print("\n1. Running 50 steps...")
    chain, log_prob = sampler.run(initial, n_steps=50, resume=False)
    print(f"   Chain shape: {chain.shape}")
    
    # Resume
    print("\n2. Resuming for 30 more steps...")
    chain, log_prob = sampler.run(initial, n_steps=80, resume=True)
    print(f"   Chain shape: {chain.shape}")
    
    # Info
    info = sampler.get_backend_info()
    print(f"\n3. Backend info:")
    for key, val in info.items():
        print(f"   {key}: {val}")
    
    print("\n✅ Test complete!")
