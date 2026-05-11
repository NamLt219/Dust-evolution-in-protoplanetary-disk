import os
import sys
import shutil
import psutil  
import numpy as np
from multiprocessing import set_start_method
from astropy.io import fits
import argparse
import traceback
import platform
from datetime import datetime

# Import config explicitly (no star-import pollution)
import mcmc_pipeline_config as config
from mcmc_logger import setup_logger
from forward_simulator import ForwardModelSimulatorV2
from likelihood_calculator import (
    LikelihoodCalculator, 
    PriorEvaluator, 
    MCMCProbability
)
from mcmc_sampler import MCMCSampler

# RAM Guardian for pre-flight check
try:
    from ram_guardian import print_ram_report, get_ram_status
    RAM_GUARDIAN_AVAILABLE = True
except ImportError:
    RAM_GUARDIAN_AVAILABLE = False

class MCMCPipeline:
    def __init__(self, resume: bool = True, n_steps_override: int = None):
        self.resume = resume
        # CLI override: allows running extra steps without editing config
        self.n_steps_total = n_steps_override if n_steps_override is not None else config.N_STEPS_TOTAL
        
        # Setup Logger
        self.logger = setup_logger(
            log_dir=config.LOG_DIR,
            console_level="INFO",
            file_level="DEBUG"
        )
        
        log_path = os.path.abspath(self.logger.log_file) if self.logger.log_file else "<console only>"
        print(f"📝 CRITICAL: Log file is being written to: {log_path}", flush=True)

        self.logger.info("="*80)
        self.logger.info("🛡️  MCMC PIPELINE ORCHESTRATOR (SAFE MODE) STARTED")
        self.logger.info("="*80)
        
      
        self._check_system_health()

        self.obs_data = None
        self.simulator = None
        self.likelihood = None
        self.prior = None
        self.prob_fn = None
        self.sampler_engine = None

    def _check_system_health(self):
        """System health check before MCMC run."""
        self.logger.info("🏥 Performing System Health Check...")
        
        # 1. Check Disk Space
        total, used, free = shutil.disk_usage(config.BASE_DIR)
        free_gb = free / (1024**3)
        self.logger.info(f"   Disk Free Space: {free_gb:.2f} GB")
        
        if free_gb < 2.0:
            self.logger.warning("⚠️ LOW DISK SPACE! Less than 2GB remaining.")
            
        # 2. Check RAM (if psutil available)
        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            total_gb = mem.total / (1024**3)
            self.logger.info(f"   RAM Available: {available_gb:.2f} / {total_gb:.2f} GB")
            if available_gb < 1.0:
                self.logger.warning("⚠️ LOW RAM! Less than 1GB available. Risk of OOM Crash.")
            
            # RAM budget check for N_PROCESSES workers
            ram_per_worker_gb = 1.2  # Estimated: DustPy + RADMC-3D peak
            ram_needed = config.N_PROCESSES * ram_per_worker_gb + 1.5  # Workers + OS/master
            if ram_needed > total_gb:
                self.logger.warning(
                    f"⚠️ RAM BUDGET WARNING: {config.N_PROCESSES} workers × {ram_per_worker_gb} GB "
                    f"+ 1.5 GB OS = {ram_needed:.1f} GB > {total_gb:.1f} GB total!"
                )
                self.logger.warning(f"   RAM Guardian will throttle workers to prevent OOM.")
            else:
                self.logger.info(f"   RAM Budget: {ram_needed:.1f} GB needed for {config.N_PROCESSES} workers (headroom: {total_gb - ram_needed:.1f} GB)")
        except ImportError:
            self.logger.info("   (psutil not installed, skipping RAM check)")

    def load_data(self):
        """Load observation data."""
        self.logger.info(f"📥 Loading observation: {config.OBS_FITS_PATH}")
        if not os.path.exists(config.OBS_FITS_PATH):
            self.logger.critical(f"❌ File not found: {config.OBS_FITS_PATH}")
            sys.exit(1)
            
        with fits.open(config.OBS_FITS_PATH) as hdul:
            self.obs_data = hdul[0].data.squeeze()
            while self.obs_data.ndim > 2:
                self.obs_data = self.obs_data[0]
                
            # Log image statistics to verify correct loading
            self.logger.info(f"   Image Stats: Min={np.nanmin(self.obs_data):.2e}, Max={np.nanmax(self.obs_data):.2e} Jy/beam")

    def setup_components(self):
        """Initialize pipeline components."""
        self.logger.info("⚙️  Setting up pipeline components...")
        
        try:
            self.simulator = ForwardModelSimulatorV2(cleanup=True)
            
            # Pixel scale computation
            pixel_scale_arcsec = config.IMAGE_SIZE_AU / config.IMAGE_NPIX / config.DISTANCE_PC  # arcsec/pixel
            
            self.likelihood = LikelihoodCalculator(
                obs_image=self.obs_data,
                rms_noise=config.RMS_NOISE_JY,
                roi_radius_pixels=None,
                beam_major_arcsec=config.BEAM_MAJOR_ARCSEC,
                beam_minor_arcsec=config.BEAM_MINOR_ARCSEC,
                pixel_scale_arcsec=pixel_scale_arcsec,
                align_centers=True  # ✅ CENTERING CORRECTION - aligns model to obs peak
            )
            
            self.prior = PriorEvaluator(config.MCMC_PARAMETERS)
            self.prob_fn = MCMCProbability(self.prior, self.likelihood, self.simulator)
            
            # Sampler Engine
            self.sampler_engine = MCMCSampler(
                log_prob_fn=self.prob_fn,
                n_params=len(config.MCMC_PARAMETERS),
                n_walkers=config.N_WALKERS,
                param_names=[p['name'] for p in config.MCMC_PARAMETERS],
                checkpoint_dir=config.MCMC_OUTPUT_DIR,
                backend_filename=config.MCMC_BACKEND_FILENAME,
                use_parallel=config.USE_MULTIPROCESSING,
                n_processes=config.N_PROCESSES
            )
            self.logger.info("✅ Components initialized.")
            
        except Exception as e:
            self.logger.critical("❌ Component Setup Failed!", exception=e)
            raise e

    def _generate_initial_positions(self):
        """Generate initial walker positions with adaptive scatter."""
        pos = []
        defaults = np.array([p['default'] for p in config.MCMC_PARAMETERS])
        
        for _ in range(config.N_WALKERS):
            # Scatter adaptively: 2% of parameter value (or 0.01 for values near 0)
            scatter = np.maximum(0.02 * np.abs(defaults), 0.01)
            p_walk = defaults + scatter * np.random.randn(len(defaults))
            
            # Clip to prior bounds
            for i, param in enumerate(config.MCMC_PARAMETERS):
                p_walk[i] = np.clip(p_walk[i], param['min'], param['max'])
            pos.append(p_walk)
        
        return np.array(pos)

    def run(self):
        """Execute MCMC pipeline."""
        backend_path = os.path.join(config.MCMC_OUTPUT_DIR, config.MCMC_BACKEND_FILENAME)
        backend_exists = os.path.exists(backend_path)
        should_reset = (not self.resume) or (not backend_exists)

        # ── RESUME VERIFICATION BANNER (look for this in logs!) ──────────────
        if not should_reset and backend_exists:
            import emcee as _emcee
            _b = _emcee.backends.HDFBackend(backend_path)
            saved_steps = _b.iteration
            remaining = self.n_steps_total - saved_steps
            self.logger.info("=" * 70)
            self.logger.info("🔄 RESUME MODE CONFIRMED — NOT starting fresh")
            self.logger.info(f"   Backend : {backend_path}")
            self.logger.info(f"   Steps already saved : {saved_steps}")
            self.logger.info(f"   Target total steps  : {self.n_steps_total}")
            self.logger.info(f"   Steps to run NOW    : {remaining}")
            self.logger.info("=" * 70)
            if remaining <= 0:
                self.logger.warning(
                    f"⚠️  Target ({self.n_steps_total}) already reached! "
                    f"Nothing to do. Use --n-steps to set a higher target."
                )
                return
        else:
            self.logger.info("=" * 70)
            self.logger.info("🏁 FRESH START — new chain will be created")
            self.logger.info(f"   Target total steps  : {self.n_steps_total}")
            self.logger.info("=" * 70)
        # ─────────────────────────────────────────────────────────────────────

        initial_state = self._generate_initial_positions()

        # ── PRE-FLIGHT VERIFICATION BANNER ────────────────────────────────────
        print("", flush=True)
        print("=" * 70, flush=True)
        print("  FINAL PHYSICAL MCMC — PRE-FLIGHT CHECKLIST", flush=True)
        print("=" * 70, flush=True)
        print(f"  LOCKED GEOMETRY", flush=True)
        print(f"    Inclination        : {config.INCLINATION_DEG:.1f} deg", flush=True)
        print(f"    PA (obs sky)       : {config.PA_OBS_DEG:.1f} deg", flush=True)
        print(f"    posang (RADMC-3D)  : {config.POSITION_ANGLE_DEG:.1f} deg", flush=True)
        print(f"  HARD-LOCKED SHIFT (applied inside likelihood)", flush=True)
        print(f"    DX_SHIFT           : {config.DX_SHIFT:+.4f} px  (col)", flush=True)
        print(f"    DY_SHIFT           : {config.DY_SHIFT:+.4f} px  (row)", flush=True)
        print(f"  FIXED PHYSICAL PARAMS", flush=True)
        print(f"    FLARING_INDEX (psi): {config.FLARING_INDEX:.1f}  (H ∝ r^psi, psi = 1 + 0.3)", flush=True)
        print(f"  FREE PARAMETERS ({len(config.MCMC_PARAMETERS)} dims)", flush=True)
        for p in config.MCMC_PARAMETERS:
            print(f"    {p['name']:12s}  in [{p['min']}, {p['max']}]  "
                  f"start={p['default']}", flush=True)
        print(f"  SAMPLER", flush=True)
        print(f"    N_walkers          : {config.N_WALKERS}", flush=True)
        print(f"    N_steps (target)   : {self.n_steps_total}", flush=True)
        print(f"    Backend file       : {config.MCMC_BACKEND_FILENAME}", flush=True)
        print(f"    Multiprocessing    : {config.USE_MULTIPROCESSING}  "
              f"(N_processes={config.N_PROCESSES})", flush=True)
        print("=" * 70, flush=True)
        print("", flush=True)
        # ─────────────────────────────────────────────────────────────────────

        try:
            self.sampler_engine.run(
                initial_positions=initial_state,
                n_steps=self.n_steps_total,
                resume=not should_reset,
                show_progress=True
            )
            self.logger.info("🎉 MCMC Run Finished Successfully!")
            
        except KeyboardInterrupt:
            self.logger.warning("🛑 Interrupted by user. Data is safe in HDF5.")
            sys.exit(0)
            
        except Exception as e:
            # Catch unexpected runtime errors (like RADMC crash)
            self.logger.critical("❌ RUNTIME ERROR in MCMC Loop", exception=e)
            raise e

def dump_crash_report(e):
    """Write crash report to dedicated file when pipeline fails catastrophically."""
    import mcmc_pipeline_config as config
    report_file = os.path.join(config.BASE_DIR, f"CRASH_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, "w") as f:
        f.write("="*40 + "\n")
        f.write("       MCMC PIPELINE CRASH REPORT       \n")
        f.write("="*40 + "\n\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"System: {platform.system()} {platform.release()}\n")
        f.write(f"Python: {sys.version}\n\n")
        f.write("-" * 20 + " ERROR DETAILS " + "-" * 20 + "\n")
        f.write(f"Exception Type: {type(e).__name__}\n")
        f.write(f"Error Message: {str(e)}\n\n")
        f.write("-" * 20 + " TRACEBACK " + "-" * 20 + "\n")
        traceback.print_exc(file=f)
    
    print(f"\n🔥 CRITICAL CRASH! Report saved to: {report_file}")
    print(f"🔥 Please send this file to your Senior Dev.")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        set_start_method('spawn')  # ✅ Must match mcmc_sampler.py (spawn, not forkserver)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Start fresh (WARNING: destroys existing chain!)")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Total target steps (overrides N_STEPS_TOTAL in config). "
                             "For resume: set to current_steps + new_steps (e.g. 600 to add 300 more to a 300-step chain).")
    args = parser.parse_args()

    try:
        pipeline = MCMCPipeline(resume=not args.clean, n_steps_override=args.n_steps)
        pipeline.load_data()
        pipeline.setup_components()
        pipeline.run()
        pipeline.logger.info("✅ Compute phase complete. No plotting executed in compute-only mode.")
        pipeline.logger.info("   Use `mcmc_visualizer.py` for live/post-run presentation plots.")
        
    except Exception as e:
        # Catch errors at top-level (highest exception handler)
        # This is the final safety layer
        dump_crash_report(e)
        sys.exit(1)
