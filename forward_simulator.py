import os
import sys
import shutil
import subprocess
import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import tempfile
import json
from scipy.ndimage import gaussian_filter, rotate as scipy_rotate
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
import uuid
import gc
import h5py

# RAM Guardian for OOM protection in multiprocessing
try:
    from ram_guardian import check_ram_or_wait
    RAM_GUARDIAN_AVAILABLE = True
except ImportError:
    RAM_GUARDIAN_AVAILABLE = False


try:
    from radmc3dPy.image import makeImage, readImage
    RADMC3DPY_AVAILABLE = True
except ImportError:
    RADMC3DPY_AVAILABLE = False
    print("WARNING: radmc3dPy not available. Install with: pip install radmc3dPy")

# Import config
try:
    import mcmc_pipeline_config as config
    from mcmc_logger import get_logger
    
    WORK_DIR = config.WORK_DIR
    RADMC3D_EXECUTABLE = config.RADMC3D_EXECUTABLE
    RADMC3D_OPACITY_FILE = config.RADMC3D_OPACITY_FILE
    # TEMPLATE_INI_PATH removed - DustPy doesn't need template file
    WAV_MICRON = config.WAV_MICRON
    SIMULATION_TIMEOUT = config.SIMULATION_TIMEOUT
    INCLINATION_DEG = config.INCLINATION_DEG
    POSITION_ANGLE_DEG = config.POSITION_ANGLE_DEG
    PA_OBS_DEG = config.PA_OBS_DEG  # Observation PA (121.5°)
    IMAGE_SIZE_AU = config.IMAGE_SIZE_AU
    IMAGE_NPIX = getattr(config, 'IMAGE_NPIX', config.NPIX)  # Fallback to NPIX
    BEAM_MAJOR_ARCSEC = config.BEAM_MAJOR_ARCSEC
    BEAM_MINOR_ARCSEC = config.BEAM_MINOR_ARCSEC
    BEAM_PA_DEG = config.BEAM_PA_DEG  # Beam position angle (22°, NOT disk PA!)
    DISTANCE_PC = config.DISTANCE_PC
    STAR_MASS_MSUN = config.STAR_MASS_MSUN
    STAR_RADIUS_RSUN = config.STAR_RADIUS_RSUN
    STAR_TEMP_K = config.STAR_TEMP_K
    STAR_LUMI_LSUN = getattr(config, 'STAR_LUMI_LSUN', 0.41)  # Default from paper
    RMS_NOISE_JY = config.RMS_NOISE_JY 
    OBSERVED_FLUX_1P3MM_MJY = getattr(config, 'OBSERVED_FLUX_1P3MM_MJY', 70.8)  # 70.8 mJy
except Exception as e:
    print(f"Error importing config: {e}")
    raise

# Import DustPy
try:
    from dustpy import Simulation
    import dustpy.constants as c
except ImportError as e:
    print(f"Error importing DustPy: {e}")
    raise

# Physical constants (matching reference implementation)
au = 1.49598e13      # Astronomical Unit [cm]
pc = 3.08572e18      # Parsec [cm]
ms = 1.98892e33      # Solar mass [g]
ts = 5.78e3          # Solar temperature [K]
ls = 3.8525e33       # Solar luminosity [erg/s]
rs = 6.96e10         # Solar radius [cm]
ss = 5.6703e-5       # Stefan-Boltzmann const [erg/cm^2/K^4/s]
kk = 1.3807e-16      # Bolzmann's constant [erg/K]
mp = 1.6726e-24      # Mass of proton [g]
GG = 6.67408e-08     # Gravitational constant [cm^3/g/s^2]
pi = np.pi


def grid_refine_inner_edge(x_orig, nlev, nspan):

    x = x_orig.copy()
    rev = x[0] > x[1]
    for ilev in range(nlev):
        x_new = 0.5 * (x[1:nspan+1] + x[:nspan])
        x_ref = np.hstack((x, x_new))
        x_ref.sort()
        x = x_ref
        if rev:
            x = x[::-1]
    return x


class ForwardModelSimulatorV2:

    def __init__(self, config_dict: Optional[Dict] = None, cleanup: bool = True):
        """Initialize simulator with optional config override."""
        # ❌ REMOVED: self.logger = get_logger() (causes pickling error)
        # Logger is now accessed via @property below
        
        # Use provided config or load from module
        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)
        
        # Store commonly used values
        self.work_dir = Path(WORK_DIR)
        self.radmc3d_exec = RADMC3D_EXECUTABLE
        self.opacity_file = RADMC3D_OPACITY_FILE
        self._cleanup = cleanup 
        
        self._log_info("ForwardModelSimulatorV2 initialized with FIXED pipeline")
        self._log_info(f"  - nphi = 64 (proper azimuthal resolution)")
        self._log_info(f"  - nphot = 500,000 (production Monte Carlo)")
        self._log_info(f"  - radmc3dPy: {'AVAILABLE' if RADMC3DPY_AVAILABLE else 'NOT AVAILABLE'}")
    
    @property
    def logger(self):
        """Get logger on-demand (for pickling safety)."""
        try:
            return get_logger()
        except:
            return None
        
    def _log_info(self, msg: str):
        """Convenience logging."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"INFO: {msg}")
    
    def _log_debug(self, msg: str):
        if self.logger:
            self.logger.debug(msg)
        else:
            print(f"DEBUG: {msg}")
    
    def _log_warning(self, msg: str):
        if self.logger:
            self.logger.warning(msg)
        else:
            print(f"WARNING: {msg}")
    
    def _log_error(self, msg: str, exception: Optional[Exception] = None):
        if self.logger:
            self.logger.error(msg, exc_info=exception)
        else:
            print(f"ERROR: {msg}")
            if exception:
                print(f"  Exception: {exception}")
    
    def __getstate__(self):
        """Pickle support: Logger is now a property, so pickling is automatic."""
        # Return full state (logger property won't be pickled)
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        """Pickle support: Restore state (logger property auto-recreates)."""
        self.__dict__.update(state)
        # No need to recreate logger - it's now a @property
    
    def simulate(self, params: Dict[str, float]) -> Tuple[bool, Optional[np.ndarray], Optional[Dict]]:
        """Run complete forward model simulation.
        
        Returns:
            Tuple of (success: bool, image: Optional[np.ndarray], metadata: Optional[Dict])
        """
        self._log_info(f"Starting simulation with params: {params}")
        
        # RAM Guardian: Wait if system memory is critically low
        if RAM_GUARDIAN_AVAILABLE:
            check_ram_or_wait(caller_info=f"simulate({list(params.values())[:2]})")
        
        # Create isolated working directory
        sim_id = uuid.uuid4().hex[:8]
        sim_dir = self.work_dir / f"sim_{sim_id}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # STEP 1: Run DustPy
            self._log_debug("Step 1/4: Running DustPy simulation")
            dustpy_success, dustpy_output = self._run_dustpy(sim_dir, params)
            if not dustpy_success:
                return False, None, None  # ✅ consistent 3-value interface
            
           
            self._log_debug("Step 2/4: Converting to RADMC3D (V2 with fixes)")
            radmc_dir = sim_dir / "radmc3d_work"
            radmc_dir.mkdir(exist_ok=True)
            self._convert_dustpy_to_radmc3d_v2(dustpy_output, radmc_dir, params)
            
            
            self._log_debug("Step 3/4: Running RADMC3D (V2 with radmc3dPy)")
            image = self._run_radmc3d_v2(radmc_dir, params)
            
            if image is None:
                return False, None, None
            
            # ===== SAFETY TRAPS =====
            # Check for NaN/Inf in image
            n_nan = np.sum(np.isnan(image))
            n_inf = np.sum(np.isinf(image))
            if n_nan > 0 or n_inf > 0:
                self._log_error(f"🚨 SAFETY TRAP: Image contains {n_nan} NaN and {n_inf} Inf values!")
                return False, None, None
            
            # Check for negative flux (unphysical for thermal continuum)
            n_neg = np.sum(image < 0)
            frac_neg = n_neg / image.size
            if frac_neg > 0.01:  # More than 1% negative pixels
                self._log_warning(f"⚠️  SAFETY: {n_neg} negative pixels ({frac_neg:.1%}) in model image")
            
            # Check peak flux (should be > noise level)
            peak_flux = np.max(image)
            if peak_flux < RMS_NOISE_JY:
                self._log_warning(f"⚠️  SAFETY: Peak flux {peak_flux:.2e} Jy/beam < RMS noise {RMS_NOISE_JY:.2e}")
            
            # Compute integrated flux ONCE (DRY principle - reused in safety trap & metadata)
            pixel_scale_as = (IMAGE_SIZE_AU / IMAGE_NPIX) / DISTANCE_PC
            pix_sr = (pixel_scale_as * np.pi / (180.0 * 3600.0))**2
            beam_sr = (np.pi / (4.0 * np.log(2.0))) * \
                      (BEAM_MAJOR_ARCSEC * np.pi / (180.0 * 3600.0)) * \
                      (BEAM_MINOR_ARCSEC * np.pi / (180.0 * 3600.0))
            model_flux_mjy = float(np.sum(image) * pix_sr / beam_sr * 1000.0)
            
            # Check integrated flux ratio (warn if way off from observed)
            obs_flux = getattr(config, 'OBSERVED_FLUX_1P3MM_MJY', 70.8)
            if obs_flux > 0 and (model_flux_mjy / obs_flux > 100 or model_flux_mjy / obs_flux < 0.01):
                self._log_warning(f"⚠️  FLUX TRAP: Model flux {model_flux_mjy:.2f} mJy is >100× or <0.01× observed {obs_flux:.1f} mJy")
            # ===== END SAFETY TRAPS =====
            
            # STEP 4: Create metadata
            self._log_debug("Step 4/4: Simulation complete")
            metadata = {
                'params': params.copy(),
                'image_shape': image.shape,
                'dustpy_dir': str(sim_dir),
                'sim_dir': str(sim_dir), 
                'integrated_flux_mjy': model_flux_mjy,  # Reuse pre-computed value
                'success': True
            }
            
            return True, image, metadata  
            
        except Exception as e:
            self._log_error(f"Simulation failed: {e}", exception=e)
            return False, None, None 
        
        finally:
            # Cleanup - RESPECT self._cleanup flag
            if self._cleanup and sim_dir.exists():
                try:
                    shutil.rmtree(sim_dir)
                    self._log_debug(f"Cleaned up {sim_dir}")
                except:
                    pass
            gc.collect()
    
    def forward_model(self, params: Dict[str, float]) -> Tuple[bool, Optional[np.ndarray], Optional[Dict]]:
        return self.simulate(params)
    
    def _run_dustpy(self, sim_dir: Path, params: Dict[str, float]) -> Tuple[bool, Optional[Dict]]:
  
        try:
            # Extract all 6 free parameters directly from the walker's param dict.
            # Geometry (incl, PA, shift) is hard-locked in likelihood.
            # r_in is NOW A FREE PARAMETER (unlocked from fixed 5.0 AU)
            log_mdisk   = params.get('log_mdisk',    -2.0)
            r_c         = params.get('r_c',           60.0)
            vFrag       = params.get('vFrag',        200.0)   # cm/s
            sigma_exp   = params.get('sigma_exp',      1.0)
            dust_to_gas = params.get('dust_to_gas',   0.01)
            r_in        = params.get('r_in',          1.855)  # AU — NOW A FREE PARAMETER
            
         
            M_gas = 10**log_mdisk * c.M_sun  # Gas mass (primary)
            M_dust = M_gas * dust_to_gas      # Dust mass (derived from gas)
            
            self._log_debug(f"🔍 DEBUG: Initializing DustPy with M_gas={M_gas/c.M_sun:.4e} M☉, M_dust={M_dust/c.M_sun:.4e} M☉")
            self._log_debug(f"         Ratio: M_disk/M_star = {M_gas/(STAR_MASS_MSUN*c.M_sun):.3f} (should be << 1)")
            
            R_c = r_c * c.au 
            R_in = r_in * c.au
            
            # Create DustPy simulation
            sim = Simulation()
            
            # Star properties
            sim.ini.star.M = STAR_MASS_MSUN * c.M_sun
            sim.ini.star.R = STAR_RADIUS_RSUN * c.R_sun
            sim.ini.star.T = STAR_TEMP_K
            
            # Disk properties
            sim.ini.gas.Mdisk = M_gas 
            sim.ini.gas.SigmaRc = R_c 
            sim.ini.gas.SigmaExp = -sigma_exp
            sim.ini.gas.alpha = 1.0e-3  
            
           
            # DustPy default grid: [r_in, r_out] with logarithmic spacing
            sim.ini.grid.rmin = R_in  # Inner edge
            # Outer radius scaled with r_c
            sim.ini.grid.rmax = max(300 * c.au, 6 * R_c) 
            
            # Dust properties
            sim.ini.dust.aIniMax = 0.001  # cm - Initial max grain size
            sim.ini.dust.vFrag = vFrag  # cm/s
            
         
            sim.ini.grid.mmax = 7.0  # g - Maximum particle mass (= 1 cm size)
            
        
            sim.ini.dust.d2gRatio = dust_to_gas  
            
            # Initialize
            sim.initialize()

            
            if r_in > 0:
                # R_in and width are both in cm (R_in = r_in * c.au set above)
                # sim.grid.r is a simframe Field in cm — cast to plain ndarray
                # to avoid Field-operator edge cases in the arithmetic.
                r_grid = np.array(sim.grid.r)   # shape (Nr,), units: cm
                width  = 0.15 * R_in            # 15% of R_in [cm]
                cavity_profile = 0.5 * (1.0 + np.tanh((r_grid - R_in) / width))
                cavity_profile = np.maximum(cavity_profile, 1e-6)  # floor at 1e-6
                sim.gas.Sigma   *= cavity_profile           # 1D: shape (Nr,)
                sim.dust.Sigma  *= cavity_profile[:, None]  # 2D: shape (Nr, Nm) broadcast
               
                sim.update()
                self._log_info(f"Cavity carved (+ update): r_in={r_in:.2f} AU, width={width/c.au:.2f} AU, floor=1e-6")

            
            N_SNAPSHOTS = getattr(config, 'N_SNAPSHOTS', 3)  # Default reduced 10→3
            T_END_YR = getattr(config, 'T_END_YR', 5.0e4)   # 50,000 yr (Class 0)
            sim.t.snapshots = np.hstack([
                sim.t,
                np.geomspace(1.0e2, T_END_YR, num=N_SNAPSHOTS) * c.year
            ])
            
            # Output directory
            dustpy_dir = sim_dir / "dustpy_output"
            dustpy_dir.mkdir(exist_ok=True)
            sim.writer.datadir = str(dustpy_dir)
            
            # Run simulation with progress monitoring
            import sys
            import time
            start_time = time.time()
            last_log_time = start_time
            
            self._log_debug(f"🔄 Running DustPy with {N_SNAPSHOTS} snapshots...")
            self._log_debug(f"   M_dust = {M_dust/c.M_sun:.4e} M_sun")
            self._log_debug(f"   M_gas = {M_gas/c.M_sun:.4e} M_sun")
            self._log_debug(f"   R_c = {R_c/c.au:.1f} AU")
            self._log_debug(f"   vFrag = {vFrag:.1f} cm/s")
            
            # Store original update function for progress monitoring
            original_update = sim.update
            iteration_count = [0]  # Use list for mutability in nested function
            
            def monitored_update():
                """Wrapper to inject progress monitoring"""
                nonlocal last_log_time
                result = original_update()
                iteration_count[0] += 1
                
                current_time = time.time()
                if current_time - last_log_time >= 10.0:  # Log every 10 seconds
                    elapsed = current_time - start_time
                    t_current = sim.t / c.year
                    t_total = sim.t.snapshots[-1] / c.year
                    progress = (sim.t / sim.t.snapshots[-1]) * 100 if sim.t.snapshots[-1] > 0 else 0
                    self._log_debug(f"⏳ DustPy progress: {progress:.1f}% | t={t_current:.2e}/{t_total:.2e} yr | Elapsed: {elapsed:.0f}s")
                    last_log_time = current_time
                    sys.stdout.flush()
                
                return result
            
            sim.update = monitored_update
            
            try:
                sim.run()
                elapsed = time.time() - start_time
                self._log_debug(f"✅ DustPy completed in {elapsed:.1f}s ({iteration_count[0]} iterations)")
            except Exception as e:
                self._log_debug(f"❌ DustPy simulation failed: {e}")
                import traceback
                self._log_debug(f"   Traceback: {traceback.format_exc()}")
                raise
            
            # Load final snapshot
            data_files = sorted(dustpy_dir.glob("data*.hdf5"))
            if not data_files:
                raise FileNotFoundError("No DustPy output files found")
            
            snapshot_file = data_files[-1]
            self._log_debug(f"Loading snapshot: {snapshot_file}")
            
            output = self._load_dustpy_snapshot(snapshot_file)
            
            self._log_debug(f"DustPy complete: Mdust={output['total_dust_mass']:.2e} g")
            
            return True, output
            
        except Exception as e:
            self._log_error("DustPy simulation failed", exception=e)
            return False, None
    
    def _load_dustpy_snapshot(self, filename: Path) -> Dict:
        """Load DustPy snapshot and extract key quantities."""
        with h5py.File(filename, 'r') as f:
            # Grid - read all arrays and check shapes
            r = f['grid/r'][:]  # [cm]
            
            # Dust surface density [Nr, Nm] or [Nm, Nr]
            sigma_dust_2d = f['dust/Sigma'][:]
            
            self._log_debug(f"DustPy snapshot shapes: r={r.shape}, sigma_dust={sigma_dust_2d.shape}")
            
            # ===== SMART SHAPE DETECTION =====
            # Auto-detect which axis is radial by matching with len(r)
            Nr = len(r)
            
            if len(sigma_dust_2d.shape) == 2:
                # Check which dimension matches Nr (radial grid size)
                if sigma_dust_2d.shape[0] == Nr:
                    # Format (Nr, Nm) -> Sum over axis=1 (mass bins) → (Nr,)
                    sigma_dust = np.sum(sigma_dust_2d, axis=1)
                    self._log_debug(f"Detected (Nr, Nm) format: {sigma_dust_2d.shape} → sum axis=1")
                elif sigma_dust_2d.shape[1] == Nr:
                    # Format (Nm, Nr) -> Sum over axis=0 (mass bins) → (Nr,)
                    sigma_dust = np.sum(sigma_dust_2d, axis=0)
                    self._log_debug(f"Detected (Nm, Nr) format: {sigma_dust_2d.shape} → sum axis=0")
                else:
                    # Neither dimension matches - use minimum and truncate
                    self._log_warning(f"Shape mismatch! r={r.shape}, sigma={sigma_dust_2d.shape}. Truncating.")
                    min_len = min(Nr, sigma_dust_2d.shape[0], sigma_dust_2d.shape[1])
                    r = r[:min_len]
                    # Assume first axis is radial and sum over second
                    sigma_dust = np.sum(sigma_dust_2d[:min_len, :], axis=1)
            else:
                sigma_dust = sigma_dust_2d
            
            # Final verification
            if len(sigma_dust) != len(r):
                self._log_warning(f"Final mismatch: r={len(r)}, sigma={len(sigma_dust)}. Truncating.")
                min_len = min(len(r), len(sigma_dust))
                r = r[:min_len]
                sigma_dust = sigma_dust[:min_len]
            # ===================================
            
            # Gas surface density
            try:
                sigma_gas = f['gas/Sigma'][:]
                # Match length with r
                if len(sigma_gas) != len(r):
                    sigma_gas = sigma_gas[:len(r)]
            except:
                sigma_gas = sigma_dust * 100  # Fallback
            
            # Scale height from sound speed and Kepler frequency
            try:
                cs = f['gas/cs'][:]
                OmegaK = f['grid/OmegaK'][:]
                
                # Match lengths
                cs = cs[:len(r)]
                OmegaK = OmegaK[:len(r)]
                
                H = cs / OmegaK
            except Exception as e:
                self._log_debug(f"Could not compute H from cs/OmegaK: {e}")
                # Fallback: H/r ~ 0.05
                H = 0.05 * r
            
            # Temperature
            try:
                T = f['gas/T'][:]
                T = T[:len(r)]
            except:
                # Fallback: T ~ r^-0.5
                T = 200 * (r / (10*au))**(-0.5)
            
            # Ensure all arrays have same length
            min_len = min(len(r), len(sigma_dust), len(sigma_gas), len(H), len(T))
            
            result = {
                'r': r[:min_len],  # [cm]
                'sigma_dust': sigma_dust[:min_len],  # [g/cm²]
                'sigma_gas': sigma_gas[:min_len],  # [g/cm²]
                'scale_height': H[:min_len],  # [cm]
                'temperature': T[:min_len],  # [K]
                'total_dust_mass': np.trapz(2*np.pi*r[:min_len]*sigma_dust[:min_len], r[:min_len])
            }
            
            self._log_debug(f"Loaded snapshot: {min_len} radial points, Mdust={result['total_dust_mass']:.2e} g")
            
            return result
    
    def _convert_dustpy_to_radmc3d_v2(self, 
                                       dustpy_output: Dict,
                                       radmc_dir: Path,
                                       params: Dict[str, float]):

        self._log_debug("Converting DustPy → RADMC3D (V2 with fixes)")
        
        # ===== GRID SETUP (matching problem_setup_origin.py) =====
        
        # Radial grid
        nr = 100
        # Extract r_in from params (NOW A FREE PARAMETER - no longer hard-coded)
        mcmc_rin_au = params.get('r_in', 1.855)  # AU - from MCMC walker
        rin_au = max(0.1, mcmc_rin_au * 0.5)  # RADMC-3D grid starts at 0.5× r_in
        rin = rin_au * au
        self._log_debug(f"Dynamic RADMC-3D grid rin set to {rin_au:.3f} AU (MCMC r_in = {mcmc_rin_au:.3f} AU)")
        

        
        r_dustpy = dustpy_output['r']
        sigma_dustpy = dustpy_output['sigma_dust']
        
        # Compute cumulative mass as function of radius
        # Use trapezoidal integration: M(r) = ∫ 2π r' Σ(r') dr'
        integrand = 2.0 * np.pi * r_dustpy * sigma_dustpy
        cumulative_mass = np.zeros_like(r_dustpy)
        for i in range(1, len(r_dustpy)):
            cumulative_mass[i] = np.trapz(integrand[:i+1], r_dustpy[:i+1])
        
        total_mass = cumulative_mass[-1]
        
        if total_mass <= 0:
            # Edge case: no dust at all — use full DustPy extent
            r_999 = r_dustpy.max()
            self._log_warning("⚠️  Total dust mass ≤ 0 — using full DustPy extent")
        else:
            # Normalised cumulative mass fraction
            mass_fraction = cumulative_mass / total_mass
            
            # Find radius where 99.9% of mass is enclosed
            MASS_CONTAINMENT = 0.999
            idx_999 = np.searchsorted(mass_fraction, MASS_CONTAINMENT)
            idx_999 = min(idx_999, len(r_dustpy) - 1)  # clamp to array bounds
            r_999 = r_dustpy[idx_999]
        
        SAFETY_FACTOR = 1.2  # 20% margin for interpolation stability
        rout = SAFETY_FACTOR * r_999
        
        # Clamp to DustPy extent (cannot exceed what DustPy computed)
        rout = min(rout, r_dustpy.max())
        
        # Also compute old density-threshold edge for comparison logging
        sigma_peak = sigma_dustpy.max()
        mask_significant = sigma_dustpy > (1e-4 * sigma_peak)
        r_dust_edge = r_dustpy[mask_significant].max() if np.any(mask_significant) else r_dustpy.max()
        
        self._log_info(f"Dynamic grid boundary (cumulative mass fraction method):")
        self._log_info(f"  - Total dust mass = {total_mass:.4e} g")
        self._log_info(f"  - r(99.9% mass) = {r_999/au:.1f} AU")
        self._log_info(f"  - r(density threshold) = {r_dust_edge/au:.1f} AU  [old method, for reference]")
        self._log_info(f"  - RADMC-3D rout = {SAFETY_FACTOR}× r_999 = {rout/au:.1f} AU")
        
        # ===== END CRITICAL FIX =====
        
        # Initial grid
        ri = np.logspace(np.log10(rin), np.log10(rout), nr+1)
        
 
        ri = grid_refine_inner_edge(ri, nlev=12, nspan=3)
        rc = 0.5 * (ri[:-1] + ri[1:])
        nr = len(rc)  # Update nr after refinement
        
        # Theta grid (colatitude from pole)
        ntheta = 32  # Match reference (was 50)
        zrmax = 0.5
        thetaup = np.pi * 0.5 - 0.5
        thetai = np.linspace(thetaup, 0.5 * np.pi, ntheta + 1)
        thetac = 0.5 * (thetai[:-1] + thetai[1:])
        
        
        nphi = 64  # Was 1 - this is THE MAIN FIX!
        phii = np.linspace(0.0, 2 * np.pi, nphi + 1)
        phic = 0.5 * (phii[:-1] + phii[1:])
        
        self._log_debug(f"Grid: nr={nr}, ntheta={ntheta}, nphi={nphi}")
        
        # ===== DENSITY INTERPOLATION =====
        

        r_dustpy_au = dustpy_output['r'] / au
        sigma_dustpy = dustpy_output['sigma_dust']
        H_dustpy = dustpy_output['scale_height']
        
        # Create interpolators
        sigma_interp = interp1d(
            r_dustpy_au,
            sigma_dustpy,
            kind='linear',
            fill_value="extrapolate",
            bounds_error=False
        )
        

        if len(H_dustpy.shape) == 2:
            # If H is 2D, take radial average or first grain size
            H_dustpy_1d = H_dustpy.mean(axis=1) if H_dustpy.shape[1] > 1 else H_dustpy[:, 0]
        else:
            H_dustpy_1d = H_dustpy
        
        # Ensure sigma_dustpy and H are 1D with same length
        min_len = min(len(r_dustpy_au), len(sigma_dustpy), len(H_dustpy_1d))
        r_dustpy_au_matched = r_dustpy_au[:min_len]
        sigma_dustpy_matched = sigma_dustpy[:min_len]
        H_dustpy_matched = H_dustpy_1d[:min_len]
        
      
        r_dustpy_min = r_dustpy_au_matched.min()
        r_dustpy_max = r_dustpy_au_matched.max()
        
        # Recreate interpolators with matched arrays
        sigma_interp = interp1d(
            r_dustpy_au_matched,
            sigma_dustpy_matched,
            kind='linear',
            fill_value="extrapolate",
            bounds_error=False
        )
        
        H_interp = interp1d(
            r_dustpy_au_matched,
            H_dustpy_matched,
            kind='linear',
            fill_value="extrapolate",
            bounds_error=False
        )
        
        # Evaluate on RADMC3D grid (rc has been updated after refinement)
        r_au_grid = rc / au
        
        
        r_radmc_min = r_au_grid.min()
        r_radmc_max = r_au_grid.max()
        
        # Check inner edge (warn only if offset > 10%)
        EXTRAPOLATION_THRESHOLD = 0.10  # 10% threshold
        if r_radmc_min < r_dustpy_min:
            inner_offset = r_dustpy_min - r_radmc_min
            inner_offset_percent = inner_offset / r_dustpy_min
            if inner_offset_percent > EXTRAPOLATION_THRESHOLD:
                self._log_warning(
                    f"RADMC3D inner edge ({r_radmc_min:.2f} AU) is {inner_offset:.3f} AU "
                    f"({inner_offset_percent:.1%}) below DustPy inner edge ({r_dustpy_min:.2f} AU). "
                    f"Using extrapolation for inner region."
                )
        
        # ===== CRITICAL CHECK 2: Outer boundary (MUST NOT truncate mass) =====
        if r_radmc_max > r_dustpy_max:
            outer_excess = r_radmc_max - r_dustpy_max
            self._log_error(
                f"🚨 CRITICAL: RADMC-3D outer edge ({r_radmc_max:.1f} AU) exceeds "
                f"DustPy range ({r_dustpy_max:.1f} AU) by {outer_excess:.1f} AU!"
            )
            # This should NOT happen after dynamic rout fix
            raise ValueError(
                f"Grid mismatch: RADMC-3D rout ({r_radmc_max:.1f} AU) > DustPy rmax ({r_dustpy_max:.1f} AU). "
                f"BUG in dynamic boundary logic - check _convert_dustpy_to_radmc3d_v2()."
            )
        
        # ===== CRITICAL CHECK 3: Mass conservation at outer edge =====
        # With cumulative mass fraction method, mass loss should be < 0.1%
        # by construction. This check is a safety net.
        r_dustpy_full = dustpy_output['r']
        sigma_dustpy_full = dustpy_output['sigma_dust']
        mask_beyond_radmc = r_dustpy_full > (r_radmc_max * au)
        
        if np.any(mask_beyond_radmc):
            mass_beyond = np.trapz(
                2 * np.pi * r_dustpy_full[mask_beyond_radmc] * sigma_dustpy_full[mask_beyond_radmc],
                r_dustpy_full[mask_beyond_radmc]
            )
            total_mass = np.trapz(2 * np.pi * r_dustpy_full * sigma_dustpy_full, r_dustpy_full)
            fraction_lost = mass_beyond / total_mass if total_mass > 0 else 0.0
            
            if fraction_lost > 0.05:  # More than 5% lost — hard failure
                self._log_error(
                    f"🚨 MASS CONSERVATION VIOLATED: {fraction_lost:.1%} of dust mass "
                    f"beyond RADMC-3D grid ({r_radmc_max:.1f} AU)!"
                )
                raise ValueError(
                    f"Unacceptable mass loss: {fraction_lost:.1%} beyond grid. "
                    f"Increase rout or adjust DustPy parameters."
                )
            elif fraction_lost > 0.01:  # 1-5% lost — warning, continue
                self._log_warning(
                    f"⚠️  Marginal mass conservation: {fraction_lost:.1%} of dust mass "
                    f"beyond RADMC-3D grid ({r_radmc_max:.1f} AU). Proceeding anyway."
                )
            else:
                self._log_debug(f"✅ Mass conservation OK: {fraction_lost:.3%} lost at boundary")
        # ===== END CRITICAL CHECKS =====
        
        sigmad = sigma_interp(r_au_grid)
        hp = H_interp(r_au_grid)
        hpr = hp / rc

        # FLARING INDEX ψ = 1.3  — MATHEMATICAL DERIVATION:
        # Class 0 disk models give aspect ratio  h/r ∝ r^β  with β ≈ 0.3
        #   i.e.  h/r = (h/r)₀ × (r/r₀)^0.3
        # Scale height:  H(r) = r × (h/r) ∝ r^1 × r^0.3 = r^1.3
        # Flaring index ψ is defined as  H ∝ r^ψ  ⇒  ψ = 1.3
        # Note: ψ = 1.3, NOT 0.3 (common confusion: 0.3 is the h/r exponent;
        #       the H exponent = 1 + 0.3 = 1.3 because H = r × (h/r)).
        # This value matches config.FLARING_INDEX and is enforced as a floor
        # to prevent artificially flat disks from the DustPy thermal solution.
        PSI_MIN = 1.3  # Minimum flaring index (ψ = 1 + β, β = 0.3)
        
        # Measure current flaring index from DustPy output
        # Use log-log fit: log(H/r) = (ψ-1)*log(r) + const
        valid = (rc > 1.0 * au) & (hp > 0)  # Avoid inner edge artifacts
        if np.sum(valid) > 5:
            log_r = np.log10(rc[valid])
            log_hpr = np.log10(hpr[valid])
            # Linear fit: log(H/r) = slope * log(r) + intercept
            # slope = ψ - 1, so ψ = slope + 1
            coeffs = np.polyfit(log_r, log_hpr, 1)
            psi_dustpy = coeffs[0] + 1.0
            
            self._log_info(f"Flaring index from DustPy: ψ = {psi_dustpy:.3f}")
            
            if psi_dustpy < PSI_MIN:
                self._log_info(f"⚠️  Flaring override: ψ={psi_dustpy:.3f} < {PSI_MIN} → enforcing ψ={PSI_MIN}")
                
                # Reference radius: use median of valid points
                r_ref = np.median(rc[valid])
                H_ref = np.interp(r_ref, rc, hp)  # H at reference radius
                hpr_ref = H_ref / r_ref
                
                # New H/r profile: (H/r) = (H/r)_ref × (r/r_ref)^(ψ_min - 1)
                hpr_new = hpr_ref * (rc / r_ref) ** (PSI_MIN - 1.0)
                hp = hpr_new * rc  # Convert back to H [cm]
                hpr = hpr_new
                
                self._log_info(f"  Flaring override applied: H(r_ref={r_ref/au:.1f} AU) = {H_ref/au:.3f} AU preserved")
            else:
                self._log_info(f"✅ Flaring OK: ψ = {psi_dustpy:.3f} ≥ {PSI_MIN}")
        else:
            self._log_warning("Not enough valid points to measure flaring index, using DustPy H as-is")
        
        self._log_debug(f"Interpolated to refined grid: {len(r_au_grid)} points (from {min_len} DustPy points)")
        
        # Create 3D meshgrid
        rr, tt, pp = np.meshgrid(rc, thetac, phic, indexing='ij')
        

        zr_angle = np.pi/2.0 - tt  # Angle from midplane [radians]
        zz = rr * np.sin(zr_angle)  # Actual height z = r*sin(angle) [cm]
        
        # Expand 1D profiles to 3D
        sigmad_3d = sigmad[:, None, None] * np.ones((1, ntheta, nphi))
        hh = hp[:, None, None] * np.ones((1, ntheta, nphi))  # H in cm
        

        rhod = (sigmad_3d / (np.sqrt(2.0 * np.pi) * hh)) * np.exp(-zz**2 / (2.0 * hh**2))
        
        # Ensure non-negative and reasonable values
        rhod = np.maximum(rhod, 1e-30)
        
        self._log_debug(f"Density range: {rhod.min():.2e} - {rhod.max():.2e} g/cm³")
        
        # ===== WRITE RADMC3D INPUT FILES =====
        
        # 1. amr_grid.inp
        with open(radmc_dir / "amr_grid.inp", "w") as f:
            f.write("1\n")  # iformat
            f.write("0\n")  # AMR grid style (0=regular grid)
            f.write("100\n")  # Coordinate system (100=spherical)
            f.write("0\n")  # gridinfo
            f.write("1 1 1\n")  # Include r, theta, phi
            f.write(f"{nr} {ntheta} {nphi}\n")
            for r in ri:
                f.write(f"{r:13.6e}\n")
            for t in thetai:
                f.write(f"{t:13.6e}\n")
            for p in phii:
                f.write(f"{p:13.6e}\n")
        
        # 2. dust_density.inp
        with open(radmc_dir / "dust_density.inp", "w") as f:
            f.write("1\n")  # Format number
            f.write(f"{nr * ntheta * nphi}\n")  # Nr of cells
            f.write("1\n")  # Nr of dust species
            data = rhod.ravel(order='F')  # Fortran-style indexing
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')
        
        # 3. wavelength_micron.inp (matching reference)
        lam1, lam2, lam3, lam4 = 0.1, 7.0, 25.0, 1.0e4
        n12, n23, n34 = 20, 100, 30
        lam12 = np.logspace(np.log10(lam1), np.log10(lam2), n12, endpoint=False)
        lam23 = np.logspace(np.log10(lam2), np.log10(lam3), n23, endpoint=False)
        lam34 = np.logspace(np.log10(lam3), np.log10(lam4), n34, endpoint=True)
        lam = np.concatenate([lam12, lam23, lam34])
        nlam = lam.size
        
        # Ensure observing wavelength is included
        if not np.any(np.isclose(lam, WAV_MICRON, rtol=0, atol=1e-6)):
            lam = np.sort(np.append(lam, WAV_MICRON))
            nlam = lam.size
        
        with open(radmc_dir / "wavelength_micron.inp", "w") as f:
            f.write(f"{nlam}\n")
            for w in lam:
                f.write(f"{w:13.6e}\n")
        
        # 4. stars.inp (matching reference format)
        # Verify: L = 4πR²σT⁴ should give L_bol ~ 0.41 Lsun
        R_star_cm = STAR_RADIUS_RSUN * rs
        M_star_cgs = STAR_MASS_MSUN * ms  # ✅ FIX: Convert to CGS grams! (was just STAR_MASS_MSUN = 0.27)
        T_star = STAR_TEMP_K  # T_eff = 3000 K (INPUT, not T_bol=54K output!)
        L_star_calc = 4.0 * pi * R_star_cm**2 * ss * T_star**4  # [erg/s]
        L_star_Lsun = L_star_calc / ls  # Convert to solar units
        
        # Log verification
        self._log_debug(f"Stellar params: M={STAR_MASS_MSUN:.2f} Msun ({M_star_cgs:.3e} g), R={STAR_RADIUS_RSUN:.2f} Rsun, T_eff={T_star} K")
        self._log_debug(f"Computed luminosity: L={L_star_Lsun:.3f} Lsun (target={STAR_LUMI_LSUN} Lsun)")
        if abs(L_star_Lsun - STAR_LUMI_LSUN) / STAR_LUMI_LSUN > 0.05:
            self._log_warning(f"Luminosity mismatch: computed {L_star_Lsun:.3f} vs target {STAR_LUMI_LSUN} Lsun")
        
        pstar = [0., 0., 0.]
        
        with open(radmc_dir / "stars.inp", "w") as f:
            f.write("2\n")  # Format number
            f.write(f"1 {nlam}\n\n")  # 1 star, N wavelengths + blank line
            f.write(f"{R_star_cm:13.6e} {M_star_cgs:13.6e} {pstar[0]:13.6e} {pstar[1]:13.6e} {pstar[2]:13.6e}\n\n")
            for w in lam:
                f.write(f"{w:13.6e}\n")
            f.write(f"\n{-T_star:13.6e}\n")  # Negative = blackbody
        
        # 5. dustopac.inp
        # NOTE: Using single opacity file (not size distribution)
        # For 1.3mm continuum fitting, this is sufficient and ~10× faster
        # Full size-dependent opacity (dustkappa_silicate_001.inp, _002.inp, etc.)
        # would be needed for spectral index fitting across multiple wavelengths
        with open(radmc_dir / "dustopac.inp", "w") as f:
            f.write("2               Format number of this file\n")
            f.write("1               Nr of dust species\n")
            f.write("============================================================================\n")
            f.write("1               Way in which this dust species is read\n")
            f.write("0               0=Thermal grain\n")
            f.write("silicate        Extension of name of dustkappa_***.inp file\n")
            f.write("----------------------------------------------------------------------------\n")
        
        # Copy opacity file
        opacity_src = Path(self.opacity_file)
        if not opacity_src.exists():
            raise FileNotFoundError(f"Opacity file not found: {opacity_src}")
        opacity_dst = radmc_dir / opacity_src.name
        shutil.copy(opacity_src, opacity_dst)
        
        # 6. radmc3d.inp (✅ nphot increased for better MC statistics)
        nphot = params.get('nphot', 500000)  # ✅ UPDATED: 500k for production (was 100k)
        with open(radmc_dir / "radmc3d.inp", "w") as f:
            f.write(f"nphot = {nphot}\n")
            f.write("scattering_mode_max = 1\n")
            f.write("iranfreqmode = 1\n")
            f.write("istar_sphere = 1\n")
        
        self._log_debug(f"RADMC3D input files created in {radmc_dir} (nphot={nphot})")
    
    def _run_radmc3d_v2(self, radmc_dir: Path, params: Dict[str, float]) -> Optional[np.ndarray]:
 
        self._log_debug("Running RADMC3D V6 (Scientifically Correct)")
        
        original_dir = os.getcwd()
        os.chdir(radmc_dir)
        
        # Cleanup old image
        if (radmc_dir / "image.out").exists():
            (radmc_dir / "image.out").unlink()
        
        try:
            # IMPORT CONFIG
            from mcmc_pipeline_config import (
                PA_OBS_DEG, INCLINATION_DEG, IMAGE_SIZE_AU,
                IMAGE_NPIX, WAV_MICRON, SIMULATION_TIMEOUT,
                BEAM_MAJOR_ARCSEC, BEAM_MINOR_ARCSEC, BEAM_PA_DEG, DISTANCE_PC
            )

 
            incl_value   = INCLINATION_DEG         
           
            posang_value = PA_OBS_DEG - 90.0        # Fixed: PA_OBS_DEG - 90°

            # 1. RUN MCTHERM (thermal Monte Carlo)
            subprocess.run(
                f"{self.radmc3d_exec} mctherm",
                shell=True, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=SIMULATION_TIMEOUT
            )

            
            self._log_info(f"RADMC-3D Image Generation:")
            self._log_info(f"  incl: {incl_value}° (disk inclination, fixed)")
            self._log_info(f"  phi: 0° (irrelevant for axisymmetric disk)")
            self._log_info(f"  posang: {posang_value}° → PA={PA_OBS_DEG}° on sky (fixed)")
            self._log_info(f"  → Direct posang rotation (no post-processing needed)")

            sizeau_radius = IMAGE_SIZE_AU / 2.0

            cmd = (
                f"{self.radmc3d_exec} image "
                f"lambda {WAV_MICRON} "
                f"npix {IMAGE_NPIX} "
                f"sizeau {sizeau_radius} "
                f"incl {incl_value} "
                f"phi 0 "
                f"posang {posang_value} "
                f"nostar"
            )
            
            self._log_debug(f"Command: {cmd}")
            subprocess.run(
                cmd, shell=True, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=SIMULATION_TIMEOUT
            )
            
            # 3. READ IMAGE
            if not RADMC3DPY_AVAILABLE:
                self._log_error("radmc3dPy needed for reading")
                # Use fallback method
                image_data = self._fallback_radmc3d_image(radmc_dir)
                if image_data is None:
                    return None
                return image_data
                
            im_mm = readImage()
            
            # 4. CONVOLVE WITH BEAM
            # Note: Beam PA (22°) is a telescope property, applied before rotation
            cim = im_mm.imConv(
                fwhm=[BEAM_MAJOR_ARCSEC, BEAM_MINOR_ARCSEC], 
                pa=BEAM_PA_DEG, 
                dpc=DISTANCE_PC
            )
            
            # Extract Data
            image_data = cim.imageJyppix.squeeze()
            
            pixel_scale_arcsec = (IMAGE_SIZE_AU / IMAGE_NPIX) / DISTANCE_PC  # arcsec/pixel
            pixel_area_sr = (pixel_scale_arcsec * np.pi / (180.0 * 3600.0))**2  # sr
            beam_area_sr = (np.pi / (4.0 * np.log(2.0))) * \
                           (BEAM_MAJOR_ARCSEC * np.pi / (180.0 * 3600.0)) * \
                           (BEAM_MINOR_ARCSEC * np.pi / (180.0 * 3600.0))  # sr
            
            # Sum in Jy/beam × (pixel_sr / beam_sr) = Jy
            flux_total_jy = np.sum(image_data) * pixel_area_sr / beam_area_sr
            flux_total_mjy = flux_total_jy * 1000.0
            
            obs_flux_mjy = getattr(config, 'OBSERVED_FLUX_1P3MM_MJY', 70.8)
            flux_ratio = flux_total_mjy / obs_flux_mjy if obs_flux_mjy > 0 else float('inf')
            
            self._log_info(f"📊 Integrated flux: {flux_total_mjy:.2f} mJy (observed: {obs_flux_mjy:.1f} mJy, ratio: {flux_ratio:.3f})")
            
            # 6. IMAGE ALREADY HAS CORRECT PA (from posang parameter)
            # ========================================================================
            # RADMC-3D generated image with posang={posang_value}° (PA={PA_OBS_DEG}°)
            # No post-processing rotation needed!
            # Image already has PA ~ {PA_OBS_DEG}° on sky
            # ========================================================================
            
            self._log_info(f"Final model image: {image_data.shape}")
            self._log_info(f"  PA on sky: ~{PA_OBS_DEG}° (set by posang={posang_value}°)")
            self._log_info(f"  ✓ No post-rotation needed")
            
            # Debug FITS
            if self._cleanup is False:
                hdu = fits.PrimaryHDU(image_data)
                hdu.header['OBJECT'] = 'IRAS 04166 Model'
                hdu.header['INCL'] = INCLINATION_DEG
                hdu.header['PA_MODEL'] = PA_OBS_DEG
                hdu.header['POSANG'] = posang_value
                hdu.header['BEAM_MAJ'] = BEAM_MAJOR_ARCSEC
                hdu.header['BEAM_MIN'] = BEAM_MINOR_ARCSEC
                hdu.header['BEAM_PA'] = BEAM_PA_DEG
                hdu.header['COMMENT'] = 'PA set directly via RADMC-3D posang parameter'
                hdu.writeto(radmc_dir / "debug_model_v7_posang.fits", overwrite=True)

            return image_data

        except subprocess.CalledProcessError as e:
            self._log_error(f"RADMC3D Failed: {e}")
            return None
        except Exception as e:
            self._log_error(f"Error: {e}", exception=e)
            return None
        finally:
            os.chdir(original_dir)
    
    def _fallback_radmc3d_image(self, radmc_dir: Path) -> Optional[np.ndarray]:
        """
        Fallback to subprocess if radmc3dPy not available.
        This won't have proper beam convolution!
        Uses post-generation rotation to match PA.
        """
        self._log_debug("Using fallback RADMC3D image generation (NO CONVOLUTION)")
        
        try:
            # Generate image with correct posang
            # posang = PA_obs - 90° to match observation PA
            posang_value = PA_OBS_DEG - 90.0
            image_cmd = (
                f"{self.radmc3d_exec} image "
                f"npix {IMAGE_NPIX} "
                f"sizeau {IMAGE_SIZE_AU / 2} "
                f"incl {INCLINATION_DEG} "
                f"phi 0 "
                f"posang {posang_value} "
                f"lambda {WAV_MICRON}"
            )
            
            result_image = subprocess.run(
                image_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=SIMULATION_TIMEOUT,
                cwd=radmc_dir
            )
            
            if result_image.returncode != 0:
                self._log_error(f"RADMC3D image stderr: {result_image.stderr}")
                return None
            
            # Read image.out
            image_file = radmc_dir / "image.out"
            if not image_file.exists():
                return None
            
            # Parse RADMC3D image.out format
            with open(image_file, 'r') as f:
                iformat = int(f.readline())
                nx, ny = map(int, f.readline().split())
                nlam = int(f.readline())
                
                # Skip pixel size header (2 lines)
                f.readline()
                f.readline()
                
                # Skip wavelength(s)
                for _ in range(nlam):
                    f.readline()
                
                # Skip empty line
                line = f.readline().strip()
                while not line:  # Skip all empty lines
                    line = f.readline().strip()
                
                # Read first data value (already got it)
                image_data = [float(line)]
                
                # Read rest of image data
                for _ in range(nx * ny - 1):
                    image_data.append(float(f.readline()))
                
                image_data = np.array(image_data)
                image_data = image_data.reshape((ny, nx))
            
            # Unit conversion: CGS (erg/s/cm²/Hz/sr) → Jy/sr → Jy/pixel
            pixel_scale_as = (IMAGE_SIZE_AU / IMAGE_NPIX) / DISTANCE_PC  # arcsec/pixel
            pixel_scale_rad = pixel_scale_as * (np.pi / 648000.0)         # rad/pixel
            pixel_sr = pixel_scale_rad ** 2                                # sr/pixel
            image_data_jy_per_pix = image_data * 1e23 * pixel_sr          # erg/s/cm²/Hz/sr → Jy/pixel

            beam_sigma_x = (BEAM_MAJOR_ARCSEC / 2.355) / pixel_scale_as  # pixels
            beam_sigma_y = (BEAM_MINOR_ARCSEC / 2.355) / pixel_scale_as  # pixels
            theta_rad = np.deg2rad(BEAM_PA_DEG)  # Beam position angle rotation (22°)
            
            kernel = Gaussian2DKernel(x_stddev=beam_sigma_x, y_stddev=beam_sigma_y, 
                                     theta=theta_rad, x_size=31, y_size=31)
            convolved_jy_per_pix = convolve(image_data_jy_per_pix, kernel, boundary='extend')

            # Convert Jy/pixel → Jy/beam
            # Jy/beam = Jy/pixel × (beam_area / pixel_area)
            bmaj_rad = BEAM_MAJOR_ARCSEC * (np.pi / 648000.0)
            bmin_rad = BEAM_MINOR_ARCSEC * (np.pi / 648000.0)
            beam_sr = (np.pi / (4.0 * np.log(2.0))) * bmaj_rad * bmin_rad
            convolved = convolved_jy_per_pix * (beam_sr / pixel_sr)  # Corrected: multiply by ppb
            
            # Rotation already done by posang parameter in radmc3d command
            # No post-processing rotation needed
            self._log_info(f"Fallback: Image generated with posang={posang_value}° → PA ~ {PA_OBS_DEG}°")
            
            return convolved
            
        except Exception as e:
            self._log_error(f"Fallback RADMC3D failed: {e}", exception=e)
            return None




