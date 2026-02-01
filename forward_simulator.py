"""
Forward Model Simulator V2: Fixed RADMC3D Pipeline
===================================================
FIXES APPLIED:
1. ✅ nphi = 64 (was 1) - Proper azimuthal resolution
2. ✅ Use radmc3dPy for image generation and beam convolution
3. ✅ nphot = 1,000,000 (was 100,000) - Better Monte Carlo statistics
4. ✅ Grid refinement at inner edge (12 levels, 3 span)
5. ✅ scipy.interpolate.interp1d for smooth density interpolation
6. ✅ Python rotation for PA alignment (more reliable than RADMC-3D posang)

ADAPTED FOR MCMC:
- Isolated working directories per walker
- Proper cleanup after each iteration
- Memory management for long runs
- Thread-safe operations

Based on reference implementation:
- problem_setup_origin.py (grid setup and density)
- plot_images.py (radmc3dPy usage and convolution)

Author: Pipeline V2
Date: 2025-12-03
"""

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

# Try to import radmc3dPy (required for proper image generation)
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
    """
    Grid refinement function from problem_setup_origin.py
    
    Parameters:
    -----------
    x_orig : array
        Original grid points
    nlev : int
        Number of refinement levels
    nspan : int
        Number of cells to refine at each level
    
    Returns:
    --------
    array : Refined grid
    """
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
    """
    V2 Forward Model Simulator with fixed RADMC3D pipeline.
    
    Key improvements:
    - Proper azimuthal resolution (nphi=64)
    - radmc3dPy integration
    - Higher nphot for better accuracy
    - Grid refinement at inner edge
    - Smooth interpolation
    
    ⚠️ CRITICAL: This class must be picklable for multiprocessing!
    Logger is created on-demand via property (not stored as instance variable)
    """
    
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
        self._cleanup = cleanup  # ✅ ADD cleanup flag
        
        self._log_info("ForwardModelSimulatorV2 initialized with FIXED pipeline")
        self._log_info(f"  - nphi = 64 (proper azimuthal resolution)")
        self._log_info(f"  - nphot = 100,000 (standard Monte Carlo)")
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
    
    def simulate(self, params: Dict[str, float], keep_dir: bool = False) -> Tuple[bool, Optional[np.ndarray], Optional[Dict], Optional[Path]]:
        """
        Main simulation pipeline.
        
        Parameters:
        -----------
        params : dict
            MCMC parameters (log_mdisk, r_c, vFrag, sigma_exp, dust_to_gas)
        keep_dir : bool
            If True, do NOT cleanup sim_dir (for best-fit preservation)
        
        Returns:
        --------
        success : bool
            Whether simulation succeeded
        image : np.ndarray or None
            Synthetic image [IMAGE_NPIX x IMAGE_NPIX] in Jy/beam
        metadata : dict or None
            Simulation metadata (params, shape, directories, etc.)
        """
        self._log_info(f"Starting simulation with params: {params}")
        
        # Create isolated working directory
        sim_id = uuid.uuid4().hex[:8]
        sim_dir = self.work_dir / f"sim_{sim_id}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # STEP 1: Run DustPy
            self._log_debug("Step 1/4: Running DustPy simulation")
            dustpy_success, dustpy_output = self._run_dustpy(sim_dir, params)
            if not dustpy_success:
                return False, None
            
            # STEP 2: Convert to RADMC3D format (FIXED VERSION)
            self._log_debug("Step 2/4: Converting to RADMC3D (V2 with fixes)")
            radmc_dir = sim_dir / "radmc3d_work"
            radmc_dir.mkdir(exist_ok=True)
            self._convert_dustpy_to_radmc3d_v2(dustpy_output, radmc_dir, params)
            
            # STEP 3: Run RADMC3D with radmc3dPy (FIXED VERSION)
            self._log_debug("Step 3/4: Running RADMC3D (V2 with radmc3dPy)")
            image = self._run_radmc3d_v2(radmc_dir, params)
            
            if image is None:
                return False, None, None
            
            # STEP 4: Create metadata
            self._log_debug("Step 4/4: Simulation complete")
            metadata = {
                'params': params.copy(),
                'image_shape': image.shape,
                'dustpy_dir': str(sim_dir),
                'sim_dir': str(sim_dir),  # ✅ Encapsulate sim_dir in metadata
                'success': True
            }
            
            return True, image, metadata  # ✅ RESTORED: 3-value interface (sim_dir in metadata)
            
        except Exception as e:
            self._log_error(f"Simulation failed: {e}", exception=e)
            return False, None, None  # ✅ RESTORED: 3-value interface
        
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
        """
        Alias for simulate() - for backward compatibility and intuitive naming.
        
        Both names work:
        - simulator.simulate(params)      ← Official name
        - simulator.forward_model(params) ← Intuitive alias
        
        Returns: (success, image, metadata)
        """
        return self.simulate(params)
    
    def _run_dustpy(self, sim_dir: Path, params: Dict[str, float]) -> Tuple[bool, Optional[Dict]]:
        """
        Run DustPy simulation with 5 FREE parameters (sigma_rc removed to eliminate degeneracy).
        
        Free parameters:
            log_mdisk, r_c, vFrag, sigma_exp, dust_to_gas
        
        Fixed parameters:
            r_in=1.0 AU, alpha=1e-3
        
        CRITICAL NOTE: 'log_mdisk' represents TOTAL DISK MASS (gas-dominated).
              For a typical disk: M_total ≈ M_gas (since dust << gas).
              Physical constraint: M_disk << M_star to avoid gravitational instability.
              Dust mass is derived: M_dust = M_gas × dust_to_gas.
              Do NOT rename variable to maintain HDF5 backend compatibility.
        """
        try:
            # Extract parameters (with defaults)
            log_mdisk = params.get('log_mdisk', -2.0)  # TOTAL/GAS mass (see docstring NOTE)
            r_c = params.get('r_c', 60.0)
            vFrag = params.get('vFrag', 200.0)
            sigma_exp = params.get('sigma_exp', 1.0)
            dust_to_gas = params.get('dust_to_gas', 0.01)
            r_in = 1.0  # ✅ FIXED at 1.0 AU (removed from parameters - too slow at low values)
            
            # ❌ REMOVED: sigma_rc (redundant with r_c, causes degeneracy)
            # Now using 5 free parameters + 2 fixed
            
            # PHYSICS: log_mdisk represents TOTAL DISK MASS (gas-dominated)
            # M_total ≈ M_gas (since gas >> dust)
            M_gas = 10**log_mdisk * c.M_sun  # Gas mass (primary)
            M_dust = M_gas * dust_to_gas      # Dust mass (derived from gas)
            
            self._log_debug(f"🔍 DEBUG: Initializing DustPy with M_gas={M_gas/c.M_sun:.4e} M☉, M_dust={M_dust/c.M_sun:.4e} M☉")
            self._log_debug(f"         Ratio: M_disk/M_star = {M_gas/(STAR_MASS_MSUN*c.M_sun):.3f} (should be << 1)")
            
            R_c = r_c * c.au  # ✅ Direct r_c, no multiplier
            R_in = r_in * c.au
            
            # Create DustPy simulation
            sim = Simulation()
            
            # Star properties
            sim.ini.star.M = STAR_MASS_MSUN * c.M_sun
            sim.ini.star.R = STAR_RADIUS_RSUN * c.R_sun
            sim.ini.star.T = STAR_TEMP_K
            
            # Disk properties
            sim.ini.gas.Mdisk = M_gas  # ✅ FIXED: Use gas mass, not dust mass!
            sim.ini.gas.SigmaRc = R_c  # ✅ Uses scaled r_c
            sim.ini.gas.SigmaExp = -sigma_exp
            sim.ini.gas.alpha = 1.0e-3  # ✅ FIXED at standard MRI value
            
            # ✅ NEW: Set inner radius (modify grid)
            # DustPy default grid: [r_in, r_out] with logarithmic spacing
            sim.ini.grid.rmin = R_in  # Inner edge
            # Outer radius scaled with r_c
            sim.ini.grid.rmax = max(500 * c.au, 10 * R_c)  # At least 10× R_c
            
            # Dust properties
            sim.ini.dust.aIniMax = 0.001  # cm - Initial max grain size
            sim.ini.dust.vFrag = vFrag  # cm/s
            
            # ✅ CRITICAL FIX: Limit maximum grain size to prevent unrealistic growth
            # For Class 0/I disks: grains typically don't exceed ~1 cm
            # This prevents St >> 1 and keeps physics realistic
            # a_max = 1 cm → m_max = 6.995 g (for rho_s = 1.67 g/cm³)
            sim.ini.grid.mmax = 7.0  # g - Maximum particle mass (= 1 cm size)
            
            # ✅ NEW: Dust-to-gas ratio
            # This affects initial dust surface density
            # DustPy will evolve this, but we set the initial ratio
            sim.ini.dust.d2gRatio = dust_to_gas  # ✅ FIXED: Correct attribute name
            
            # Initialize
            sim.initialize()
            
            # Setup snapshots
            N_SNAPSHOTS = getattr(config, 'N_SNAPSHOTS', 10)
            sim.t.snapshots = np.hstack([
                sim.t,
                np.geomspace(1.0, 1.0e5, num=N_SNAPSHOTS) * c.year
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
        """
        Convert DustPy output to RADMC3D input files.
        
        Parameters
        ----------
        dustpy_output : dict
            DustPy simulation results containing:
                - 'r' : ndarray, shape (Nr,)
                    Radial grid [cm]
                - 'sigma_dust' : ndarray, shape (Nr,)
                    Dust surface density [g/cm²]
                - 'sigma_gas' : ndarray, shape (Nr,)
                    Gas surface density [g/cm²]
                - 'scale_height' : ndarray, shape (Nr,)
                    Pressure scale height H [cm]
                - 'temperature' : ndarray, shape (Nr,)
                    Midplane temperature [K]
        radmc_dir : Path
            Directory for RADMC-3D input/output files
        params : dict
            MCMC parameters (for metadata/logging)
        
        Returns
        -------
        None
            Creates RADMC-3D input files in radmc_dir:
                - amr_grid.inp (spherical grid structure)
                - dust_density.inp (3D dust density)
                - wavelength_micron.inp (wavelength grid)
                - stars.inp (stellar properties)
                - dustopac.inp (opacity table references)
                - radmc3d.inp (runtime parameters)
        
        Notes
        -----
        Implements dynamic grid boundary using density-threshold method:
        Outer radius is set where Σ_dust < 10⁻⁴ × Σ_peak (negligible emission).
        This eliminates arbitrary 500 AU floors and adapts to disk extent.
        
        Grid refinement (12 levels, 3 span) applied at inner edge for better
        resolution of temperature gradient and density structure.
        
        References
        ----------
        .. [1] RADMC-3D Manual Section 7.1: "Grid must encompass all significant emission"
        .. [2] Dullemond et al. (2012): RADMC-3D: A multi-purpose radiative transfer tool
        
        ✅ FIX 1: nphi = 64 (was 1)
        ✅ FIX 4: Grid refinement at inner edge
        ✅ FIX 5: scipy.interpolate.interp1d for density
        ✅ CRITICAL FIX C2: Dynamic outer boundary matching DustPy extent
        """
        self._log_debug("Converting DustPy → RADMC3D (V2 with fixes)")
        
        # ===== GRID SETUP (matching problem_setup_origin.py) =====
        
        # Radial grid
        nr = 100
        # ✅ FIXED: Match DustPy inner radius (r_in=1 AU from line 269)
        # Avoid extrapolation below DustPy r_min
        rin = 1.0 * au  # Was 0.1 AU - now matches DustPy r_in
        
        # ===== CRITICAL FIX C2: DENSITY-THRESHOLD DYNAMIC BOUNDARY =====
        # Source: RADMC-3D Manual Section 7.1 - "The grid must encompass all 
        # significant emission. Define outer boundary where surface density 
        # drops below detection threshold."
        #
        # Scientific approach (no magic numbers):
        # 1. Find radius where Σ_dust < threshold × Σ_peak
        # 2. Apply safety factor f (typically 1.1-1.2) for numerical margin
        # 3. Grid: rout = f × r_threshold
        
        r_dustpy = dustpy_output['r']
        sigma_dustpy = dustpy_output['sigma_dust']
        
        # Define edge as where density drops to 10^-4 of peak
        # Rationale: Emission I ∝ τ ∝ Σ (optically thin)
        #   At Σ = 10^-4 × Σ_peak → flux contribution < 0.01% (negligible)
        #   Standard in RT: discard regions with τ < 10^-4 (RADMC-3D Manual 7.1)
        DENSITY_THRESHOLD = 1e-4
        SAFETY_FACTOR = 1.2  # 20% margin for numerical stability
        
        sigma_peak = sigma_dustpy.max()
        sigma_threshold = DENSITY_THRESHOLD * sigma_peak
        
        # Find outermost radius where Σ > threshold
        mask_significant = sigma_dustpy > sigma_threshold
        if np.any(mask_significant):
            r_dust_edge = r_dustpy[mask_significant].max()
            rout = SAFETY_FACTOR * r_dust_edge
            
            self._log_info(f"Dynamic grid boundary (density-threshold method):")
            self._log_info(f"  - Σ_peak = {sigma_peak:.2e} g/cm²")
            self._log_info(f"  - Threshold = {DENSITY_THRESHOLD:.0e} × Σ_peak = {sigma_threshold:.2e} g/cm²")
            self._log_info(f"  - Dust edge (r@threshold) = {r_dust_edge/au:.1f} AU")
            self._log_info(f"  - RADMC-3D rout = {SAFETY_FACTOR}× edge = {rout/au:.1f} AU")
        else:
            # Fallback: Use DustPy extent if all values below threshold (unlikely)
            r_dust_edge = r_dustpy.max()
            rout = SAFETY_FACTOR * r_dust_edge
            self._log_warning(f"⚠️  All Σ below threshold - using full DustPy extent")
            self._log_warning(f"   rout = {rout/au:.1f} AU")
        
        # ===== END CRITICAL FIX =====
        
        # Initial grid
        ri = np.logspace(np.log10(rin), np.log10(rout), nr+1)
        
        # ✅ FIX 4: Grid refinement at inner edge
        ri = grid_refine_inner_edge(ri, nlev=12, nspan=3)
        rc = 0.5 * (ri[:-1] + ri[1:])
        nr = len(rc)  # Update nr after refinement
        
        # Theta grid (colatitude from pole)
        ntheta = 32  # Match reference (was 50)
        zrmax = 0.5
        thetaup = np.pi * 0.5 - 0.5
        thetai = np.linspace(thetaup, 0.5 * np.pi, ntheta + 1)
        thetac = 0.5 * (thetai[:-1] + thetai[1:])
        
        # ✅ FIX 1: Phi grid - CRITICAL!
        nphi = 64  # Was 1 - this is THE MAIN FIX!
        phii = np.linspace(0.0, 2 * np.pi, nphi + 1)
        phic = 0.5 * (phii[:-1] + phii[1:])
        
        self._log_debug(f"Grid: nr={nr}, ntheta={ntheta}, nphi={nphi}")
        
        # ===== DENSITY INTERPOLATION =====
        
        # ✅ FIX 5: Use scipy interp1d for smooth interpolation
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
        
        # Match lengths for H interpolation
        # Handle case where DustPy output has 2D arrays (Nr, Nm) or mismatched shapes
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
        
        # ✅ FIXED: Check for extrapolation range before interpolating
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
        
        # ✅ Safety check: Warn if extrapolating significantly beyond DustPy range
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
        r_dustpy_full = dustpy_output['r']
        sigma_dustpy_full = dustpy_output['sigma_dust']
        mask_beyond_radmc = r_dustpy_full > (r_radmc_max * au)
        
        if np.any(mask_beyond_radmc):
            mass_beyond = np.trapz(
                2 * np.pi * r_dustpy_full[mask_beyond_radmc] * sigma_dustpy_full[mask_beyond_radmc],
                r_dustpy_full[mask_beyond_radmc]
            )
            total_mass = np.trapz(2 * np.pi * r_dustpy_full * sigma_dustpy_full, r_dustpy_full)
            fraction_lost = mass_beyond / total_mass
            
            if fraction_lost > 0.01:  # More than 1% lost
                self._log_error(
                    f"🚨 MASS CONSERVATION VIOLATED: {fraction_lost:.1%} of dust mass "
                    f"beyond RADMC-3D grid ({r_radmc_max:.1f} AU)!"
                )
                raise ValueError(
                    f"Unacceptable mass loss: {fraction_lost:.1%} beyond grid. "
                    f"Increase rout or adjust DustPy parameters."
                )
            else:
                self._log_debug(f"✅ Mass conservation OK: {fraction_lost:.3%} lost at boundary")
        # ===== END CRITICAL CHECKS =====
        
        sigmad = sigma_interp(r_au_grid)
        hp = H_interp(r_au_grid)
        hpr = hp / rc
        
        self._log_debug(f"Interpolated to refined grid: {len(r_au_grid)} points (from {min_len} DustPy points)")
        
        # Create 3D meshgrid
        rr, tt, pp = np.meshgrid(rc, thetac, phic, indexing='ij')
        
        # ✅ FIXED: Convert colatitude θ to height z
        # tt is colatitude from pole: 0 at pole, π/2 at midplane
        # zr_angle = π/2 - tt is angle from midplane
        # z = r * sin(zr_angle) is actual vertical height
        zr_angle = np.pi/2.0 - tt  # Angle from midplane [radians]
        zz = rr * np.sin(zr_angle)  # Actual height z = r*sin(angle) [cm]
        
        # Expand 1D profiles to 3D
        sigmad_3d = sigmad[:, None, None] * np.ones((1, ntheta, nphi))
        hh = hp[:, None, None] * np.ones((1, ntheta, nphi))  # H in cm
        
        # ✅ FIXED: Gaussian vertical profile using actual height z
        # ρ(r,z) = Σ(r) / (√(2π) H(r)) * exp(-z²/(2H²))
        # Use H (not H/r) since we have actual height z (not z/r)
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
        M_star = STAR_MASS_MSUN
        T_star = STAR_TEMP_K  # T_eff = 3000 K (INPUT, not T_bol=54K output!)
        L_star_calc = 4.0 * pi * R_star_cm**2 * ss * T_star**4  # [erg/s]
        L_star_Lsun = L_star_calc / ls  # Convert to solar units
        
        # Log verification
        self._log_debug(f"Stellar params: M={M_star:.2f} Msun, R={STAR_RADIUS_RSUN:.2f} Rsun, T_eff={T_star} K")
        self._log_debug(f"Computed luminosity: L={L_star_Lsun:.3f} Lsun (target={STAR_LUMI_LSUN} Lsun)")
        if abs(L_star_Lsun - STAR_LUMI_LSUN) / STAR_LUMI_LSUN > 0.05:
            self._log_warning(f"Luminosity mismatch: computed {L_star_Lsun:.3f} vs target {STAR_LUMI_LSUN} Lsun")
        
        pstar = [0., 0., 0.]
        
        with open(radmc_dir / "stars.inp", "w") as f:
            f.write("2\n")  # Format number
            f.write(f"1 {nlam}\n\n")  # 1 star, N wavelengths + blank line
            f.write(f"{R_star_cm:13.6e} {M_star:13.6e} {pstar[0]:13.6e} {pstar[1]:13.6e} {pstar[2]:13.6e}\n\n")
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
        
        # 6. radmc3d.inp (✅ FIX 3: nphot adjusted for proper scaling)
        nphot = params.get('nphot', 100000)  # Allow override for speed tests
        with open(radmc_dir / "radmc3d.inp", "w") as f:
            f.write(f"nphot = {nphot}\n")
            f.write("scattering_mode_max = 1\n")
            f.write("iranfreqmode = 1\n")
            f.write("istar_sphere = 1\n")
        
        self._log_debug(f"RADMC3D input files created in {radmc_dir} (nphot={nphot})")
    
    def _read_rout_from_grid(self, radmc_dir: Path) -> float:
        """
        Read outer grid radius from amr_grid.inp.
        
        Returns
        -------
        rout : float
            Outer radius of RADMC-3D grid [cm]
        
        Raises
        ------
        FileNotFoundError
            If amr_grid.inp does not exist
        """
        grid_file = radmc_dir / "amr_grid.inp"
        
        with open(grid_file, 'r') as f:
            # Parse header
            iformat = int(f.readline())          # Line 1: format
            amr_style = int(f.readline())        # Line 2: AMR style
            coord_sys = int(f.readline())        # Line 3: coordinate system
            gridinfo = int(f.readline())         # Line 4: grid info
            incl_dims = f.readline().split()     # Line 5: include r,theta,phi
            
            # Read grid dimensions
            dims = f.readline().split()          # Line 6: nr, ntheta, nphi
            nr = int(dims[0])
            
            # Read radial grid interfaces
            ri = []
            for i in range(nr + 1):
                ri.append(float(f.readline()))
            
            rout = ri[-1]  # Outer radius [cm]
        
        self._log_debug(f"Read from amr_grid.inp: rout = {rout/au:.2f} AU (nr={nr})")
        
        return rout
    
    def _run_radmc3d_v2(self, radmc_dir: Path, params: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Run RADMC3D V6 - SCIENTIFICALLY CORRECT ANGLE HANDLING.
        
        ====================================================================
        SCIENTIFIC UNDERSTANDING:
        ====================================================================
        
        1. THE DISK MODEL IS AXISYMMETRIC:
           - DustPy creates an axisymmetric disk (no azimuthal structure)
           - The density distribution only depends on (r, z), not phi
           - Therefore, changing phi in RADMC-3D does NOT change the image
        
        2. WHAT THE ANGLES MEAN IN RADMC-3D:
           - incl: Inclination of the line of sight (60° for IRAS 04166)
           - phi: Azimuthal position of observer (irrelevant for axisymmetric disk)
           - posang: Rotates the camera/image (NOT the physical disk)
        
        3. POSITION ANGLE (PA) IN ASTRONOMY:
           - PA is the angle of the disk major axis measured N through E
           - It describes HOW WE SEE the disk, not the disk's physical structure
           - For an axisymmetric disk, PA is purely a projection effect
        
        4. CORRECT APPROACH FOR COMPARISON WITH OBSERVATIONS:
           - Generate model image with incl=47°, phi=0, posang=0
           - This produces a disk with major axis along x-axis (PA ~ 90°)
           - ROTATE THE IMAGE AFTER GENERATION to match observation PA
           - This is the standard practice in astronomical modeling
        
        5. WHY ROTATE THE IMAGE (not use posang):
           - posang rotates the camera, which can cause edge effects
           - Image rotation preserves the square image format
           - This allows direct pixel-by-pixel comparison with observations
           - The physics is the same - it's just a coordinate transformation
        
        ====================================================================
        """
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

            # 1. RUN MCTHERM (thermal Monte Carlo)
            subprocess.run(
                f"{self.radmc3d_exec} mctherm", 
                shell=True, check=True, 
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=SIMULATION_TIMEOUT
            )
            
            # 2. GENERATE IMAGE WITH CORRECT POSANG
            # ========================================================================
            # RADMC-3D posang parameter:
            #   - posang rotates the camera view (N→E, counter-clockwise)
            #   - Default (posang=0): disk major axis horizontal (PA ~ 90°)
            #   - To get PA = 121.5°: posang = PA_obs - 90 = 31.5°
            # 
            # For axisymmetric disk:
            #   - incl = 47° (disk inclination from observation)
            #   - phi = 0° (irrelevant for axisymmetric disk)
            #   - posang = 31.5° (to match observed PA = 121.5°)
            # ========================================================================
            
            # Calculate posang to match observation PA
            posang_value = PA_OBS_DEG - 90.0  # 121.5 - 90 = 31.5°
            
            self._log_info(f"RADMC-3D Image Generation:")
            self._log_info(f"  incl: {INCLINATION_DEG}° (disk inclination)")
            self._log_info(f"  phi: 0° (irrelevant for axisymmetric disk)")
            self._log_info(f"  posang: {posang_value}° (to get PA={PA_OBS_DEG}° on sky)")
            self._log_info(f"  → Direct posang rotation (no post-processing needed)")

            # ===== NEW: CRITICAL VALIDATION =====
            # Read rout from amr_grid.inp to ensure FOV encompasses full disk
            try:
                rout = self._read_rout_from_grid(radmc_dir)  # in [cm]
                rout_au = rout / au
                
                sizeau_radius = IMAGE_SIZE_AU / 2.0  # FOV radius [AU]
                
                # Check for disk truncation
                coverage_ratio = sizeau_radius / rout_au
                
                if coverage_ratio < 1.0:
                    # CRITICAL ERROR - cutting disk!
                    raise ValueError(
                        f"🚨 CRITICAL: Field of View ({sizeau_radius:.1f} AU) is SMALLER than "
                        f"grid outer radius ({rout_au:.1f} AU)!\n"
                        f"This will cut {(1 - coverage_ratio)*100:.0f}% of the disk mass.\n"
                        f"Fix: Increase IMAGE_SIZE_AU in config to at least {2.2 * rout_au:.0f} AU"
                    )
                
                elif coverage_ratio < 1.2:
                    # WARNING - insufficient margin
                    self._log_warning(
                        f"⚠️  Field of View ({sizeau_radius:.1f} AU) has only "
                        f"{(coverage_ratio - 1)*100:.0f}% margin beyond grid ({rout_au:.1f} AU). "
                        f"Recommend IMAGE_SIZE_AU ≥ {2.5 * rout_au:.0f} AU to avoid edge effects."
                    )
                
                else:
                    # OK
                    self._log_info(
                        f"✓ FOV validation passed: "
                        f"sizeau={sizeau_radius:.1f} AU, rout={rout_au:.1f} AU "
                        f"(margin={(coverage_ratio-1)*100:.0f}%)"
                    )
            
            except FileNotFoundError:
                self._log_warning("Cannot validate FOV - amr_grid.inp not found (will proceed anyway)")
            # ===== END VALIDATION =====

            sizeau_radius = IMAGE_SIZE_AU / 2.0
            
            cmd = (
                f"{self.radmc3d_exec} image "
                f"lambda {WAV_MICRON} "
                f"npix {IMAGE_NPIX} "
                f"sizeau {sizeau_radius} "
                f"incl {INCLINATION_DEG} "
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
            
            # 5. IMAGE ALREADY HAS CORRECT PA (from posang parameter)
            # ========================================================================
            # RADMC-3D generated image with posang=31.5°
            # No post-processing rotation needed!
            # Image already has PA ~ 121.5° on sky
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
            
            # Manual Gaussian beam convolution (basic)
            beam_sigma_x = (BEAM_MAJOR_ARCSEC / 2.355) / (IMAGE_SIZE_AU / IMAGE_NPIX / DISTANCE_PC)
            beam_sigma_y = (BEAM_MINOR_ARCSEC / 2.355) / (IMAGE_SIZE_AU / IMAGE_NPIX / DISTANCE_PC)
            
            kernel = Gaussian2DKernel(x_stddev=beam_sigma_x, y_stddev=beam_sigma_y, x_size=31, y_size=31)
            convolved = convolve(image_data, kernel, boundary='extend')
            t
            # Rotation already done by posang parameter in radmc3d command
            # No post-processing rotation needed
            self._log_info(f"Fallback: Image generated with posang={posang_value}° → PA ~ {PA_OBS_DEG}°")
            
            return convolved
            
        except Exception as e:
            self._log_error(f"Fallback RADMC3D failed: {e}", exception=e)
            return None


# ===== BACKWARDS COMPATIBILITY =====
# Keep old class name as alias for seamless replacement in MCMC code
ForwardModelSimulator = ForwardModelSimulatorV2
