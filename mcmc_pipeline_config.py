"""
Configuration file for MCMC Pipeline
=====================================
Định nghĩa tất cả parameters, paths, và settings cho dự án MCMC fitting.

Author: Pipeline Builder
Date: 2025-11-19
"""

import os
import sys
import numpy as np
from pathlib import Path
import shutil

# =============================================================================
# PATHS & DIRECTORIES (PORTABLE CONFIGURATION)
# =============================================================================
# Uses environment variables with fallbacks for server/cluster deployment

# Base directory: Try env var, then fall back to script location
BASE_DIR = os.getenv(
    'MCMC_PIPELINE_BASE',
    str(Path(__file__).parent.absolute())
)

# Work directory: Try env var, then relative to base
WORK_DIR = os.getenv(
    'MCMC_WORK_DIR',
    os.path.join(BASE_DIR, "16-9-2025_dustpy_radmc3d_change_init", "WORK_DIR")
)

# Output directories
MCMC_OUTPUT_DIR = os.getenv(
    'MCMC_OUTPUT_DIR',
    os.path.join(BASE_DIR, "mcmc_results")
)
CHECKPOINT_DIR = os.path.join(MCMC_OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(MCMC_OUTPUT_DIR, "logs")

# Observation data: Try env var, then search standard locations
def _find_obs_fits():
    """Search for observation FITS in standard locations."""
    # First try environment variable
    env_path = os.getenv('MCMC_OBS_FITS')
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Search in standard locations
    search_paths = [
        os.path.join(BASE_DIR, "IRAS04166_robust_0.0.fits"),  # In MCMC directory
        os.path.join(os.path.dirname(BASE_DIR), "IRAS04166_robust_0.0.fits"),  # Parent directory
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        f"IRAS04166_robust_0.0.fits not found. Searched:\n" +
        "\n".join(f"  - {p}" for p in search_paths) +
        "\nSet MCMC_OBS_FITS environment variable or place file in one of these locations."
    )

OBS_FITS_PATH = _find_obs_fits()

# RADMC-3D executable: Search common locations
def _find_radmc3d():
    """Search for radmc3d executable in common locations."""
    search_paths = [
        os.getenv('RADMC3D_EXECUTABLE'),  # User-specified env var
        os.path.join(os.path.expanduser('~'), 'bin', 'radmc3d'),  # ~/bin
        '/usr/local/bin/radmc3d',  # System install
        '/opt/bin/radmc3d',  # Alternative location
        'radmc3d',  # In PATH
    ]
    
    for path in search_paths:
        if path and shutil.which(path):
            return path
    
    # Fallback for backward compatibility
    legacy_path = "/home/nam/bin/radmc3d"
    if os.path.exists(legacy_path):
        return legacy_path
    
    raise FileNotFoundError(
        "radmc3d executable not found. Install RADMC-3D or set RADMC3D_EXECUTABLE environment variable."
    )

RADMC3D_EXECUTABLE = _find_radmc3d()

# Dust opacity table (RADMC3D requires this file in each run directory)
# Try current directory first, then parent directory (for typical project layout)
def _find_opacity_file():
    """Search for opacity file in standard locations."""
    search_paths = [
        os.path.join(BASE_DIR, "dustkappa_silicate.inp"),  # In MCMC directory
        os.path.join(os.path.dirname(BASE_DIR), "dustkappa_silicate.inp"),  # Parent directory
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        f"dustkappa_silicate.inp not found. Searched:\n" +
        "\n".join(f"  - {p}" for p in search_paths) +
        "\nPlease place dustkappa_silicate.inp in one of these locations."
    )

RADMC3D_OPACITY_FILE = _find_opacity_file()

# Templates - REMOVED: DustPy creates Simulation() directly, no template needed
# TEMPLATE_INI_PATH = os.path.join(WORK_DIR, "sim_template.ini")

# Create directories if not exist
for d in [WORK_DIR, MCMC_OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# OBSERVATION PARAMETERS
# =============================================================================

# Source properties (from Phuong+2025 ApJ paper)
DISTANCE_PC = 156.0  # Distance to IRAS04166 in parsecs (Gaia EDR3)
INCLINATION_DEG = 47.0  # Disk inclination in degrees (from cos⁻¹(94/138))

# ═══════════════════════════════════════════════════════════════════════════
# ⚠️  CRITICAL: POSITION ANGLE COORDINATE TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════
# Observed PA (astronomical): 121.5° ± 0.4° (Table 3, Phuong+2025)
# Measured counter-clockwise from North toward East (IAU convention)
PA_OBS_DEG = 121.5

# RADMC-3D phi parameter conversion:
# The makeImage(phi=φ) parameter rotates the CAMERA azimuthally, NOT the disk.
# Mathematical derivation (see GEOMETRIC_FIX_REPORT.md):
#   - phi=0°  → Disk major axis horizontal (along RA)
#   - phi=90° → Disk major axis vertical (along Dec)
#   - For PA measured from Dec (North) toward RA (East):
#       phi = 90° - PA_obs
#
# Application:
#   phi = 90° - 121.5° = -31.5° ≡ 328.5° (mod 360°)
#
# Physical interpretation:
#   Camera at azimuthal angle 328.5° looking toward disk center
#   produces image with major axis at PA=121.5° on sky
POSITION_ANGLE_DEG = 31.5  # = 328.5°

# Image properties
# ========== IMAGE PARAMETERS (FIXED) ==========
# Scientific justification:
#   - Dynamic grid boundary rout ~ 70-100 AU (from DustPy)
#   - Need margin: FOV ≥ 1.5 × rout_max for safety
#   - Recommended: FOV = 2.5 × rout_max to avoid edge effects
#
# Calculation:
#   rout_max ≈ 100 AU (observed from logs)
#   → IMAGE_SIZE_AU = 2.5 × 100 = 250 AU
#
# Impact on pixel scale:
#   Old: 94 AU / 201 pixels = 0.47 AU/pixel
#   New: 250 AU / 201 pixels = 1.24 AU/pixel
#   (Still sufficient for 1.3mm ALMA beam ~10-15 AU)
#
# CRITICAL: IMAGE_SIZE_AU stores the FULL FOV DIAMETER (total width of image)
# RADMC3D's "sizeau" parameter expects HALF-WIDTH (radius: -sizeau to +sizeau)
# → Must pass IMAGE_SIZE_AU/2 to RADMC3D to get correct FOV!
IMAGE_SIZE_AU = 250.0  # [AU] - INCREASED from 94.0 to prevent disk truncation
NPIX = 201  # Number of pixels (201×201)
IMAGE_NPIX = NPIX  # Alias for compatibility
WAV_MICRON = 1300.0  # Observation wavelength in microns

# Physical disk size from paper (Table 3, Phuong+2025) - EXACT values
# These are NOT used in grid setup (grid is larger to include envelope)
DISK_MAJOR_ARCSEC = 0.1379  # Major axis: 137.9 ± 0.7 mas (Table 3)
DISK_MINOR_ARCSEC = 0.0937  # Minor axis: 93.7 ± 0.5 mas (Table 3)
DISK_RADIUS_MAJOR_AU = DISK_MAJOR_ARCSEC * DISTANCE_PC  # ~21.5 AU at 156 pc
DISK_RADIUS_MINOR_AU = DISK_MINOR_ARCSEC * DISTANCE_PC  # ~14.6 AU at 156 pc

# Beam properties (from Phuong+2025: "synthesized beam is 50 mas × 37 mas (P.A. = 22°)")
BEAM_MAJOR_ARCSEC = 0.05  # Major axis: 50 mas = 0.05 arcsec
BEAM_MINOR_ARCSEC = 0.037  # Minor axis: 37 mas = 0.037 arcsec (NOT 36.6!)
BEAM_PA_DEG = 22.0  # Beam position angle: 22° (exact from paper)

# Noise properties (from Phuong+2025 Table 1)
# Paper reports: 23 μJy/beam for continuum (robust=0.0)
RMS_NOISE_JY = 2.3e-05  # RMS noise level in Jy/beam (23 μJy/beam)


# =============================================================================
# DUSTPY PARAMETERS
# =============================================================================

N_SNAPSHOTS = 10  # Number of snapshots per DustPy run (matching your setup)
DUSTPY_TIMEOUT = 1200  # Maximum time for DustPy simulation in seconds (20 min)
DUSTPY_FINAL_TIME = 1.e5  # Final time in years (100,000 years NOT 300,000!)

# Star parameters (from Phuong+2025 ApJ paper)
# CRITICAL: T_bol=54K is OUTPUT (observational classification), NOT INPUT!
# For Class 0 protostar embedded in envelope:
STAR_MASS_MSUN = 0.27      # Solar masses (within paper range 0.15-0.39 Msun from Keplerian analysis)
STAR_MASS_MIN = 0.15       # Minimum stellar mass constraint [Msun] from paper Section 4.3
STAR_MASS_MAX = 0.39       # Maximum stellar mass constraint [Msun] from paper Section 4.3
STAR_TEMP_K = 3000         # Effective temperature [K] for low-mass protostar (INPUT to RADMC3D)
STAR_LUMI_LSUN = 0.41      # Bolometric luminosity [Lsun] from paper (Phuong+2025)
T_BOL_OBSERVED = 54        # Observed bolometric temperature [K] - EXPECTED OUTPUT from SED
# Radius calculated from L = 4πR²σT⁴: R = sqrt(L/(4πσT⁴))
# R = sqrt(0.41 * 3.8525e33 / (4π * 5.6703e-5 * 3000⁴)) = 2.38 Rsun (verified)
STAR_RADIUS_RSUN = 2.38    # Computed to match L_bol = 0.41 Lsun at T_eff = 3000 K

# Observational targets (for validation/comparison)
OBSERVED_FLUX_1P3MM_MJY = 70.8   # Integrated flux at 1.3mm [mJy] from Table 2
OBSERVED_FLUX_ERROR_MJY = 0.3    # Flux uncertainty [mJy]
DISK_MAJOR_AXIS_MAS = 137.9      # Deconvolved major axis [mas] from Table 2
DISK_MINOR_AXIS_MAS = 93.7       # Deconvolved minor axis [mas] from Table 2


# =============================================================================
# MCMC PARAMETERS TO FIT
# =============================================================================

# Định nghĩa các parameters cần fit
# Format: {"name": str, "min": float, "max": float, "default": float, "log_scale": bool}

MCMC_PARAMETERS = [
    {
        # ⚠️ CRITICAL: 'log_mdisk' represents TOTAL DISK (GAS) MASS in DustPy.
        # Physics correction (16 Jan 2025): M_gas = 10^log_mdisk, M_dust derived from d2g ratio.
        # Previous interpretation (dust mass) was INCORRECT and broke mass conservation.
        # DO NOT REVERT to dust mass interpretation - physics logic treats this as GAS mass.
        # Variable name kept as 'log_mdisk' for HDF5 backend compatibility with existing runs.
        "name": "log_mdisk",  # Log10 of TOTAL GAS MASS in solar masses (see WARNING above)
        "label": r"$\log_{10}(M_{\rm gas}/M_{\odot})$",  # ✅ CORRECTED LABEL
        "min": -3.5,          # ✅ CORRECTED: Realistic gas mass range (M_gas ≥ 0.000316 M☉)
        "max": -1.5,          # ✅ Upper bound ~0.0316 M☉ (widened for safety)
        "default": -2.8,      # ✅ FLUX-MATCHED: From high-fidelity test (M_gas = 0.00158 M☉)
        "log_scale": False,   # Already in log space
        "unit": "M_sun",
    },
    {
        "name": "r_c",  # Characteristic radius in AU
        "label": r"$R_c$ (AU)",
        "min": 15.0,      # ✅ FIXED: Observed R_disk ≈ 22 AU (Table 3)
                          # r_c typically 0.5-2× R_disk for exponential taper
        "max": 60.0,      # ✅ Upper bound allows moderate radial extent
        "default": 22.0,  # ✅ Start at observed disk radius
        "log_scale": False,  # Linear sampling for this narrow range
        "unit": "AU",
    },
    {
        "name": "vFrag",  # Fragmentation velocity (cm/s NOT m/s!)
        "label": r"$v_{\rm frag}$ (cm/s)",
        "min": 100.0,     # ✅ OPTIMIZED: Narrowed from 10-1000
        "max": 500.0,     # ✅ Focus on realistic range around 200 cm/s
        "default": 200.0,  # Test showed this works well
        "log_scale": False,  # Linear for narrow range
        "unit": "cm/s",
    },
    {
        "name": "sigma_exp",  # Surface density power-law exponent
        "label": r"$p$ (Σ exponent)",
        "min": 0.5,       # ✅ NEW PARAMETER: Allow variation in Σ profile shape
        "max": 1.5,       # ✅ Typical range for protoplanetary disks
        "default": 1.0,   # Standard value used in tests
        "log_scale": False,
        "unit": "",
    },
    # ❌ REMOVED: sigma_rc parameter (redundant with r_c, causes degeneracy)
    # {
    #     "name": "sigma_rc",  # Critical radius multiplier for surface density
    #     "label": r"$f_{R_c}$ (radius multiplier)",
    #     "min": 0.8,       # ✅ REPLACED alpha: Allow r_c variation ±20%
    #     "max": 1.2,       # ✅ Fast parameter, no DustPy convergence issues
    #     "default": 1.0,   # Standard r_c (no modification)
    #     "log_scale": False,  # Linear sampling around 1.0
    #     "unit": "",
    # },
    {
        "name": "dust_to_gas",  # Dust-to-gas mass ratio
        "label": r"$\epsilon$ (d2g ratio)",
        "min": 0.001,     # ✅ Depleted disks (settled/grown dust)
        "max": 0.05,      # ✅ WIDENED: Allow super-ISM enrichment (0.05 = 5× ISM)
        "default": 0.01,  # ISM value
        "log_scale": False,
        "unit": "",
    },
    # REMOVED: r_in parameter (causes extreme slowdown at low values like 0.1 AU)
    # Now FIXED at 1.0 AU in forward_simulator.py for stable performance
]

# Số parameters
N_PARAMS = len(MCMC_PARAMETERS)

# Extract bounds và default values (handle both dict and string formats)
if MCMC_PARAMETERS and isinstance(MCMC_PARAMETERS[0], dict):
    # Old format: list of dicts
    PARAM_NAMES = [p["name"] for p in MCMC_PARAMETERS]
    PARAM_LABELS = [p["label"] for p in MCMC_PARAMETERS]
    PARAM_BOUNDS = [(p["min"], p["max"]) for p in MCMC_PARAMETERS]
    PARAM_DEFAULTS = [p["default"] for p in MCMC_PARAMETERS]
    MCMC_PARAMETER_BOUNDS = {p["name"]: (p["min"], p["max"]) for p in MCMC_PARAMETERS}
    MCMC_PARAMETER_UNITS = {p["name"]: p["unit"] for p in MCMC_PARAMETERS}
    MCMC_PARAMETER_TRUTHS = {p["name"]: p["default"] for p in MCMC_PARAMETERS}
else:
    # New format: list of strings (parameter names)
    PARAM_NAMES = MCMC_PARAMETERS
    # These need to be defined externally when using string format
    MCMC_PARAMETER_BOUNDS = getattr(sys.modules[__name__], 'MCMC_PARAMETER_BOUNDS', {})
    MCMC_PARAMETER_UNITS = getattr(sys.modules[__name__], 'MCMC_PARAMETER_UNITS', {})
    MCMC_PARAMETER_TRUTHS = getattr(sys.modules[__name__], 'MCMC_PARAMETER_TRUTHS', {})
    PARAM_LABELS = [p for p in PARAM_NAMES]  # Default labels
    PARAM_BOUNDS = [MCMC_PARAMETER_BOUNDS.get(p, (0, 1)) for p in PARAM_NAMES]
    PARAM_DEFAULTS = [MCMC_PARAMETER_TRUTHS.get(p, 0.5) for p in PARAM_NAMES]


# =============================================================================
# MCMC SAMPLER SETTINGS
# =============================================================================

# ✅ OPTIMIZED based on test results (5.2 min/simulation):
# Updated for 7 parameters (was 4)
# Number of walkers (user requested: 5 walkers × 2 = 10 total)
N_WALKERS = 10  # Use 5 user-specified walkers × 2 = 10 total walkers

# Number of steps - QUICK TEST for resource optimization
N_STEPS_BURNIN = 5       # ✅ Quick burn-in for testing
N_STEPS_PRODUCTION = 15  # ✅ SHORT: 20 total steps for RAM optimization
N_STEPS_TOTAL = N_STEPS_BURNIN + N_STEPS_PRODUCTION

# Thinning (save every Nth sample to reduce autocorrelation)
THIN_BY = 1  # Keep all samples initially, can increase if autocorrelation high

# Parallel processing
import multiprocessing

# Tự động lấy số luồng CPU của máy
TOTAL_CORES = multiprocessing.cpu_count()

# An toàn: Chừa lại 2 luồng để chạy Windows/VSCode mượt mà
# Ví dụ: Máy 16 luồng -> dùng 14. Máy 4 luồng -> dùng 2.
SAFE_CORES = max(1, TOTAL_CORES - 4) 

# Tính toán số lượng Process tối ưu
# ⚠️  RAM OPTIMIZATION: Single worker for maximum stability
# Analysis: 10 walkers × 1 worker = sequential execution
# Trade-off: Longer runtime but minimal RAM usage (~500MB max)
# Perfect for 20-step test run
N_PROCESSES = 3  # Single worker for RAM optimization

USE_MULTIPROCESSING = True
# Checkpoint frequency
CHECKPOINT_INTERVAL = 5  # ✅ Save every 5 steps

# Convergence criteria
MIN_AUTOCORR_TIMES = 50  # Minimum number of autocorrelation times
MAX_AUTOCORR_CHANGE = 0.01  # Maximum change in autocorrelation time for convergence


# =============================================================================
# LIKELIHOOD SETTINGS
# =============================================================================

# Chi-squared calculation
USE_WEIGHTS = True  # Use inverse variance weighting
MASK_THRESHOLD_SIGMA = 3.0  # Only fit pixels above N*sigma in observation

# Prior types: "uniform", "gaussian", "log-uniform"
PRIOR_TYPE = "uniform"  # Default uniform priors within bounds

# ✅ OPTIMIZED: Custom priors based on test results (χ²_red = 1.77 excellent!)
# Format: {"param_name": {"type": "gaussian", "mean": float, "std": float}}
CUSTOM_PRIORS = {
    # ✅ Constrain r_c based on disk size mismatch analysis:
    # Test shows model disk 1.36× larger than IRAS → need smaller r_c
    # r_c=45 AU gave same χ² as 60 AU, suggesting r_c around 40-50 AU optimal
    "r_c": {"type": "gaussian", "mean": 45.0, "std": 10.0},  # Soft constraint
    
    # ✅ Dust-to-gas - ISM value as reference
    "dust_to_gas": {"type": "gaussian", "mean": 0.01, "std": 0.005},  # Allow depletion
    
    # ❌ REMOVED: sigma_rc (redundant parameter)
    # "sigma_rc": {"type": "gaussian", "mean": 1.0, "std": 0.1},
}


# =============================================================================
# LOGGING & DIAGNOSTICS
# =============================================================================

# Logging levels
LOG_LEVEL_CONSOLE = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL_FILE = "DEBUG"  # More detailed logging to file

# Progress bar
SHOW_PROGRESS = True  # Show tqdm progress bar
PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N steps

# Diagnostic plots frequency
PLOT_INTERVAL = 100  # Generate diagnostic plots every N steps
SAVE_CHAIN_INTERVAL = 50  # Save partial chain every N steps

# Verbosity
VERBOSE = True  # Print detailed information during run


# =============================================================================
# RADMC3D SETTINGS
# =============================================================================

# Opacity settings
USE_MIE_OPACITY = False  # Use Mie scattering (slower) or power-law (faster)
OPACITY_POWERLAW_INDEX = 1.0  # Beta for opacity ~ lambda^-beta

# Scattering
SCATTERING_MODE = 1  # 0=no scattering, 1=isotropic, 2=HG, 3=full

# Number of photons for Monte Carlo
N_PHOTONS_SCAT = 500000  # For scattering Monte Carlo
N_PHOTONS_THERM = 500000  # For thermal Monte Carlo

# Image rendering
RENDER_NPIX = NPIX
RENDER_SIZEAU = IMAGE_SIZE_AU


# =============================================================================
# ERROR HANDLING & RECOVERY
# =============================================================================

# Maximum retries for failed simulations
MAX_RETRIES = 3

# Fallback values for failed simulations
FAILED_LIKELIHOOD_VALUE = -1e10  # Very low likelihood for failed models

# Timeout settings
SIMULATION_TIMEOUT = 1200  # Maximum time for one complete simulation (seconds)

# Cleanup settings
CLEANUP_TEMP_FILES = True  # Remove temporary files after each iteration
KEEP_FAILED_RUNS = False  # Keep failed run directories for debugging


# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Result files
CHAIN_FILE = os.path.join(MCMC_OUTPUT_DIR, "chain.npy")
LNPROB_FILE = os.path.join(MCMC_OUTPUT_DIR, "lnprob.npy")
METADATA_FILE = os.path.join(MCMC_OUTPUT_DIR, "metadata.json")
SUMMARY_FILE = os.path.join(MCMC_OUTPUT_DIR, "summary.txt")

# Best-fit model
BEST_FIT_DIR = os.path.join(MCMC_OUTPUT_DIR, "best_fit_model")
BEST_FIT_PARAMS_FILE = os.path.join(BEST_FIT_DIR, "best_params.json")
BEST_FIT_IMAGE_FILE = os.path.join(BEST_FIT_DIR, "best_fit_image.fits")

# Plots
CORNER_PLOT_FILE = os.path.join(MCMC_OUTPUT_DIR, "corner_plot.png")
TRACE_PLOT_FILE = os.path.join(MCMC_OUTPUT_DIR, "trace_plot.png")
COMPARISON_PLOT_FILE = os.path.join(MCMC_OUTPUT_DIR, "obs_vs_model.png")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_initial_positions(n_walkers=N_WALKERS, scale=0.1):
    """
    Tạo initial positions cho MCMC walkers.
    
    Parameters:
    -----------
    n_walkers : int
        Number of walkers
    scale : float
        Fractional perturbation scale (0.1 = 10% of parameter range)
    
    Returns:
    --------
    positions : ndarray
        Initial positions with shape (n_walkers, n_params)
    """
    positions = []
    for _ in range(n_walkers):
        pos = []
        for param in MCMC_PARAMETERS:
            pmin, pmax = param["min"], param["max"]
            center = param["default"]
            
            # Perturbation range
            perturb = scale * (pmax - pmin)
            
            # Random position around default
            if param["log_scale"]:
                # Log-uniform distribution
                log_min = np.log10(max(pmin, center - perturb))
                log_max = np.log10(min(pmax, center + perturb))
                val = 10 ** np.random.uniform(log_min, log_max)
            else:
                # Uniform distribution
                val = np.random.uniform(
                    max(pmin, center - perturb),
                    min(pmax, center + perturb)
                )
            
            pos.append(val)
        positions.append(pos)
    
    return np.array(positions)


def params_to_dict(params_array):
    """Convert parameter array to dictionary."""
    return {name: val for name, val in zip(PARAM_NAMES, params_array)}


def dict_to_params(params_dict):
    """Convert parameter dictionary to array."""
    return np.array([params_dict[name] for name in PARAM_NAMES])


def print_config_summary():
    """Print configuration summary."""
    print("=" * 80)
    print("MCMC PIPELINE CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\n📁 Directories:")
    print(f"  Work directory: {WORK_DIR}")
    print(f"  Output directory: {MCMC_OUTPUT_DIR}")
    print(f"  Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"\n📊 Observation:")
    print(f"  FITS file: {OBS_FITS_PATH}")
    print(f"  Distance: {DISTANCE_PC} pc")
    print(f"  Inclination: {INCLINATION_DEG}°")
    print(f"  Wavelength: {WAV_MICRON} μm")
    print(f"\n🎯 MCMC Parameters ({N_PARAMS} parameters):")
    for param in MCMC_PARAMETERS:
        if isinstance(param, dict):
            # Old format: list of dicts
            print(f"  {param['name']:15s}: [{param['min']:8.3f}, {param['max']:8.3f}]  "
                  f"default={param['default']:8.3f}  log_scale={param['log_scale']}")
        else:
            # New format: list of strings (parameter names)
            bounds = MCMC_PARAMETER_BOUNDS.get(param, (0, 1))
            print(f"  {param:15s}: [{bounds[0]:8.3f}, {bounds[1]:8.3f}]")
    
    # Show fixed parameters if any
    fixed_params = getattr(sys.modules[__name__], 'FIXED_PARAMS', {})
    if fixed_params:
        print(f"\n🔒 Fixed Parameters:")
        for name, value in fixed_params.items():
            print(f"  {name:15s}: {value:8.3f} (fixed)")
    
    print(f"\n🔄 MCMC Settings:")
    print(f"  Walkers: {N_WALKERS}")
    print(f"  Burn-in steps: {N_STEPS_BURNIN}")
    print(f"  Production steps: {N_STEPS_PRODUCTION}")
    print(f"  Total steps: {N_STEPS_TOTAL}")
    print(f"  Parallel processes: {N_PROCESSES if USE_MULTIPROCESSING else 'Disabled'}")
    print(f"\n💾 Checkpoints:")
    print(f"  Checkpoint interval: every {CHECKPOINT_INTERVAL} steps")
    print(f"  Auto-save chain: every {SAVE_CHAIN_INTERVAL} steps")
    print("=" * 80)


if __name__ == "__main__":
    # Test configuration
    print_config_summary()
    
    # Test initial positions
    print("\n🎲 Testing initial positions generation...")
    pos = get_initial_positions(n_walkers=4)
    print(f"Generated {len(pos)} walker positions with {len(pos[0])} parameters each")
    print(f"Example walker #1: {params_to_dict(pos[0])}")
