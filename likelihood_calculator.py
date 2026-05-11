import numpy as np
from typing import Dict, Optional, Tuple, Any
from scipy import stats
from scipy.ndimage import shift as ndimage_shift
from astropy.convolution import Gaussian2DKernel, convolve_fft

# RAM Guardian for OOM protection
try:
    from ram_guardian import check_ram_or_wait
    RAM_GUARDIAN_AVAILABLE = True
except ImportError:
    RAM_GUARDIAN_AVAILABLE = False

try:
    import mcmc_pipeline_config as config
    from mcmc_logger import get_logger
except ImportError:
    # Fallback configs for testing
    class config:
        IMAGE_NPIX = 201
        RMS_NOISE_JY = 2.3e-05


def apply_gaussian_peak_shift(model_data: np.ndarray,
                              dx_shift: float = None,
                              dy_shift: float = None) -> np.ndarray:
    """Apply the locked astrometric shift based on 2D Gaussian peak offsets.

    IMPORTANT: This is the physical alignment operation used in the likelihood math.
    shift order is [row_shift, col_shift] = [DY_SHIFT, DX_SHIFT].
    """
    dx = float(config.DX_SHIFT if dx_shift is None else dx_shift)
    dy = float(config.DY_SHIFT if dy_shift is None else dy_shift)
    return ndimage_shift(
        np.asarray(model_data, dtype=float),
        shift=[dy, dx],
        order=3,
        mode='constant',
        cval=0.0
    )


def apply_observation_beam_convolution(model_data: np.ndarray,
                                       beam_major_arcsec: float = None,
                                       beam_minor_arcsec: float = None,
                                       beam_pa_deg: float = None,
                                       pixel_scale_arcsec: float = None) -> np.ndarray:
    """Convolve model with ALMA beam in image plane before comparison.

    Uses astropy Gaussian2DKernel and FFT convolution.
    """
    bmaj = float(config.BEAM_MAJOR_ARCSEC if beam_major_arcsec is None else beam_major_arcsec)
    bmin = float(config.BEAM_MINOR_ARCSEC if beam_minor_arcsec is None else beam_minor_arcsec)
    bpa = float(config.BEAM_PA_DEG if beam_pa_deg is None else beam_pa_deg)
    pix = float(
        ((config.IMAGE_SIZE_AU / config.IMAGE_NPIX) / config.DISTANCE_PC)
        if pixel_scale_arcsec is None else pixel_scale_arcsec
    )

    if pix <= 0.0:
        raise ValueError(f"Invalid pixel_scale_arcsec={pix}")

    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_x_pix = (bmaj * fwhm_to_sigma) / pix
    sigma_y_pix = (bmin * fwhm_to_sigma) / pix
    theta_rad = np.deg2rad(bpa)

    kernel = Gaussian2DKernel(
        x_stddev=sigma_x_pix,
        y_stddev=sigma_y_pix,
        theta=theta_rad,
    )

    arr = np.asarray(model_data, dtype=float)
    return convolve_fft(
        arr,
        kernel,
        boundary='fill',
        fill_value=0.0,
        normalize_kernel=True,
        allow_huge=True,
        nan_treatment='interpolate',
        preserve_nan=True,
    )


def prepare_model_for_observation_comparison(model_data: np.ndarray,
                                             dx_shift: float = None,
                                             dy_shift: float = None,
                                             apply_beam_convolution: bool = True) -> np.ndarray:
    """Unified math path: shift (Gaussian-peak) then ALMA beam convolution."""
    shifted_model = apply_gaussian_peak_shift(
        model_data=np.asarray(model_data, dtype=float),
        dx_shift=dx_shift,
        dy_shift=dy_shift,
    )
    if not apply_beam_convolution:
        return shifted_model
    return apply_observation_beam_convolution(shifted_model)


def _gaussian_core_centroid(image: np.ndarray) -> Tuple[float, float]:
    """Estimate 2D Gaussian-like centroid from the bright core in pixel space."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")

    finite = np.isfinite(arr)
    if not np.any(finite):
        ny, nx = arr.shape
        return 0.5 * (nx - 1), 0.5 * (ny - 1)

    work = np.where(finite, arr, 0.0)
    work = np.clip(work, a_min=0.0, a_max=None)
    peak = float(np.max(work))
    if peak <= 0.0:
        ny, nx = arr.shape
        return 0.5 * (nx - 1), 0.5 * (ny - 1)

    core_mask = work >= (0.3 * peak)
    if not np.any(core_mask):
        core_mask = work > 0.0

    y_idx, x_idx = np.indices(work.shape)
    weights = np.where(core_mask, work, 0.0)
    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        ny, nx = arr.shape
        return 0.5 * (nx - 1), 0.5 * (ny - 1)

    cx = float(np.sum(weights * x_idx) / wsum)
    cy = float(np.sum(weights * y_idx) / wsum)
    return cx, cy


def get_gaussian_centroid(image: np.ndarray) -> Tuple[float, float]:
    """Public wrapper to retrieve Gaussian-core centroid in pixel space."""
    return _gaussian_core_centroid(image)


def verify_centroid_alignment(obs_array: np.ndarray,
                              shifted_model_array: np.ndarray) -> Tuple[float, float, float]:
    """Safety-net centroid check in pixel space.

    Returns
    -------
    dx_pix, dy_pix, distance_pix
        Pixel offsets defined as (model - observation).
    """
    obs_cx, obs_cy = _gaussian_core_centroid(obs_array)
    model_cx, model_cy = _gaussian_core_centroid(shifted_model_array)
    dx_pix = float(model_cx - obs_cx)
    dy_pix = float(model_cy - obs_cy)
    distance_pix = float(np.hypot(dx_pix, dy_pix))
    return dx_pix, dy_pix, distance_pix

class LikelihoodCalculator:

    def __init__(self, 
                 obs_image: np.ndarray, 
                 rms_noise: float = None,
                 roi_radius_pixels: int = None,
                 beam_major_arcsec: float = None,
                 beam_minor_arcsec: float = None,
                 pixel_scale_arcsec: float = None,
                 align_centers: bool = True):

        self.obs_image = obs_image
        # Use config default if rms_noise not provided
        self.rms_noise = rms_noise if rms_noise is not None else config.RMS_NOISE_JY
        

        if beam_major_arcsec and beam_minor_arcsec and pixel_scale_arcsec:
            # Beam solid angle: Ω_beam = π/(4 ln2) × BMAJ × BMIN  [arcsec²]
            beam_area_arcsec2 = (np.pi / (4.0 * np.log(2.0))) * beam_major_arcsec * beam_minor_arcsec
            pixel_area_arcsec2 = pixel_scale_arcsec ** 2
            self.pixels_per_beam = beam_area_arcsec2 / pixel_area_arcsec2

            # σ_eff = σ_thermal × √(N_ppb)  — the down-weighting factor
            self.effective_rms = rms_noise * np.sqrt(self.pixels_per_beam)

            print(f"INFO: Beam correlation correction applied (Czekala+2015 method):")
            print(f"  Thermal RMS      : {rms_noise:.3e} Jy/beam")
            print(f"  Beam area        : {beam_area_arcsec2:.4f} arcsec²")
            print(f"  Pixel area       : {pixel_area_arcsec2:.6f} arcsec²")
            print(f"  Pixels per beam  : {self.pixels_per_beam:.1f}")
            print(f"  σ_eff            : {self.effective_rms:.3e} Jy/beam  (×{np.sqrt(self.pixels_per_beam):.1f})")
            print(f"  → χ² will be normalised to ~N_beams independent d.o.f.")
        else:
            # Fallback if beam info not provided: use bare thermal RMS
            # (conservative — caller should always pass beam parameters)
            self.pixels_per_beam = 1.0
            self.effective_rms = rms_noise
            print(f"WARNING: Beam parameters not provided. Using bare thermal RMS.")
            print(f"  Thermal RMS: {rms_noise:.3e} Jy/beam  (NO beam-correlation correction)")

        self.inv_sigma2 = 1.0 / (self.effective_rms ** 2)
        

        # 2D GAUSSIAN PEAK ALIGNMENT (matches reference paper phase center)
        # The reference paper positioned their R=0 by running CASA imfit on the
        # brightest continuum source (2D Gaussian peak), NOT center-of-mass.
        # We load the same offset here so every model evaluation is registered
        # to that exact same phase center before the χ² is computed.
        try:
            self._dx_shift = float(config.DX_SHIFT)
            self._dy_shift = float(config.DY_SHIFT)
            self._shift_reference_method = str(getattr(config, "SHIFT_REFERENCE_METHOD", "2D_GAUSSIAN_PEAK"))
            print(f"INFO: 2D Gaussian peak alignment shift loaded from config:")
            print(f"  DX_SHIFT = {self._dx_shift:+.6f} px  (col, +ve = right)")
            print(f"  DY_SHIFT = {self._dy_shift:+.6f} px  (row, +ve = up)")
            print(f"  Method   : {self._shift_reference_method} (matches CASA imfit reference frame)")
        except (ImportError, AttributeError):
            self._dx_shift = 0.0
            self._dy_shift = 0.0
            self._shift_reference_method = "UNKNOWN"
            print("WARNING: DX_SHIFT/DY_SHIFT not found in config — shift set to zero.")

        self._alignment_check_every = int(getattr(config, "ALIGNMENT_CHECK_EVERY", 50))
        self._alignment_warning_emitted = False
        self._likelihood_calls = 0

        # Legacy attribute kept for API compatibility (no longer used for centering)
        self.align_centers = align_centers
        self.obs_peak = None
        

        ny, nx = obs_image.shape
        y, x = np.indices((ny, nx))
        center_y, center_x = ny // 2, nx // 2
        
        # Tính khoảng cách từ tâm
        r_sq = (x - center_x)**2 + (y - center_y)**2
        
        if roi_radius_pixels:
            # Chỉ tính trong vùng bán kính cho phép
            self.mask = r_sq <= (roi_radius_pixels**2)
            print(f"INFO: Likelihood using Circular ROI mask (R={roi_radius_pixels} pix)")
        else:
            # 3σ SIGNAL MASK — only fit pixels where the observed emission
            # is detected at ≥ 3σ significance.  This excludes noise-dominated
            # pixels from the χ², which would otherwise dominate the sum and
            # dilute the model sensitivity to the real disk structure.
            # Threshold uses the *bare* thermal RMS (not beam-corrected), because
            # the observation FITS pixel values are still in Jy/beam units.
            three_sigma_threshold = 3.0 * rms_noise
            self.mask = obs_image >= three_sigma_threshold
            n_signal = int(np.sum(self.mask))
            print(f"INFO: Likelihood using 3\u03c3 SIGNAL MASK:")
            print(f"  Threshold          : {three_sigma_threshold:.2e} Jy/beam  (3 \u00d7 {rms_noise:.2e})")
            print(f"  Signal pixels      : {n_signal}  /  {obs_image.size}  total")
            print(f"  Coverage           : {100.0*n_signal/obs_image.size:.1f} %")

        # Thống kê sơ bộ
        self.n_pixels = np.sum(self.mask)
        print(f"INFO: Likelihood initialized. Noise RMS={rms_noise:.2e}. Pixels used={self.n_pixels}")

    def log_likelihood(self, model_image: np.ndarray) -> float:

        # 1. Kiểm tra Model hợp lệ
        if model_image is None:
            return -np.inf
        
        if np.any(np.isnan(model_image)) or np.any(np.isinf(model_image)):
            # Silent return (logging in multiprocessing causes pickling issues)
            return -np.inf

        # 2. Kiểm tra kích thước
        if model_image.shape != self.obs_image.shape:
            # Silent return (logging in multiprocessing causes pickling issues)
            return -np.inf
        
        # 2.5. 2D GAUSSIAN PEAK ALIGNMENT — applied unconditionally before χ²
        # Translates the model to the same phase center used in the reference
        # paper (CASA imfit 2D Gaussian peak).  Uses order-3 spline interpolation
        # to preserve flux while avoiding ringing artefacts.
        # shift=[row_shift, col_shift] = [DY_SHIFT, DX_SHIFT]
        model_image = prepare_model_for_observation_comparison(
            model_data=model_image,
            dx_shift=self._dx_shift,
            dy_shift=self._dy_shift,
            apply_beam_convolution=True,
        )

        self._likelihood_calls += 1
        if self._alignment_check_every > 0 and (
            self._likelihood_calls == 1 or self._likelihood_calls % self._alignment_check_every == 0
        ):
            dx_pix, dy_pix, dist_pix = verify_centroid_alignment(self.obs_image, model_image)
            if dist_pix > 0.5 and not self._alignment_warning_emitted:
                warning_msg = (
                    "CRITICAL WARNING: Misalignment detected in pixel space! "
                    f"Δx={dx_pix:+.3f} px, Δy={dy_pix:+.3f} px, |Δ|={dist_pix:.3f} px "
                    "(threshold=0.5 px)"
                )
                try:
                    logger = get_logger()
                    logger.critical(warning_msg)
                except Exception:
                    print(warning_msg)
                self._alignment_warning_emitted = True

        # 3. Tính Residual (Dư lượng)
        # Residual = Model - Data (hoặc Data - Model, bình phương lên như nhau)
        residuals = model_image - self.obs_image
        
        # 4. Áp dụng Mask Hình Học
        # Chỉ lấy các pixel nằm trong ROI (nếu có set), hoặc toàn bộ ảnh
        valid_residuals = residuals[self.mask]
        
        # 5. Tính Chi-Square
        # Chi2 = Sum( (Residual / Sigma)^2 )
        # Tối ưu hóa: inv_sigma2 đã tính trước
        chi2 = np.sum(valid_residuals**2) * self.inv_sigma2
        
        # 6. Trả về Log Likelihood
        # ln(L) = -0.5 * Chi2
        return -0.5 * chi2

    def compute_reduced_chi2(self, model_image: np.ndarray, n_free_params: int) -> float:

        if model_image is None: 
            return np.inf
            
        log_L = self.log_likelihood(model_image)
        if log_L == -np.inf:
            return np.inf
            
        chi2 = -2.0 * log_L
        dof = self.n_pixels - n_free_params # Degrees of Freedom
        
        if dof <= 0:
            return chi2 # Tránh chia cho 0
            
        return chi2 / dof

class PriorEvaluator:
    """
    Đánh giá Prior (Tiền nghiệm) cho các tham số.
    """
    def __init__(self, config_params: list):
        self.params_config = config_params
        
    def log_prior(self, params_values: list) -> float:
        """
        Tính Log-Prior. 
        Uniform Prior: 0 nếu trong khoảng, -inf nếu ngoài khoảng.
        """
        if len(params_values) != len(self.params_config):
            return -np.inf
            
        for val, config in zip(params_values, self.params_config):
            p_min = config['min']
            p_max = config['max']
            
            if not (p_min <= val <= p_max):
                return -np.inf
                
        return 0.0

class MCMCProbability:
    """
    Wrapper class kết hợp Prior và Likelihood.
    Hàm này sẽ được gọi trực tiếp bởi emcee sampler.
    
    """
    def __init__(self, 
                 prior_evaluator: PriorEvaluator,
                 likelihood_calculator: LikelihoodCalculator,
                 forward_simulator: Any):
        
        self.prior = prior_evaluator
        self.likelihood = likelihood_calculator
        self.simulator = forward_simulator
     

    def __call__(self, params_values):
        """
        Hàm gọi chính (Callable).
        Input: Vector tham số từ walker.
        Output: Log Probability (ln P).
        """

        lp = self.prior.log_prior(params_values)
        if not np.isfinite(lp):
            return -np.inf

        param_dict = {}
        for val, config in zip(params_values, self.prior.params_config):
            param_dict[config['name']] = val
            

        success, model_image, metadata = self.simulator.simulate(param_dict)

        if not success or model_image is None:
            try:
                logger = get_logger()
                logger.debug("Walker penalized with -inf due to simulation failure/timeout.")
            except Exception:
                pass
            return -np.inf
            
        # 3. Tính Likelihood
        ll = self.likelihood.log_likelihood(model_image)
        
        if not np.isfinite(ll):
            return -np.inf
            
        # Log Probability = Log Prior + Log Likelihood
        return lp + ll
