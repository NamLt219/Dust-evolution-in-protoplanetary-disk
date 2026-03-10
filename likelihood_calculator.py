import numpy as np
from typing import Dict, Optional, Tuple, Any
from scipy import stats
from scipy.ndimage import shift as ndimage_shift

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
            print(f"INFO: 2D Gaussian peak alignment shift loaded from config:")
            print(f"  DX_SHIFT = {self._dx_shift:+.6f} px  (col, +ve = right)")
            print(f"  DY_SHIFT = {self._dy_shift:+.6f} px  (row, +ve = up)")
            print(f"  Method   : 2D Gaussian peak (matches CASA imfit reference frame)")
        except (ImportError, AttributeError):
            self._dx_shift = 0.0
            self._dy_shift = 0.0
            print("WARNING: DX_SHIFT/DY_SHIFT not found in config — shift set to zero.")

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
        model_image = ndimage_shift(
            model_image,
            shift=[self._dy_shift, self._dx_shift],
            order=3,
            mode='constant',
            cval=0.0
        )

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
            return -np.inf
            
        # 3. Tính Likelihood
        ll = self.likelihood.log_likelihood(model_image)
        
        if not np.isfinite(ll):
            return -np.inf
            
        # Log Probability = Log Prior + Log Likelihood
        return lp + ll
