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
    from mcmc_pipeline_config import *
    from mcmc_logger import get_logger
except ImportError:
    # Fallback configs for testing
    IMAGE_NPIX = 201
    RMS_NOISE_JY = 2.3e-05

class LikelihoodCalculator:

    def __init__(self, 
                 obs_image: np.ndarray, 
                 rms_noise: float = RMS_NOISE_JY,
                 roi_radius_pixels: int = None,
                 beam_major_arcsec: float = None,
                 beam_minor_arcsec: float = None,
                 pixel_scale_arcsec: float = None,
                 align_centers: bool = True):

        
        self.obs_image = obs_image
        self.rms_noise = rms_noise
        

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
        

        self.align_centers = align_centers
        if align_centers:
  
            nonzero_vals = obs_image[obs_image > 0]
            if len(nonzero_vals) == 0:
                self.obs_peak = None
                print("WARNING: Observation image is all zeros — centering disabled.")
            else:
                threshold = np.percentile(nonzero_vals, 90)  # top 10% of actual signal
                bright_mask = obs_image >= threshold
                total_flux = obs_image[bright_mask].sum()
                ny_o, nx_o = obs_image.shape
                yy, xx = np.indices((ny_o, nx_o))
                cen_y = (obs_image * bright_mask * yy).sum() / total_flux
                cen_x = (obs_image * bright_mask * xx).sum() / total_flux
                self.obs_peak = (cen_y, cen_x)
                argmax_yx = np.unravel_index(np.argmax(obs_image), obs_image.shape)
                print(f"INFO: Centering correction enabled (flux-weighted centroid, non-zero pixels only).")
                print(f"  centroid  : ({cen_x:.2f}, {cen_y:.2f}) [x, y]")
                print(f"  argmax    : ({argmax_yx[1]}, {argmax_yx[0]}) [x, y]")
                print(f"  image ctr : ({nx_o//2}, {ny_o//2}) [x, y]")
        else:
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
            # Mặc định: Lấy toàn bộ ảnh (An toàn nhất để tránh bias)
            self.mask = np.ones_like(obs_image, dtype=bool)
            print("INFO: Likelihood using FULL IMAGE (No masking) - Safest option")
                
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
        
        # 2.5. CENTERING CORRECTION
        # Align model centroid to observation centroid using sub-pixel shift.
        # Uses flux-weighted centroid (top 10% pixels) — stable against noise spikes.
        if self.align_centers and self.obs_peak is not None:
            nonzero_mod = model_image[model_image > 0]
            if len(nonzero_mod) == 0:
                return -np.inf  # degenerate model (all zeros)
            threshold = np.percentile(nonzero_mod, 90)  # top 10% of actual signal
            bright_mask = model_image >= threshold
            total_flux = model_image[bright_mask].sum()
            if total_flux > 0:
                ny_m, nx_m = model_image.shape
                yy, xx = np.indices((ny_m, nx_m))
                mod_cen_y = (model_image * bright_mask * yy).sum() / total_flux
                mod_cen_x = (model_image * bright_mask * xx).sum() / total_flux
                dy = self.obs_peak[0] - mod_cen_y
                dx = self.obs_peak[1] - mod_cen_x
                # Only shift if offset > 0.3 pixel (sub-pixel precision)
                if abs(dy) > 0.3 or abs(dx) > 0.3:
                    model_image = ndimage_shift(model_image, shift=(dy, dx),
                                               order=1, mode='constant', cval=0.0)

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
