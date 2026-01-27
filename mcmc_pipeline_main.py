"""
MCMC Pipeline Main Runner (The Orchestrator - Bulletproof Version)
=================================================================
Tính năng an toàn cao cấp:
- System Health Check (RAM/Disk)
- Crash Dump Reporting
- Graceful Shutdown
- Safe Multiprocessing

Author: Gemini Senior Dev & Professor
Date: 2025-12-21
"""

import os
import sys
import shutil
import psutil  # Cần cài: pip install psutil (Không có cũng không sao, code tự xử lý)
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import set_start_method
from astropy.io import fits
import argparse
import traceback
import platform
from datetime import datetime

# Import các module vệ tinh
from mcmc_pipeline_config import *
from mcmc_logger import setup_logger
from forward_simulator import ForwardModelSimulatorV2
from likelihood_calculator import (
    LikelihoodCalculator, 
    PriorEvaluator, 
    MCMCProbability
)
from mcmc_sampler import MCMCSampler

class MCMCPipeline:
    def __init__(self, resume: bool = True):
        self.resume = resume
        
        # Setup Logger
        self.logger = setup_logger(
            log_dir=LOG_DIR,
            console_level="INFO",
            file_level="DEBUG"
        )
        
        self.logger.info("="*80)
        self.logger.info("🛡️  MCMC PIPELINE ORCHESTRATOR (SAFE MODE) STARTED")
        self.logger.info("="*80)
        
        # System Health Check ngay khi khởi động
        self._check_system_health()

        self.obs_data = None
        self.simulator = None
        self.likelihood = None
        self.prior = None
        self.prob_fn = None
        self.sampler_engine = None

    def _check_system_health(self):
        """Kiểm tra sức khỏe hệ thống trước khi cất cánh."""
        self.logger.info("🏥 Performing System Health Check...")
        
        # 1. Check Disk Space
        total, used, free = shutil.disk_usage(BASE_DIR)
        free_gb = free / (1024**3)
        self.logger.info(f"   Disk Free Space: {free_gb:.2f} GB")
        
        if free_gb < 2.0: # Cảnh báo nếu còn dưới 2GB
            self.logger.warning("⚠️ LOW DISK SPACE! Less than 2GB remaining.")
            
        # 2. Check RAM (nếu có psutil)
        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            self.logger.info(f"   RAM Available: {available_gb:.2f} GB")
            if available_gb < 1.0:
                self.logger.warning("⚠️ LOW RAM! Less than 1GB available. Risk of OOM Crash.")
        except ImportError:
            self.logger.info("   (psutil not installed, skipping RAM check)")

    def load_data(self):
        """Load dữ liệu quan sát."""
        self.logger.info(f"📥 Loading observation: {OBS_FITS_PATH}")
        if not os.path.exists(OBS_FITS_PATH):
            self.logger.critical(f"❌ File not found: {OBS_FITS_PATH}")
            sys.exit(1)
            
        with fits.open(OBS_FITS_PATH) as hdul:
            self.obs_data = hdul[0].data.squeeze()
            while self.obs_data.ndim > 2:
                self.obs_data = self.obs_data[0]
                
            # Log statistic của ảnh để chắc chắn load đúng
            self.logger.info(f"   Image Stats: Min={np.nanmin(self.obs_data):.2e}, Max={np.nanmax(self.obs_data):.2e} Jy/beam")

    def setup_components(self):
        """Khởi tạo các module con."""
        self.logger.info("⚙️  Setting up pipeline components...")
        
        try:
            self.simulator = ForwardModelSimulatorV2(cleanup=True)
            
            # ✅ CRITICAL FIX: Pass beam parameters for correlation correction
            # Calculate pixel scale from observation FITS
            pixel_scale_arcsec = IMAGE_SIZE_AU / IMAGE_NPIX / DISTANCE_PC  # arcsec/pixel
            
            self.likelihood = LikelihoodCalculator(
                obs_image=self.obs_data,
                rms_noise=RMS_NOISE_JY,
                roi_radius_pixels=None,
                beam_major_arcsec=BEAM_MAJOR_ARCSEC,  # From config
                beam_minor_arcsec=BEAM_MINOR_ARCSEC,  # From config
                pixel_scale_arcsec=pixel_scale_arcsec
            )
            
            self.prior = PriorEvaluator(MCMC_PARAMETERS)
            self.prob_fn = MCMCProbability(self.prior, self.likelihood, self.simulator)
            
            # 2.4 Sampler Engine (The Worker)
            self.sampler_engine = MCMCSampler(
                log_prob_fn=self.prob_fn, # <--- TRUYỀN HÀM XÁC SUẤT VÀO NGAY TẠI ĐÂY
                n_params=len(MCMC_PARAMETERS),
                n_walkers=N_WALKERS,
                param_names=[p['name'] for p in MCMC_PARAMETERS],
                checkpoint_dir=MCMC_OUTPUT_DIR,
                backend_filename="mcmc_chain.h5",
                use_parallel=USE_MULTIPROCESSING,
                n_processes=N_PROCESSES
            )
            self.logger.info("✅ Components initialized.")
            
        except Exception as e:
            self.logger.critical("❌ Component Setup Failed!", exception=e)
            raise e

    def _generate_initial_positions(self):
        """Tạo vị trí xuất phát với scatter hợp lý."""
        pos = []
        defaults = np.array([p['default'] for p in MCMC_PARAMETERS])
        
        for _ in range(N_WALKERS):
            # Scatter adaptively: 2% of parameter value (or 0.01 for values near 0)
            scatter = np.maximum(0.02 * np.abs(defaults), 0.01)
            p_walk = defaults + scatter * np.random.randn(len(defaults))
            
            # Clip to prior bounds
            for i, param in enumerate(MCMC_PARAMETERS):
                p_walk[i] = np.clip(p_walk[i], param['min'], param['max'])
            pos.append(p_walk)
        
        return np.array(pos)

    def run(self):
        """Chạy Pipeline."""
        self.logger.info("🏁 STARTING PRODUCTION RUN")
        
        backend_exists = os.path.exists(os.path.join(MCMC_OUTPUT_DIR, "mcmc_chain.h5"))
        should_reset = (not self.resume) or (not backend_exists)
        
        initial_state = self._generate_initial_positions()
        
        try:
            self.sampler_engine.run(
                initial_positions=initial_state,
                n_steps=N_STEPS_TOTAL,
                resume=not should_reset,
                show_progress=True
            )
            self.logger.info("🎉 MCMC Run Finished Successfully!")
            
        except KeyboardInterrupt:
            self.logger.warning("🛑 Interrupted by user. Data is safe in HDF5.")
            sys.exit(0)
            
        except Exception as e:
            # Đây là nơi bắt các lỗi Runtime bất ngờ (như RADMC crash)
            self.logger.critical("❌ RUNTIME ERROR in MCMC Loop", exception=e)
            raise e

    def analyze_and_plot(self):
        """Phân tích kết quả."""
        self.logger.info("📊 Analyzing results...")
        try:
            samples, log_probs = self.sampler_engine.get_samples(
                discard=N_STEPS_BURNIN, thin=1, flat=True
            )
            
            # 1. Corner Plot
            try:
                import corner
                labels = [p['name'] for p in MCMC_PARAMETERS]
                truths = [p['default'] for p in MCMC_PARAMETERS]
                fig = corner.corner(samples, labels=labels, truths=truths, show_titles=True)
                fig.savefig(os.path.join(MCMC_OUTPUT_DIR, "corner_plot.png"), bbox_inches='tight')
                plt.close(fig)
                self.logger.info("✅ Corner plot saved.")
            except ImportError:
                self.logger.warning("Skipping corner plot (library missing).")

            # 2. Trace Plot
            chain = self.sampler_engine.get_chain(discard=0, flat=False)
            n_dim = chain.shape[2]
            fig, axes = plt.subplots(n_dim, figsize=(10, 2*n_dim), sharex=True)
            if n_dim == 1: axes = [axes]
            labels = [p['name'] for p in MCMC_PARAMETERS]
            for i in range(n_dim):
                axes[i].plot(chain[:, :, i], "k", alpha=0.3)
                axes[i].set_ylabel(labels[i])
                axes[i].axvline(N_STEPS_BURNIN, color="r", ls="--")
            fig.savefig(os.path.join(MCMC_OUTPUT_DIR, "trace_plot.png"), bbox_inches='tight')
            plt.close(fig)
            self.logger.info("✅ Trace plot saved.")

            # 3. Best Fit Image
            self._generate_best_fit_image()

        except Exception as e:
            self.logger.error("Analysis failed", exception=e)

    def _generate_best_fit_image(self):
        if self.sampler_engine.best_params is not None:
            self.logger.info("🖼️ Generating Best-Fit Image...")
            best_p = self.sampler_engine.best_params
            param_dict = {cfg['name']: val for val, cfg in zip(best_p, MCMC_PARAMETERS)}
            
            success, image, _ = self.simulator.simulate(param_dict)
            if success:
                hdu = fits.PrimaryHDU(image)
                hdu.writeto(os.path.join(MCMC_OUTPUT_DIR, "best_fit_model.fits"), overwrite=True)
                self.logger.info("✅ Best fit FITS saved.")

def dump_crash_report(e):
    """Ghi báo cáo lỗi ra file riêng khi pipeline sập hoàn toàn."""
    report_file = os.path.join(BASE_DIR, f"CRASH_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
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
        set_start_method('forkserver')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Start fresh")
    args = parser.parse_args()

    try:
        pipeline = MCMCPipeline(resume=not args.clean)
        pipeline.load_data()
        pipeline.setup_components()
        pipeline.run()
        pipeline.analyze_and_plot()
        
    except Exception as e:
        # Bắt lỗi ở tầng cao nhất (Top-level Exception Handler)
        # Đây là lớp bảo vệ cuối cùng
        dump_crash_report(e)
        sys.exit(1)
