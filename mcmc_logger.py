"""
Comprehensive Logging System for MCMC Pipeline
===============================================
Hierarchical logging với multiple handlers, context tracking, và error capture.

Features:
- Console + File logging với different levels
- Structured logging cho MCMC iterations
- Error tracking và exception handling
- Performance metrics
- Progress reporting

Author: Pipeline Builder
Date: 2025-11-19
"""

import logging
import sys
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, Dict, Any
import threading


class MCMCLogger:
    """
    Advanced logger cho MCMC pipeline với context-aware logging.
    """
    
    def __init__(self, 
                 name: str = "mcmc_pipeline",
                 log_dir: str = "./logs",
                 console_level: str = "INFO",
                 file_level: str = "DEBUG",
                 log_to_file: bool = True):
        """
        Initialize MCMC logger.
        
        Parameters:
        -----------
        name : str
            Logger name
        log_dir : str
            Directory to save log files
        console_level : str
            Console logging level
        file_level : str
            File logging level
        log_to_file : bool
            Enable file logging
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(getattr(logging, file_level.upper()))
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file = log_file
            self.logger.info(f"Logging to file: {log_file}")
        
        # Error log file (separate file for errors only)
        self.error_log_file = self.log_dir / f"{name}_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        error_handler = logging.FileHandler(self.error_log_file, mode='a')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        # Metrics tracking
        self.metrics = {
            "start_time": time.time(),
            "iterations": 0,
            "errors": 0,
            "warnings": 0,
            "simulation_times": [],
            "likelihood_values": [],
        }
        
        # Thread-safe lock for metrics
        self.lock = threading.Lock()
        
        self.logger.info("="*80)
        self.logger.info("MCMC Pipeline Logger Initialized")
        self.logger.info("="*80)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(message, kwargs))
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        with self.lock:
            self.metrics["warnings"] += 1
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details."""
        with self.lock:
            self.metrics["errors"] += 1
        
        error_msg = self._format_message(message, kwargs)
        
        if exception:
            error_msg += f"\n  Exception: {type(exception).__name__}: {str(exception)}"
            error_msg += f"\n  Traceback:\n{traceback.format_exc()}"
        
        self.logger.error(error_msg)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical error."""
        with self.lock:
            self.metrics["errors"] += 1
        
        error_msg = self._format_message(message, kwargs)
        
        if exception:
            error_msg += f"\n  Exception: {type(exception).__name__}: {str(exception)}"
            error_msg += f"\n  Traceback:\n{traceback.format_exc()}"
        
        self.logger.critical(error_msg)
    
    def _format_message(self, message: str, kwargs: Dict[str, Any]) -> str:
        """Format message with additional context."""
        if kwargs:
            context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} [{context}]"
        return message
    
    def log_iteration(self, 
                     iteration: int, 
                     walker_id: int,
                     params: Dict[str, float],
                     likelihood: float,
                     accepted: bool,
                     sim_time: float):
        """
        Log MCMC iteration with structured format.
        
        Format: Timestamp | Step | Walker | Params | LogProb | Accept | Time
        Example: 2025-12-17 14:32:15 | Step 0042 | W03 | log_mdisk=-1.70 r_c=22.0 | -1234.56 | ✓ | 5.2s
        """
        """
        Log một MCMC iteration với full context.
        
        Parameters:
        -----------
        iteration : int
            Iteration number
        walker_id : int
            Walker ID
        params : dict
            Parameter values
        likelihood : float
            Log-likelihood value
        accepted : bool
            Whether step was accepted
        sim_time : float
            Simulation time in seconds
        """
        with self.lock:
            self.metrics["iterations"] += 1
            self.metrics["simulation_times"].append(sim_time)
            self.metrics["likelihood_values"].append(likelihood)
        
        # Structured format for easy parsing
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        status = "✓" if accepted else "✗"
        
        # Compact parameter display (show first 3, abbreviate rest)
        param_str = ", ".join([f"{k}={v:.3f}" for k, v in list(params.items())[:3]])
        if len(params) > 3:
            param_str += f" +{len(params)-3}more"
        
        # Structured log line: Timestamp | Step | Walker | Params | LogProb | Status | Time
        log_line = (
            f"{timestamp} | Step {iteration:04d} | W{walker_id:02d} | "
            f"{param_str} | LogProb={likelihood:+.2f} | {status} | {sim_time:.1f}s"
        )
        
        self.logger.info(log_line)
        
        msg = (f"Iter {iteration:5d} | Walker {walker_id:3d} | {status} | "
               f"ln(L)={likelihood:12.4f} | t={sim_time:6.2f}s | {param_str}")
        
        self.debug(msg)
    
    def log_checkpoint(self, iteration: int, filename: str):
        """Log checkpoint save."""
        self.info(f"💾 Checkpoint saved at iteration {iteration}: {filename}")
    
    def log_chain_stats(self, 
                       iteration: int,
                       acceptance_rate: float,
                       mean_likelihood: float,
                       std_likelihood: float,
                       autocorr_time: Optional[float] = None):
        """
        Log chain statistics.
        
        Parameters:
        -----------
        iteration : int
            Current iteration
        acceptance_rate : float
            Overall acceptance rate
        mean_likelihood : float
            Mean log-likelihood
        std_likelihood : float
            Std of log-likelihood
        autocorr_time : float, optional
            Autocorrelation time
        """
        msg = (f"📊 Chain Stats [Iter {iteration}]: "
               f"AcceptRate={acceptance_rate:.3f}, "
               f"<ln(L)>={mean_likelihood:.4f}±{std_likelihood:.4f}")
        
        if autocorr_time is not None:
            msg += f", τ_autocorr={autocorr_time:.1f}"
        
        self.info(msg)
    
    def log_convergence_check(self, 
                             iteration: int,
                             converged: bool,
                             autocorr_times: Optional[Dict[str, float]] = None,
                             gelman_rubin: Optional[Dict[str, float]] = None):
        """Log convergence diagnostics."""
        if converged:
            self.info(f"✅ CONVERGENCE achieved at iteration {iteration}!")
        else:
            msg = f"🔄 Convergence check [Iter {iteration}]: Not yet converged"
            if autocorr_times:
                msg += f" | τ_autocorr: {list(autocorr_times.values())[:3]}"
            self.debug(msg)
    
    def log_simulation_start(self, params: Dict[str, float], sim_id: str):
        """Log start of forward simulation."""
        param_str = ", ".join([f"{k}={v:.4f}" for k, v in params.items()])
        self.debug(f"🚀 Starting simulation [{sim_id}]: {param_str}")
    
    def log_simulation_end(self, 
                          sim_id: str,
                          success: bool,
                          sim_time: float,
                          error: Optional[str] = None):
        """Log end of forward simulation."""
        if success:
            self.debug(f"✅ Simulation [{sim_id}] completed in {sim_time:.2f}s")
        else:
            self.error(f"❌ Simulation [{sim_id}] FAILED after {sim_time:.2f}s: {error}")
    
    def log_phase_transition(self, phase: str, iteration: int):
        """Log transition between MCMC phases (e.g., burn-in -> production)."""
        self.info("="*80)
        self.info(f"🔄 PHASE TRANSITION: Entering {phase.upper()} phase at iteration {iteration}")
        self.info("="*80)
    
    def log_best_fit_update(self, 
                           iteration: int,
                           params: Dict[str, float],
                           likelihood: float):
        """Log new best-fit model."""
        param_str = ", ".join([f"{k}={v:.4f}" for k, v in params.items()])
        self.info(f"⭐ NEW BEST FIT [Iter {iteration}]: ln(L)={likelihood:.4f} | {param_str}")
    
    def log_performance_summary(self):
        """Log performance metrics summary."""
        elapsed = time.time() - self.metrics["start_time"]
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        avg_sim_time = (sum(self.metrics["simulation_times"]) / len(self.metrics["simulation_times"]) 
                       if self.metrics["simulation_times"] else 0)
        
        self.info("="*80)
        self.info("PERFORMANCE SUMMARY")
        self.info("="*80)
        self.info(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        self.info(f"Total iterations: {self.metrics['iterations']}")
        self.info(f"Average simulation time: {avg_sim_time:.2f}s")
        self.info(f"Errors encountered: {self.metrics['errors']}")
        self.info(f"Warnings: {self.metrics['warnings']}")
        
        if self.metrics["simulation_times"]:
            total_sim_time = sum(self.metrics["simulation_times"])
            overhead_time = elapsed - total_sim_time
            self.info(f"Simulation time: {total_sim_time:.1f}s ({100*total_sim_time/elapsed:.1f}%)")
            self.info(f"Overhead time: {overhead_time:.1f}s ({100*overhead_time/elapsed:.1f}%)")
        
        self.info("="*80)
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        metrics_export = self.metrics.copy()
        metrics_export["total_runtime"] = time.time() - self.metrics["start_time"]
        
        # Remove large arrays to keep file small
        if len(metrics_export["simulation_times"]) > 1000:
            metrics_export["simulation_times"] = metrics_export["simulation_times"][-1000:]
        if len(metrics_export["likelihood_values"]) > 1000:
            metrics_export["likelihood_values"] = metrics_export["likelihood_values"][-1000:]
        
        with open(filepath, 'w') as f:
            json.dump(metrics_export, f, indent=2)
        
        self.info(f"Metrics saved to {filepath}")
    
    def close(self):
        """Close logger and save final metrics."""
        self.log_performance_summary()
        
        # Save metrics
        metrics_file = self.log_dir / f"{self.name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.save_metrics(str(metrics_file))
        
        self.info(f"Log file: {getattr(self, 'log_file', 'N/A')}")
        self.info(f"Error log: {self.error_log_file}")
        self.info("Logger closed.")
        
        # Close handlers
        for handler in self.logger.handlers:
            handler.close()


# Global logger instance
_global_logger: Optional[MCMCLogger] = None


def get_logger() -> MCMCLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        raise RuntimeError("Logger not initialized. Call setup_logger() first.")
    return _global_logger


def setup_logger(log_dir: str = "./logs", 
                console_level: str = "INFO",
                file_level: str = "DEBUG") -> MCMCLogger:
    """Setup global logger."""
    global _global_logger
    _global_logger = MCMCLogger(
        name="mcmc_pipeline",
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level
    )
    return _global_logger


if __name__ == "__main__":
    # Test logger
    logger = setup_logger(log_dir="./test_logs")
    
    logger.info("Testing logger functionality")
    logger.debug("Debug message with details", param1=1.23, param2="value")
    logger.warning("Warning message")
    
    # Test iteration logging
    logger.log_iteration(
        iteration=100,
        walker_id=5,
        params={"alpha": 1.5, "beta": 0.3, "gamma": 2.1},
        likelihood=-1234.56,
        accepted=True,
        sim_time=15.3
    )
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error("Caught an error during testing", exception=e)
    
    logger.close()
