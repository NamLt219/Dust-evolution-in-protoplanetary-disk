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
import multiprocessing

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
        """
        self.name = name
        self.log_dir = Path(log_dir)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        
        # Create main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Clear existing handlers
        self.logger.propagate = False # Prevent double logging in some envs
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler — UNBUFFERED so every log line is flushed to disk
        # immediately. This prevents data loss if the process crashes.
        self.log_file = None
        if log_to_file:
            # Use process ID in filename to prevent conflicts in multiprocessing
            pid = os.getpid()
            timestamp = datetime.now().strftime('%Y%m%d') # Daily log file
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            
            try:
                # Open underlying stream with write-through (unbuffered text)
                # so every emit() is immediately persisted to disk.
                unbuffered_stream = open(log_file, mode='a', encoding='utf-8', buffering=1)  # line-buffered
                file_handler = logging.StreamHandler(unbuffered_stream)
                file_handler.setLevel(getattr(logging, file_level.upper()))
                file_formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)-8s | P%(process)d | %(funcName)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
                self.log_file = log_file
                # ✅ Print absolute log path so user can always find it
                abs_path = str(log_file.resolve())
                print(f"📝 LOG FILE: {abs_path}", flush=True)
            except Exception as e:
                print(f"WARNING: Could not setup file logging: {e}", flush=True)
        
        # Metrics tracking
        self.metrics = {
            "start_time": time.time(),
            "iterations": 0,
            "errors": 0,
            "warnings": 0,
            "simulation_times": [],
            "likelihood_values": [],
        }
        
        # Thread-safe lock for metrics (Not process-safe, but helps in threads)
        self.lock = threading.Lock()
        
        # self.logger.info("="*80)
        # self.logger.info(f"Logger initialized (PID: {os.getpid()})")
    
    def info(self, message: str, **kwargs):
        self.logger.info(self._format_message(message, kwargs))
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        with self.lock:
            self.metrics["warnings"] += 1
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        with self.lock:
            self.metrics["errors"] += 1
        error_msg = self._format_message(message, kwargs)
        if exception:
            error_msg += f"\n  Exception: {type(exception).__name__}: {str(exception)}"
            # Truncate traceback to avoid huge logs
            tb = traceback.format_exc()
            error_msg += f"\n  Traceback: {tb[-500:]}..." 
        self.logger.error(error_msg)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        with self.lock:
            self.metrics["errors"] += 1
        error_msg = self._format_message(message, kwargs)
        if exception:
            error_msg += f"\n  Exception: {type(exception).__name__}: {str(exception)}"
            error_msg += f"\n  Traceback:\n{traceback.format_exc()}"
        self.logger.critical(error_msg)
    
    def _format_message(self, message: str, kwargs: Dict[str, Any]) -> str:
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
        """
        # Only keep last 1000 metrics to save RAM
        with self.lock:
            self.metrics["iterations"] += 1
            if len(self.metrics["simulation_times"]) < 1000:
                self.metrics["simulation_times"].append(sim_time)
                self.metrics["likelihood_values"].append(likelihood)
        
        status = "✓" if accepted else "✗"
        # Compact parameter display
        param_str = ", ".join([f"{k}={v:.3f}" for k, v in list(params.items())[:3]])
        
        msg = (f"Iter {iteration:5d} | Walker {walker_id:3d} | {status} | "
               f"ln(L)={likelihood:12.4f} | t={sim_time:6.2f}s | {param_str}")
        
        self.debug(msg)
    
    def log_best_fit_update(self, iteration: int, params: Dict[str, float], likelihood: float):
        param_str = ", ".join([f"{k}={v:.4f}" for k, v in params.items()])
        self.info(f"⭐ NEW BEST FIT [Iter {iteration}]: ln(L)={likelihood:.4f} | {param_str}")
    
    def log_chain_stats(self, iteration: int, acceptance_rate: float, mean_likelihood: float, std_likelihood: float, autocorr_time: Optional[float] = None):
        msg = (f"📊 Chain Stats [Iter {iteration}]: AcceptRate={acceptance_rate:.3f}, <ln(L)>={mean_likelihood:.4f}")
        if autocorr_time:
            msg += f", τ={autocorr_time:.1f}"
        self.info(msg)

    def close(self):
        for handler in self.logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass
            handler.close()

# Global logger instance
_global_logger: Optional[MCMCLogger] = None

def get_logger() -> MCMCLogger:
    """
    Get global logger instance.
    ROBUST VERSION: Auto-initializes if called from a worker process
    where setup_logger() hasn't been called.
    """
    global _global_logger
    if _global_logger is None:
        # Emergency setup for worker processes to prevent crash
        # This logs to console only to be safe
        _global_logger = MCMCLogger(
            name=f"worker_{os.getpid()}",
            log_dir="./logs",
            console_level="INFO",
            log_to_file=False 
        )
    return _global_logger

def setup_logger(log_dir: str = "./logs", 
                console_level: str = "INFO",
                file_level: str = "DEBUG") -> MCMCLogger:
    """Setup global logger (Main Process)."""
    global _global_logger
    _global_logger = MCMCLogger(
        name="mcmc_pipeline",
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level
    )
    return _global_logger

if __name__ == "__main__":
    logger = setup_logger(log_dir="./test_logs")
    logger.info("Test message")
    logger.close()
