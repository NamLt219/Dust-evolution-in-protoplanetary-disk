# 🚀 GETTING STARTED - MCMC Pipeline for IRAS 04166
**Date:** 2025-12-08  
**Status:** Production Ready (5 parameters, validated)

---

## 📋 6 FILE XƯƠNG SỐNG (CORE FILES)

### 1. **`mcmc_pipeline_config.py`** - Cấu hình trung tâm
```python
# Chứa TẤT CẢ config:
MCMC_PARAMETERS = [...]  # 5 free parameters
FIXED_PARAMETERS = {...}  # r_in, alpha
BEAM_SIZE_AU = (6.9, 5.1)
OBSERVED_FLUX = ...
```
**Vai trò:** Single source of truth cho toàn bộ pipeline

---

### 2. **`forward_simulator.py`** - Forward model
```python
class ForwardModelSimulatorV2:
    def simulate(self, params):
        # DustPy → RADMC3D → Beam convolution
        return model_flux
```
**Vai trò:** 
- Chạy DustPy simulation
- Generate RADMC3D input
- Ray tracing
- Beam convolution
- Return synthetic image

---

### 3. **`likelihood_calculator.py`** - Likelihood function
```python
def log_likelihood(params, observed_flux, model_simulator):
    model_flux = model_simulator.simulate(params)
    chi2 = np.sum((observed_flux - model_flux)**2 / sigma**2)
    return -0.5 * chi2
```
**Vai trò:** 
- Calculate χ² giữa model và data
- Return log-likelihood cho MCMC

---

### 4. **`mcmc_sampler.py`** - MCMC engine
```python
import emcee

def run_mcmc(n_walkers, n_steps, initial_params):
    sampler = emcee.EnsembleSampler(...)
    sampler.run_mcmc(...)
    return chain, log_prob
```
**Vai trò:**
- Wrapper cho emcee
- Parameter space exploration
- Convergence monitoring
- Chain management

---

### 5. **`mcmc_pipeline_main.py`** - Main orchestrator
```python
def main():
    # 1. Load config
    # 2. Initialize simulator
    # 3. Load observations
    # 4. Run MCMC
    # 5. Save results
    # 6. Generate plots
```
**Vai trò:**
- Entry point chính
- Orchestrate toàn bộ pipeline
- Error handling
- Progress monitoring

---

### 6. **`mcmc_logger.py`** - Logging system
```python
def setup_logger(log_dir):
    # Dual logging: file + console
    # Timestamp, levels, formatters
    return logger
```
**Vai trò:**
- Unified logging
- Debug information
- Progress tracking
- Error reporting

---

## 🔄 WORKFLOW DIAGRAM

```
┌─────────────────────────────────────────────┐
│   mcmc_pipeline_main.py (ORCHESTRATOR)      │
└────────┬────────────────────────────────────┘
         │
         ├─→ [1] Load mcmc_pipeline_config.py
         │        ├─ MCMC_PARAMETERS (5 params)
         │        ├─ BEAM_SIZE_AU
         │        └─ OBSERVED_FLUX
         │
         ├─→ [2] Initialize forward_simulator.py
         │        └─ ForwardModelSimulatorV2
         │
         ├─→ [3] Setup likelihood_calculator.py
         │        └─ log_likelihood function
         │
         ├─→ [4] Run mcmc_sampler.py
         │        ├─ emcee.EnsembleSampler
         │        ├─ n_walkers × n_steps
         │        └─ For each walker:
         │             │
         │             ├─→ Propose new params
         │             ├─→ forward_simulator.simulate()
         │             │    ├─ DustPy evolution
         │             │    ├─ RADMC3D ray tracing
         │             │    └─ Beam convolution
         │             ├─→ likelihood_calculator()
         │             └─→ Accept/Reject (Metropolis)
         │
         └─→ [5] Save & Analyze
              ├─ chain.npy
              ├─ log_prob.npy
              └─ Corner plots

(All logged by mcmc_logger.py)
```

---

## 🎯 6 THAM SỐ TỰ DO (FINAL CONFIG)

```python
MCMC_PARAMETERS = [
    {
        
        "name": "log_mdisk",  
        "label": r"$\log_{10}(M_{\rm gas}/M_{\odot})$",  # ✅ CORRECTED LABEL
        "min": -3.5,         
        "max": -1.5,         
        "default": -2.4,      
        "log_scale": False,   
        "unit": "M_sun",
    },
    {
        "name": "r_c",  # Characteristic radius in AU
        "label": r"$R_c$ (AU)",
        "min": 10.0,     
        "max": 30.0,     
        "default": 20.0,  
        "log_scale": False,  # Linear sampling for this narrow range
        "unit": "AU",
    },
    {
        "name": "vFrag", 
        "label": r"$v_{\rm frag}$ (cm/s)",
        "min": 100.0,    
        "max": 500.0,     
        "default": 200.0,  # Test showed this works well
        "log_scale": False,  # Linear for narrow range
        "unit": "cm/s",
    },
    {
        "name": "sigma_exp",  # Surface density power-law exponent (γ)
        "label": r"$\gamma$ (Σ exponent)",
        "min": 0.5,       
        "max": 2.5,
        "default": 1.6,  
        "log_scale": False,
        "unit": "",

    },

    {
        "name": "dust_to_gas",  # Dust-to-gas mass ratio
        "label": r"$\epsilon$ (d2g ratio)",
        "min": 0.001,   
        "max": 0.05,      
        "default": 0.01,  # ISM value
        "log_scale": False,
        "unit": "",
    },

    {
        "name": "r_in",          # Inner cavity (dust depletion) radius [AU]
        "label": r"$R_{\rm in}$ (AU)",
        "min": 1.0,   # Minimum: ~1 AU (sub-beam but affects thermal structure)
        "max": 10.0,  # Maximum: ~10 AU (ALMA beam at 156 pc ≈ 8 AU, stay resolvable)
        "default": 3.0,  # Starting guess: moderate inner clearing
        "log_scale": False,
        "unit": "AU",
    },
]
```


---

## 🚀 QUICK START



 Production MCMC
```bash
python3 mcmc_pipeline_main.py
```

---


## 🔧 KEY FEATURES

### Forward Model Pipeline:
1. **DustPy:** Disk evolution (1D, radial)
2. **RADMC3D:** Radiative transfer (3D)
3. **Beam:** Telescope convolution (ALMA-like)
4. **Output:** Synthetic image matching observations

### MCMC Configuration:
- **Sampler
- **Walkers
- **Steps
- **Total
- **Runtime

---



## 🎓 REFERENCES

### Key Papers:
- Birnstiel et al. (2012) - DustPy physics
- Dullemond et al. (2012) - RADMC3D
- Foreman-Mackey et al. (2013) - emcee

### Documentation:
- DustPy: https://stammler.github.io/dustpy/
- RADMC3D: https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/
- emcee: https://emcee.readthedocs.io/

---

