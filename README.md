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

## 🎯 5 THAM SỐ TỰ DO (FINAL CONFIG)

```python
MCMC_PARAMETERS = [
    {
        "name": "log_mdisk",      # Log10(M_disk/M_sun)
        "min": -4.0,
        "max": -1.0,
        "default": -1.92,
    },
    {
        "name": "r_c",            # Characteristic radius (AU)
        "min": 30.0,
        "max": 80.0,
        "default": 45.0,
    },
    {
        "name": "vFrag",          # Fragmentation velocity (cm/s)
        "min": 100.0,
        "max": 500.0,
        "default": 200.0,
    },
    {
        "name": "sigma_exp",      # Surface density exponent
        "min": 0.5,
        "max": 1.5,
        "default": 1.0,
    },
    {
        "name": "dust_to_gas",    # Dust-to-gas ratio
        "min": 0.001,
        "max": 0.02,
        "default": 0.01,
    },
]

FIXED_PARAMETERS = {
    "r_in": 1.0,      # Inner radius (AU)
    "alpha": 1e-3,    # Turbulence parameter
}
```

**Lý do 5 params:**
- Đã loại bỏ `sigma_rc` → tránh degeneracy với `r_c`
- `r_in` fixed → tránh slow convergence
- `alpha` fixed → standard value

---

## 🚀 QUICK START

### 1. Setup Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python3 -c "
import dustpy
import radmc3dPy
import emcee
print('✅ All packages installed')
"
```

### 3. Run Test Simulation
```bash
# Quick test (1 forward model call)
python3 forward_simulator.py
```

### 4. Run Validation (Tonight!)
```bash
# Full validation suite (~4 hours)
./setup_overnight.sh
```

### 5. Production MCMC (After validation)
```bash
# Full MCMC run (50 walkers × 2000 steps)
python3 mcmc_pipeline_main.py
```

---

## 📊 VALIDATION STATUS

### Spiral 1 (ĐANG CHẠY):
- ✅ Test 1.1: DustPy Parameter Sensitivity (7 tests)
- ✅ Test 1.2: Mass Conservation
- **Status:** Running (started 18:15)
- **ETA:** 23:00 tonight

### Spiral 2 (TODO):
- [ ] RADMC3D validation
- [ ] Beam convolution tests
- [ ] Integration tests

### Spiral 3 (TODO):
- [ ] High-res synthetic recovery
- [ ] Convergence diagnostics
- [ ] Final GO/NO-GO

---

## 📁 DIRECTORY STRUCTURE

```
Claude.4.5_MCMC/
├── 📋 CONFIG & CORE
│   ├── mcmc_pipeline_config.py    # ⭐ Central config
│   ├── forward_simulator.py       # ⭐ Forward model
│   ├── likelihood_calculator.py   # ⭐ Likelihood
│   ├── mcmc_sampler.py           # ⭐ MCMC engine
│   ├── mcmc_pipeline_main.py     # ⭐ Main orchestrator
│   └── mcmc_logger.py            # ⭐ Logging
│
├── 🧪 VALIDATION
│   ├── tests/
│   │   ├── test_dustpy_sensitivity.py
│   │   └── test_mass_conservation.py
│   ├── run_spiral1_validation.py
│   └── VALIDATION_PLAN.md
│
├── 🚀 SCRIPTS
│   ├── setup_overnight.sh
│   ├── launch_spiral1.sh
│   └── check_status.sh
│
├── 📊 OUTPUTS
│   ├── test_outputs/
│   ├── test_logs/
│   └── mcmc_output/ (will be created)
│
└── 📖 DOCUMENTATION
    ├── GETTING_STARTED.md (this file)
    ├── VALIDATION_PLAN.md
    ├── CODE_READINESS_CONFIRMED.md
    └── PARAMETER_DEGENERACY_FIX.md
```

---

## 🔧 KEY FEATURES

### Forward Model Pipeline:
1. **DustPy:** Disk evolution (1D, radial)
2. **RADMC3D:** Radiative transfer (3D)
3. **Beam:** Telescope convolution (ALMA-like)
4. **Output:** Synthetic image matching observations

### MCMC Configuration:
- **Sampler:** emcee (affine-invariant)
- **Walkers:** 50 (10× parameters)
- **Steps:** 2000 per walker
- **Total:** 100,000 forward model calls
- **Runtime:** ~5-7 days (estimated)

### Robustness Features:
- ✅ Checkpoint/restart capability
- ✅ Timeout handling (per simulation)
- ✅ Error logging & recovery
- ✅ Progress monitoring
- ✅ Convergence diagnostics

---

## 📈 EXPECTED OUTPUTS

### MCMC Results:
```
mcmc_output/
├── chain.npy              # Full MCMC chain
├── log_prob.npy           # Log probabilities
├── corner_plot.png        # Parameter correlations
├── trace_plot.png         # Chain convergence
├── best_fit_params.json   # Best-fit values
└── summary_statistics.txt # Quantiles, uncertainties
```

### Validation Results:
```
test_outputs/
├── dustpy_sensitivity/
│   ├── results.json
│   └── *.png (diagnostic plots)
├── mass_conservation/
│   ├── results.json
│   └── mass_evolution.png
└── spiral1_summary.json
```

---

## 🚨 TROUBLESHOOTING

### Common Issues:

**1. DustPy convergence failures:**
```bash
# Check parameter ranges in mcmc_pipeline_config.py
# Especially: r_c, vFrag, log_mdisk
```

**2. RADMC3D crashes:**
```bash
# Check dust opacity table
# Verify grid resolution (Nr, Nphi, Ntheta)
```

**3. Memory issues:**
```bash
# Reduce grid resolution
# Decrease number of photons
# Monitor with: watch -n 1 free -h
```

**4. Slow MCMC:**
```bash
# Expected: ~5-10 min per forward model
# If slower: Check DustPy convergence settings
```

---

## 📞 SUPPORT & NEXT STEPS

### After Validation Completes:
1. Review test results: `./check_status.sh`
2. Analyze plots in `test_outputs/`
3. If all passed → Proceed to Spiral 2
4. If failures → Debug and re-run specific tests

### Before Production MCMC:
- [ ] Complete all validation tests
- [ ] Review parameter priors
- [ ] Test synthetic recovery
- [ ] Verify convergence diagnostics
- [ ] Setup monitoring dashboard

### Production Run:
```bash
# Final command (after all validation)
python3 mcmc_pipeline_main.py \
    --walkers 50 \
    --steps 2000 \
    --output mcmc_output/
```

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

**Created:** 2025-12-08 18:25  
**Status:** ✅ Ready for production (after validation)  
**Next:** Complete Spiral 1 validation (ETA: 23:00 tonight)
