#!/usr/bin/env python3
import os
import argparse
import numpy as np
import emcee
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize, LogNorm

try:
    import corner
    CORNER_AVAILABLE = True
except Exception:
    corner = None
    CORNER_AVAILABLE = False

import mcmc_pipeline_config as config
from likelihood_calculator import prepare_model_for_observation_comparison, get_gaussian_centroid


def _squeeze_to_2d(data: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(data))
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={arr.shape}")
    return arr.astype(float)


def _load_fits_2d(path: str):
    if not path or not os.path.exists(path):
        return None
    with fits.open(path) as hdul:
        return _squeeze_to_2d(hdul[0].data)


def _param_dict_from_vector(vector: np.ndarray) -> dict:
    names = [p['name'] for p in config.MCMC_PARAMETERS]
    return {name: float(val) for name, val in zip(names, vector)}


def _default_param_dict() -> dict:
    return {p['name']: float(p['default']) for p in config.MCMC_PARAMETERS}


def _load_backend_state(backend_path: str):
    if not os.path.exists(backend_path):
        return None

    backend = emcee.backends.HDFBackend(backend_path, read_only=True)
    if backend.iteration <= 0:
        return None

    chain = backend.get_chain(flat=True)
    log_prob = backend.get_log_prob(flat=True)
    best_idx = int(np.argmax(log_prob))

    return {
        "iteration": int(backend.iteration),
        "n_walkers": int(backend.shape[0]),
        "acceptance_rate": float(np.mean(backend.accepted / max(int(backend.iteration), 1))),
        "best_log_prob": float(log_prob[best_idx]),
        "best_params": _param_dict_from_vector(chain[best_idx]),
    }


def generate_trace_plot(config,
                        backend_path: str,
                        iteration_idx: int = None,
                        output_path: str = None,
                        logger=None):
    """Generate MCMC trace plot from backend raw chain (flat=False)."""
    if not backend_path or not os.path.exists(backend_path):
        if logger:
            logger.warning(f"Trace plot skipped: backend not found at {backend_path}")
        return None

    try:
        backend = emcee.backends.HDFBackend(backend_path, read_only=True)
        n_steps = int(backend.iteration)
        if n_steps <= 1:
            if logger:
                logger.warning("Trace plot skipped: backend chain too short.")
            return None

        chain = backend.get_chain(flat=False)
        if chain.ndim != 3:
            if logger:
                logger.warning(f"Trace plot skipped: unexpected chain shape {chain.shape}")
            return None

        n_steps, n_walkers, n_dim = chain.shape
        labels = [p['name'] for p in config.MCMC_PARAMETERS]
        if len(labels) != n_dim:
            labels = [f"param_{i}" for i in range(n_dim)]

        if iteration_idx is None:
            iteration_idx = n_steps

        plt.style.use('dark_background')
        fig, axes = plt.subplots(n_dim, 1, figsize=(14, max(2.2 * n_dim, 6.0)), sharex=True)
        if n_dim == 1:
            axes = [axes]

        x = np.arange(n_steps)
        for dim_idx, ax in enumerate(axes):
            for walker_idx in range(n_walkers):
                ax.plot(x, chain[:, walker_idx, dim_idx], color='deepskyblue', alpha=0.25, lw=0.7)
            ax.set_ylabel(labels[dim_idx], color='white')
            ax.tick_params(axis='both', which='both', direction='in', colors='white')
            ax.grid(alpha=0.15, color='white')
            ax.set_facecolor('black')

        axes[-1].set_xlabel('Iteration', color='white')
        fig.suptitle(f"MCMC Trace Plot — Iteration {iteration_idx}", color='white', fontsize=13)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])

        out_png = output_path or os.path.join(
            config.MCMC_OUTPUT_DIR,
            f"mcmc_trace_iter_{iteration_idx}.png"
        )
        fig.savefig(out_png, dpi=170, bbox_inches='tight')
        plt.close(fig)
        return out_png
    except Exception as e:
        if logger:
            logger.error("Trace plot generation failed", exception=e)
        return None


def generate_corner_plot(config,
                         backend_path: str,
                         best_fit_params: dict,
                         iteration_idx: int = None,
                         output_path: str = None,
                         logger=None):
    """Generate corner plot from flattened backend chain with burn-in discard."""
    if not CORNER_AVAILABLE:
        if logger:
            logger.warning("Corner plot skipped: `corner` library is not available.")
        return None

    if not backend_path or not os.path.exists(backend_path):
        if logger:
            logger.warning(f"Corner plot skipped: backend not found at {backend_path}")
        return None

    try:
        backend = emcee.backends.HDFBackend(backend_path, read_only=True)
        n_steps = int(backend.iteration)
        if n_steps <= 2:
            if logger:
                logger.warning("Corner plot skipped: backend chain too short.")
            return None

        n_walkers = int(backend.shape[0])
        n_dim = int(backend.shape[1])
        discard = int(n_steps * 0.3)
        if discard >= n_steps:
            discard = max(0, n_steps - 1)

        flat_chain = backend.get_chain(discard=discard, flat=True)
        min_required = max(20, 2 * n_dim)
        if flat_chain.shape[0] < min_required and discard > 0:
            if logger:
                logger.warning(
                    f"Corner chain too short after discard={discard}; retrying with discard=0."
                )
            discard = 0
            flat_chain = backend.get_chain(discard=discard, flat=True)

        if flat_chain.shape[0] < min_required:
            if logger:
                logger.warning(
                    f"Corner plot skipped: insufficient samples ({flat_chain.shape[0]})."
                )
            return None

        labels = [p['name'] for p in config.MCMC_PARAMETERS]
        if len(labels) != n_dim:
            labels = [f"param_{i}" for i in range(n_dim)]
        truths = [float(best_fit_params.get(name, np.nan)) for name in labels]

        if iteration_idx is None:
            iteration_idx = n_steps

        fig = corner.corner(
            flat_chain,
            labels=labels,
            truths=truths,
            show_titles=True,
            title_fmt=".4g",
            quantiles=[0.16, 0.5, 0.84],
            color='deepskyblue',
            truth_color='tomato',
            plot_datapoints=True,
            fill_contours=True,
        )
        fig.set_size_inches(max(10, 2.2 * n_dim), max(10, 2.2 * n_dim))
        fig.patch.set_facecolor('black')
        for ax in fig.get_axes():
            ax.set_facecolor('black')
            ax.tick_params(axis='both', which='both', direction='in', colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            if ax.xaxis.label:
                ax.xaxis.label.set_color('white')
            if ax.yaxis.label:
                ax.yaxis.label.set_color('white')
            if ax.title:
                ax.title.set_color('white')

        fig.suptitle(
            f"MCMC Corner Plot — Iteration {iteration_idx} | Burn-in discard={discard} steps",
            color='white',
            fontsize=13,
            y=0.995,
        )

        out_png = output_path or os.path.join(
            config.MCMC_OUTPUT_DIR,
            f"mcmc_corner_iter_{iteration_idx}.png"
        )
        fig.savefig(out_png, dpi=170, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_png
    except Exception as e:
        if logger:
            logger.error("Corner plot generation failed", exception=e)
        return None


def generate_ultimate_9panel_diagnostic(config,
                                        obs_data,
                                        best_fit_params: dict,
                                        best_log_prob: float,
                                        n_steps: int,
                                        n_walkers: int,
                                        output_path: str = None,
                                        logger=None,
                                        acceptance_rate: float = np.nan,
                                        iteration_idx: int = None,
                                        simulator=None,
                                        best_model_data=None):
    """Core plotting engine (library mode) for the unified 9-panel diagnostic."""
    if obs_data is None:
        if logger:
            logger.warning("Observation image unavailable — skip 9-panel diagnostic.")
        return None

    obs = np.asarray(obs_data, dtype=float)

    if best_model_data is not None:
        best_model_raw = np.asarray(best_model_data, dtype=float)
    else:
        if simulator is None:
            if logger:
                logger.error("No simulator provided and no precomputed model FITS provided.")
            return None
        try:
            ok_best, best_model_raw, _ = simulator.simulate(best_fit_params)
        except Exception as e:
            if logger:
                logger.error("Simulation failed while generating 9-panel diagnostic", exception=e)
            return None

        if not ok_best or best_model_raw is None:
            if logger:
                logger.warning("Best-fit simulation failed — skip 9-panel diagnostic.")
            return None

    # CRITICAL RESIDUAL MATH: apply shift + ALMA beam convolution before subtraction
    best_model = prepare_model_for_observation_comparison(
        model_data=np.asarray(best_model_raw, dtype=float),
        dx_shift=float(config.DX_SHIFT),
        dy_shift=float(config.DY_SHIFT),
        apply_beam_convolution=True,
    )

    if obs.shape != best_model.shape:
        if logger:
            logger.error(f"Shape mismatch: obs={obs.shape}, best={best_model.shape}")
        return None

    residual_best = obs - best_model
    sigma_residual = residual_best / float(config.RMS_NOISE_JY)
    obs_cx, obs_cy = get_gaussian_centroid(obs)
    model_cx, model_cy = get_gaussian_centroid(best_model)

    ny, nx = obs.shape
    pixel_scale_arcsec = (config.IMAGE_SIZE_AU / config.IMAGE_NPIX) / config.DISTANCE_PC
    half_x = 0.5 * nx * pixel_scale_arcsec
    half_y = 0.5 * ny * pixel_scale_arcsec
    extent = [-half_x, half_x, -half_y, half_y]

    # Shared anchoring: P1/P2/P4/P5 locked to ALMA peak flux
    vmax = float(np.nanmax(obs))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    if vmax <= 1e-4:
        vmax = 1e-3
    image_norm = Normalize(vmin=0.0, vmax=vmax)
    log_vmin = max(3.0 * float(config.RMS_NOISE_JY), 1e-12)
    log_norm = LogNorm(vmin=log_vmin, vmax=vmax)
    image_cmap = 'magma'
    rainbow_cmap = plt.get_cmap('nipy_spectral').copy()
    rainbow_cmap.set_under('black')
    sigma_norm = Normalize(vmin=-5.0, vmax=5.0)

    obs_peak = np.nanmax(obs)
    best_peak = np.nanmax(best_model)
    contour_obs = np.array([0.10, 0.30, 0.50, 0.70, 0.90]) * obs_peak
    contour_best = np.array([0.10, 0.30, 0.50, 0.70, 0.90]) * best_peak

    y_idx, x_idx = np.indices(obs.shape)

    def _radial_profile(image, center_y, center_x):
        r_pix = np.hypot(x_idx - center_x, y_idx - center_y)
        r_int = r_pix.astype(int)
        finite = np.isfinite(image)
        if not np.any(finite):
            return np.array([]), np.array([])
        sum_r = np.bincount(r_int[finite].ravel(), weights=image[finite].ravel())
        n_r = np.bincount(r_int[finite].ravel())
        valid = n_r > 0
        prof = np.zeros_like(sum_r, dtype=float)
        prof[valid] = sum_r[valid] / n_r[valid]
        radii_pix = np.arange(len(prof), dtype=float)
        return radii_pix, prof

    peak_y, peak_x = np.unravel_index(np.nanargmax(obs), obs.shape)
    r_obs_pix, p_obs = _radial_profile(obs, peak_y, peak_x)
    r_best_pix, p_best = _radial_profile(best_model, peak_y, peak_x)
    r_obs_au = r_obs_pix * pixel_scale_arcsec * config.DISTANCE_PC
    r_best_au = r_best_pix * pixel_scale_arcsec * config.DISTANCE_PC

    zoom_half = 2
    y0 = max(0, int(round(peak_y)) - zoom_half)
    y1 = min(ny, int(round(peak_y)) + zoom_half + 1)
    x0 = max(0, int(round(peak_x)) - zoom_half)
    x1 = min(nx, int(round(peak_x)) + zoom_half + 1)
    centroid_cutout = obs[y0:y1, x0:x1]

    if iteration_idx is None:
        iteration_idx = int(n_steps)

    # Dark publication style
    plt.style.use('dark_background')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor('black')
    gs = gridspec.GridSpec(3, 3, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)

    def _add_beam(ax):
        ex, ey = extent[0], extent[2]
        span_x = extent[1] - extent[0]
        span_y = extent[3] - extent[2]
        bx = ex + 0.13 * span_x
        by = ey + 0.13 * span_y
        beam = Ellipse(
            xy=(bx, by),
            width=config.BEAM_MINOR_ARCSEC,
            height=config.BEAM_MAJOR_ARCSEC,
            angle=float(config.BEAM_PA_DEG),
            facecolor='none',
            edgecolor='white',
            lw=1.4
        )
        ax.add_patch(beam)

    def _style_axis(ax):
        ax.set_facecolor('black')
        ax.tick_params(axis='both', which='both', direction='in', labelsize=10, colors='white')
        ax.minorticks_on()
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

    def _style_cbar(cbar, label):
        cbar.set_label(label, color='white')
        cbar.ax.tick_params(which='both', direction='in', labelsize=10, colors='white')

    # ── ROW 1: Linear intensity & geometry ─────────────────────────────
    # P1 - ALMA Observation (Linear)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(obs, origin='lower', extent=extent, cmap=image_cmap, norm=image_norm)
    _add_beam(ax1)
    ax1.set_title('P1 — ALMA Observation (Linear)')
    ax1.set_xlabel(r'$\Delta\alpha$ (arcsec)')
    ax1.set_ylabel(r'$\Delta\delta$ (arcsec)')
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    _style_cbar(cbar1, r'$Jy/beam$')
    _style_axis(ax1)

    # P2 - Golden Model (Linear)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(best_model, origin='lower', extent=extent, cmap=image_cmap, norm=image_norm)
    _add_beam(ax2)
    ax2.set_title(r'P2 — Golden Model (Linear)')
    ax2.set_xlabel(r'$\Delta\alpha$ (arcsec)')
    ax2.set_ylabel(r'$\Delta\delta$ (arcsec)')
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    _style_cbar(cbar2, r'$Jy/beam$')
    _style_axis(ax2)

    # P3 - Contours Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(np.zeros_like(obs), origin='lower', extent=extent, cmap='gray', vmin=0.0, vmax=1.0)
    ax3.contour(obs, levels=contour_obs, origin='lower', extent=extent,
                colors='white', linewidths=1.2)
    ax3.contour(best_model, levels=contour_best, origin='lower', extent=extent,
                colors='magenta', linewidths=1.0, linestyles='--')
    _add_beam(ax3)
    ax3.set_title('P3 — Contours Overlay')
    ax3.set_xlabel(r'$\Delta\alpha$ (arcsec)')
    ax3.set_ylabel(r'$\Delta\delta$ (arcsec)')
    _style_axis(ax3)

    # ── ROW 2: Log-scale rainbow + sigma residual ──────────────────────
    # P4 - ALMA Observation (Log)
    ax4 = fig.add_subplot(gs[1, 0])
    obs_log = np.where(obs > 0.0, obs, np.nan)
    im4 = ax4.imshow(obs_log, origin='lower', extent=extent, cmap=rainbow_cmap, norm=log_norm)
    _add_beam(ax4)
    ax4.set_title('P4 — ALMA Observation (Log)')
    ax4.set_xlabel(r'$\Delta\alpha$ (arcsec)')
    ax4.set_ylabel(r'$\Delta\delta$ (arcsec)')
    cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.02)
    _style_cbar(cbar4, r'$Jy/beam$')
    _style_axis(ax4)

    # P5 - Golden Model (Log)
    ax5 = fig.add_subplot(gs[1, 1])
    model_log = np.where(best_model > 0.0, best_model, np.nan)
    im5 = ax5.imshow(model_log, origin='lower', extent=extent, cmap=rainbow_cmap, norm=log_norm)
    _add_beam(ax5)
    ax5.set_title('P5 — Golden Model (Log)')
    ax5.set_xlabel(r'$\Delta\alpha$ (arcsec)')
    ax5.set_ylabel(r'$\Delta\delta$ (arcsec)')
    cbar5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.02)
    _style_cbar(cbar5, r'$Jy/beam$')
    _style_axis(ax5)

    # P6 - Residual sigma map
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(sigma_residual, origin='lower', extent=extent, cmap='RdBu_r', norm=sigma_norm)
    ax6.set_title(r'P6 — Residual Sigma Map: $(Data-Model)/\sigma$')
    ax6.set_xlabel(r'$\Delta\alpha$ (arcsec)')
    ax6.set_ylabel(r'$\Delta\delta$ (arcsec)')
    cbar6 = fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.02)
    _style_cbar(cbar6, r'$\sigma$')
    _style_axis(ax6)

    # ── ROW 3: Quantitative analysis ────────────────────────────────────
    # P7 - True azimuthal average radial profile
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(r_obs_au, p_obs, 'o', ms=3.2, color='deepskyblue', alpha=0.9, label='ALMA (azimuthal avg)')
    ax7.plot(r_best_au, p_best, '-', lw=2.0, color='gold', label='Model (azimuthal avg)')
    if 'r_c' in best_fit_params:
        ax7.axvline(best_fit_params['r_c'], color='tomato', lw=1.3, ls='-', label=r'$R_c$')
    if 'r_in' in best_fit_params:
        ax7.axvline(best_fit_params['r_in'], color='seagreen', lw=1.3, ls='-.', label=r'$R_{in}$')
    ax7.text(0.02, 0.03,
             'Profile is symmetric azimuthal average; asymmetry is captured in P6.',
             transform=ax7.transAxes, fontsize=8.5, color='white', ha='left', va='bottom')
    ax7.set_title('P7 — Radial Profile Evolution')
    ax7.set_xlabel(r'Radius ($AU$)')
    ax7.set_ylabel(r'Intensity ($Jy/beam$)')
    ax7.grid(alpha=0.25, color='white')
    ax7.legend(fontsize=8, loc='best', facecolor='black', edgecolor='white')
    _style_axis(ax7)

    # P8 - Centroid zoom (5x5 px)
    ax8 = fig.add_subplot(gs[2, 1])
    cut_extent = [x0 - peak_x, x1 - peak_x, y0 - peak_y, y1 - peak_y]
    ax8.imshow(centroid_cutout, origin='lower', cmap='magma', extent=cut_extent, interpolation='nearest')
    ax8.scatter(obs_cx - peak_x, obs_cy - peak_y,
                marker='x', s=75, c='white', linewidths=2.0, label='Obs G-centroid')
    ax8.scatter(model_cx - peak_x, model_cy - peak_y,
                marker='o', s=60, facecolors='none', edgecolors='red', linewidths=1.7,
                label='Model G-centroid')
    ax8.set_title('P8 — Centroid Zoom (5×5 px)')
    ax8.set_xlabel('Pixel offset x')
    ax8.set_ylabel('Pixel offset y')
    ax8.legend(fontsize=8, loc='upper right', facecolor='black', edgecolor='white')
    ax8.grid(alpha=0.2, color='white')
    _style_axis(ax8)

    # P9 - Advanced stats summary
    finite_sigma = np.isfinite(sigma_residual)
    chi2 = float(np.sum((sigma_residual[finite_sigma])**2)) if np.any(finite_sigma) else np.nan
    dof = max(int(np.sum(finite_sigma)) - len(config.MCMC_PARAMETERS), 1)
    chi2_nu = chi2 / dof if np.isfinite(chi2) else np.nan

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_facecolor('black')
    ax9.axis('off')
    lines = [
        r'P9 — Advanced Stats Summary',
        r'',
        rf'$\ln(L)_{{best}} = {best_log_prob:.4f}$',
        rf'$\chi^2_\nu = {chi2_nu:.4f}$',
        r'',
        r'Best-fit parameters:',
    ]
    for p in config.MCMC_PARAMETERS:
        name = p['name']
        unit = p.get('unit', '')
        val = best_fit_params.get(name, np.nan)
        lines.append(f"  {name:10s} = {val:.6g} {unit}")
    lines += [
        r'',
        f"Acceptance rate = {acceptance_rate:.4f}" if np.isfinite(acceptance_rate) else "Acceptance rate = n/a",
        f"Chain iteration = {iteration_idx}",
        f"Walkers = {n_walkers}",
        rf"RMS noise = {config.RMS_NOISE_JY:.2e} $Jy/beam$",
    ]
    ax9.text(
        0.02, 0.98,
        "\n".join(lines),
        va='top', ha='left',
        color='white',
        fontsize=9,
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', fc='black', ec='white', lw=0.8)
    )

    fig.suptitle(
        rf"Ultimate Dark-Mode Diagnostic — Iteration {iteration_idx} | "
        rf"Best $\ln(L)$={best_log_prob:.2f}",
        color='white',
        fontsize=14,
        y=0.98
    )

    out_png = output_path or os.path.join(
        config.MCMC_OUTPUT_DIR,
        f"ultimate_9panel_diagnostic_iter_{iteration_idx}.png"
    )
    fig.savefig(out_png, dpi=170, bbox_inches='tight')
    plt.close(fig)
    if logger:
        logger.info(f"✅ Ultimate 9-panel diagnostic saved: {out_png}")
    return out_png


def _resolve_obs_path(results_dir: str, obs_arg: str = None) -> str:
    if obs_arg and os.path.exists(obs_arg):
        return obs_arg

    candidates = [
        os.path.join(results_dir, "observation.fits"),
        os.path.join(results_dir, "obs.fits"),
        config.OBS_FITS_PATH,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Observation FITS not found. Checked: " + ", ".join(candidates)
    )


def main():
    parser = argparse.ArgumentParser(description="Unified MCMC Visualizer (library + execution mode)")
    parser.add_argument("--results-dir", default=config.MCMC_OUTPUT_DIR, help="Directory with HDF5/FITS results")
    parser.add_argument("--obs", default=None, help="Optional observation FITS path override")
    parser.add_argument("--best-fits", default=None, help="Optional best-fit FITS path override")
    parser.add_argument("--backend", default=None, help="Optional backend path override")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    backend_path = os.path.abspath(args.backend) if args.backend else os.path.join(results_dir, config.MCMC_BACKEND_FILENAME)
    best_fits_path = os.path.abspath(args.best_fits) if args.best_fits else os.path.join(results_dir, "best_fit_model.fits")
    output_path = os.path.abspath(args.output) if args.output else os.path.join(results_dir, "ultimate_diagnostic_dark.png")

    obs_path = _resolve_obs_path(results_dir, args.obs)
    obs_data = _load_fits_2d(obs_path)

    backend_state = _load_backend_state(backend_path)
    if backend_state is None:
        print(f"⚠️ Backend unavailable or empty: {backend_path}", flush=True)

    best_model_data = _load_fits_2d(best_fits_path)

    best_params = backend_state["best_params"] if backend_state else _default_param_dict()
    best_log_prob = backend_state["best_log_prob"] if backend_state else np.nan
    n_steps = backend_state["iteration"] if backend_state else 0
    n_walkers = backend_state["n_walkers"] if backend_state else int(config.N_WALKERS)
    acceptance_rate = backend_state["acceptance_rate"] if backend_state else np.nan
    iteration_idx = n_steps

    simulator = None
    if best_model_data is None:
        print("ℹ️ Missing best-fit FITS. Falling back to HDF5 + single best-fit simulation.", flush=True)
        if backend_state is None:
            raise RuntimeError(
                "Cannot reconstruct model: best-fit FITS missing and backend is unavailable."
            )
        from forward_simulator import ForwardModelSimulatorV2
        simulator = ForwardModelSimulatorV2(cleanup=True)
        ok_best, generated_model, _ = simulator.simulate(best_params)
        if not ok_best or generated_model is None:
            raise RuntimeError("Failed to generate fresh best-fit model from backend parameters.")
        best_model_data = np.asarray(generated_model, dtype=float)
        best_model_path = os.path.join(results_dir, "best_model_image.fits")
        fits.PrimaryHDU(best_model_data).writeto(best_model_path, overwrite=True)
        print(f"✅ Saved fresh best model FITS: {best_model_path}", flush=True)

    saved = generate_ultimate_9panel_diagnostic(
        config=config,
        obs_data=obs_data,
        best_fit_params=best_params,
        best_log_prob=best_log_prob,
        n_steps=n_steps,
        n_walkers=n_walkers,
        output_path=output_path,
        logger=None,
        acceptance_rate=acceptance_rate,
        iteration_idx=iteration_idx,
        simulator=simulator,
        best_model_data=best_model_data,
    )

    if not saved:
        raise RuntimeError("Failed to generate unified 9-panel diagnostic.")

    trace_path = generate_trace_plot(
        config=config,
        backend_path=backend_path,
        iteration_idx=iteration_idx,
        logger=None,
    )

    corner_path = generate_corner_plot(
        config=config,
        backend_path=backend_path,
        best_fit_params=best_params,
        iteration_idx=iteration_idx,
        logger=None,
    )

    print(f"✅ Unified diagnostic saved: {saved}", flush=True)
    if trace_path:
        print(f"✅ Trace plot saved: {trace_path}", flush=True)
    else:
        print("⚠️ Trace plot was not generated.", flush=True)
    if corner_path:
        print(f"✅ Corner plot saved: {corner_path}", flush=True)
    else:
        print("⚠️ Corner plot was not generated.", flush=True)


if __name__ == "__main__":
    main()
