"""LIFEsimMC — Single Epoch Observation GUI."""
import base64
import inspect
import json
import re
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from pathlib import Path

_here = Path(__file__).resolve().parent
_candidate = _here.parent.parent.parent
if (_candidate / "lifesimmc" / "util" / "spectrum.py").exists():
    sys.path.insert(0, str(_candidate))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
import torch
from astropy import units as u


# ── Docstring tooltip extraction ───────────────────────────────────────────────
def _parse_docstring_params(cls):
    """Return {param_name: description} parsed from a NumPy-style docstring."""
    doc = inspect.getdoc(cls) or ""
    params = {}
    in_params = False
    current_param = None
    current_lines = []

    for line in doc.split("\n"):
        if line.strip() == "Parameters":
            in_params = True
            continue
        if in_params and re.match(r"^-{3,}$", line.strip()):
            continue
        if not in_params:
            continue
        # Non-indented non-empty line that isn't a "name : type" header → new section
        if line and not line.startswith(" ") and not re.match(r"^\w+\s*:", line):
            break
        param_match = re.match(r"^(\w+)\s*:", line)
        if param_match and not line.startswith(" "):
            if current_param and current_lines:
                params[current_param] = " ".join(current_lines).strip()
            current_param = param_match.group(1)
            current_lines = []
        elif line.startswith(" ") and current_param:
            t = line.strip()
            if t:
                current_lines.append(t)

    if current_param and current_lines:
        params[current_param] = " ".join(current_lines).strip()
    return params


def _build_tips():
    """Build elem_id → tooltip text from class docstrings."""
    tips = {}

    try:
        from phringe.core.sources.star import Star
        p = _parse_docstring_params(Star)
        tips["p-star-dist"] = p.get("distance", "")
        tips["p-star-temp"] = p.get("temperature", "")
        tips["p-star-mass"] = p.get("mass", "")
        tips["p-star-rad"] = p.get("radius", "")
        tips["p-star-ra"] = p.get("right_ascension", "")
        tips["p-star-dec"] = p.get("declination", "")
    except Exception:
        pass

    try:
        from phringe.core.sources.planet import Planet
        p = _parse_docstring_params(Planet)
        tips["p-planet-sma"] = p.get("semi_major_axis", "")
        tips["p-planet-temp"] = p.get("temperature", "")
        tips["p-planet-mass"] = p.get("mass", "")
        tips["p-planet-rad"] = p.get("radius", "")
        tips["p-planet-ecc"] = p.get("eccentricity", "")
        tips["p-planet-incl"] = p.get("inclination", "")
        tips["p-planet-raan"] = p.get("raan", "")
        tips["p-planet-aop"] = p.get("argument_of_periapsis", "")
        tips["p-planet-ta"] = p.get("true_anomaly", "")
        tips["p-sed-mode"] = p.get("sed_loader", "")
    except Exception:
        pass

    try:
        from phringe.core.sources.exozodi import Exozodi
        p = _parse_docstring_params(Exozodi)
        tips["p-exozodi-level"] = p.get("level", "")
    except Exception:
        pass

    try:
        from lifesimmc.presets.single_epoch_observation.single_epoch_observation import SingleEpochObservation
        p = _parse_docstring_params(SingleEpochObservation)
        tips["p-tot-int"] = p.get("total_integration_time", "")
        tips["p-spec-res"] = p.get("spectral_resolving_power", "")
        tips["p-det-int"] = p.get("detector_integration_time", "")
        tips["p-mod-val"] = p.get("modulation_period", "")
        tips["p-sol-ecl-lat"] = p.get("solar_ecliptic_latitude", "")
        tips["p-custom-bl"] = p.get("nulling_baseline", "")
        tips["p-ap-diam"] = p.get("aperture_diameter", "")
        tips["p-throughput"] = p.get("throughput", "")
        tips["p-qe"] = p.get("quantum_efficiency", "")
        tips["p-bl-min"] = p.get("nulling_baseline_min", "")
        tips["p-bl-max"] = p.get("nulling_baseline_max", "")
        tips["p-wl-min"] = p.get("wavelength_min", "")
        tips["p-wl-max"] = p.get("wavelength_max", "")
        tips["p-noise"] = p.get("instrumental_noise", "")
        tips["p-templ-fov"] = p.get("template_fov_rad", "")
        tips["p-seed"] = p.get("seed", "")
        tips["p-grid-size"] = p.get("grid_size", "")
        tips["p-device"] = p.get("device", "")
        tips["p-ref-rad"] = p.get("host_star_radius", "")
        tips["p-ref-temp"] = p.get("host_star_temperature", "")
        tips["p-ref-mass"] = p.get("host_star_mass", "")
        tips["p-ref-dist"] = p.get("host_star_distance", "")
        tips["p-ref-ra"] = p.get("host_star_right_ascension", "")
        tips["p-ref-dec"] = p.get("host_star_declination", "")
    except Exception:
        pass

    return {k: v for k, v in tips.items() if v}


_TIPS = _build_tips()

# ── Theme ──────────────────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0f1117",
    body_background_fill_dark="#0f1117",
    block_background_fill="#161b27",
    block_background_fill_dark="#161b27",
    block_border_color="#252d3d",
    block_border_color_dark="#252d3d",
    block_label_text_color="#5a6480",
    block_label_text_color_dark="#5a6480",
    input_background_fill="#0d1117",
    input_background_fill_dark="#0d1117",
    input_border_color="#252d3d",
    input_border_color_dark="#252d3d",
    button_primary_background_fill="linear-gradient(135deg,#4f83f5,#2952e3)",
    button_primary_background_fill_dark="linear-gradient(135deg,#4f83f5,#2952e3)",
    button_primary_background_fill_hover="linear-gradient(135deg,#6494ff,#3b62f0)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg,#6494ff,#3b62f0)",
    button_primary_text_color="#fff",
    button_primary_text_color_dark="#fff",
    button_secondary_background_fill="#1a2030",
    button_secondary_background_fill_dark="#1a2030",
    button_secondary_border_color="#2d3650",
    button_secondary_border_color_dark="#2d3650",
    button_secondary_text_color="#8898b8",
    button_secondary_text_color_dark="#8898b8",
    body_text_color="#c9d1e0",
    body_text_color_dark="#c9d1e0",
    border_color_primary="#252d3d",
    border_color_primary_dark="#252d3d",
    background_fill_primary="#0f1117",
    background_fill_primary_dark="#0f1117",
    background_fill_secondary="#161b27",
    background_fill_secondary_dark="#161b27",
    color_accent="#5b8af5",
    color_accent_soft="#2952e3",
    shadow_drop="0 4px 20px rgba(0,0,0,0.5)",
    shadow_drop_lg="0 8px 36px rgba(0,0,0,0.6)",
    checkbox_background_color="#0d1117",
    checkbox_background_color_dark="#0d1117",
    checkbox_border_color="#2d3650",
    checkbox_border_color_dark="#2d3650",
)

CSS = """
footer { display: none !important; }
.source-card { border-radius: 12px !important;}

.panel-results .source-card {
    border-radius: 12px !important;
    overflow: hidden;
}

.panel-results .source-card .block {
    background: var(--block-background-fill) !important;
}

.avg-runs {
    background: none !important;
    border: none !important;
    flex: 0 0 auto !important;
    width: auto !important;
}

/* section labels */
.sec-label {
    font-size: 0.6rem; font-weight: 800; letter-spacing: 0.18em;
    text-transform: uppercase; color: #5b8af5;
    margin: 1.1rem 0 0.5rem; padding-bottom: 0.3rem;
    border-bottom: 1px solid #2040a0;
}

/* detection significance card */
.metric-card {
    background: #111828; border: 1px solid #2952e3;
    border-radius: 12px; padding: 1.4rem; text-align: center;
    margin-top: 0.5rem;
}
.metric-val   { font-size: 2.4rem; font-weight: 700; color: #6aaaf9; line-height: 1.1; }
.metric-label { font-size: 0.68rem; color: #4a5570; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 0.35rem; }
.metric-sub   { font-size: 0.75rem; color: #3a4460; margin-top: 0.5rem; }

/* Accordions */
.block:has(> button.label-wrap) { border-radius: 0px 0px 12px 12px !important; }
button.label-wrap { border-radius: 0px 0px 12px 12px !important; }

/* spinner */
@keyframes lsm-spin { to { transform: rotate(360deg); } }
.spin { display: inline-block; animation: lsm-spin 0.9s linear infinite; }

/* status bar */
.status-bar {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; border-radius: 12px;
    border: 1px solid #252d3d; background: #0d1117;
    font-size: 0.82rem;
}
.dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }

/* log */
.log-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important; line-height: 1.5 !important;
    background: #0a0e17 !important; color: #8898b8 !important;
    border: 1px solid #1e2538 !important;
}

/* Buttons keep 12px (they are standalone, not clipped by a container) */
button, .btn,
button.lg, button.sm, button.md,
button.primary, button.secondary { border-radius: 12px !important; }

/* Ensure number inputs show their value in the dark theme */
input[type="number"] { color: #c9d1e0 !important; }

/* ── Info tooltip button ───────────────────────────────────────────────────── */
.info-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #141e33;
    border: 1px solid #2d4070;
    color: #4f72d4;
    font-size: 9px;
    font-weight: 800;
    cursor: default;
    vertical-align: middle;
    margin-left: 5px;
    line-height: 1;
    font-style: normal;
    flex-shrink: 0;
    user-select: none;
    transition: border-color 0.12s, color 0.12s, background 0.12s;
}
.info-btn:hover {
    border-color: #5b8af5;
    color: #7aa8ff;
    background: #1a2845;
}
"""

# ── Cache for replot ───────────────────────────────────────────────────────────
_cache: dict = {}

# ── Plot helpers ───────────────────────────────────────────────────────────────
_BG = "#0f1117"


def _ax_style(fig, ax):
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color("#252d3d")
    ax.tick_params(colors="#5a6480", labelsize=8)
    ax.xaxis.label.set_color("#8898b8")
    ax.yaxis.label.set_color("#8898b8")
    ax.grid(color="#161b27", linewidth=0.5, linestyle="--")


def _sed_fig(wl, sed, std, sed_true, wl_u, sed_u, ylim_min=None, ylim_max=None):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    _ax_style(fig, ax)
    ax.fill_between(wl, np.array(sed_true) - np.array(std),
                    np.array(sed_true) + np.array(std),
                    color="#4f83f5", alpha=0.13, lw=0, zorder=0)
    ax.errorbar(wl, sed, yerr=np.stack([std, std]), fmt="none",
                ecolor="#2d3650", capsize=1.5, capthick=0.5, lw=0.5, zorder=1)
    ax.scatter(wl, sed, color="#6aaaf9", s=16, zorder=2, marker=".", label="Estimated")
    ax.plot(wl, sed_true, color="#c9d1e0", lw=0.9, alpha=0.55,
            linestyle="--", zorder=1, label="True")
    ax.set_xlabel(f"Wavelength ({wl_u})")
    ax.set_ylabel(f"SED ({sed_u})")
    ax.legend(facecolor="#161b27", edgecolor="#252d3d", labelcolor="#8898b8", fontsize=8)
    if ylim_min is not None and ylim_max is not None and ylim_min < ylim_max:
        ax.set_ylim(ylim_min, ylim_max)
    fig.tight_layout(pad=0.5)
    return fig


def _cov_fig(cov, scale="Linear", linthresh=None):
    vmax = max(abs(float(np.nanmin(cov))), abs(float(np.nanmax(cov)))) or 1.0
    if scale == "SymLog":
        try:
            lt = float(linthresh) if linthresh else None
        except (TypeError, ValueError):
            lt = None
        lt = lt if (lt is not None and lt > 0) else vmax * 1e-3
        norm = mcolors.SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax, base=10)
    else:
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    fig, ax = plt.subplots(figsize=(5, 5))
    _ax_style(fig, ax)
    im = ax.imshow(cov, origin="lower", cmap="bwr", norm=norm)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#5a6480", labelsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Wavelength Channel")
    ax.set_ylabel("Wavelength Channel")
    fig.tight_layout(pad=0.5)
    return fig


def _map_fig(fmap):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "lsm", [mcolors.hex2color(c) for c in ["#000", "#2749f4", "#4aaaf9", "#fff"]], N=256
    )
    sz = fmap.shape[1]
    fig, ax = plt.subplots(figsize=(5, 5))
    _ax_style(fig, ax)
    im = ax.imshow(fmap, cmap=cmap)
    ax.plot(sz / 2 - 0.5, sz / 2 - 0.5, marker="*", ms=17, color="white", zorder=5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#4a4a4a", labelsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("RA [mas]")
    ax.set_ylabel("Dec [mas]")
    fig.tight_layout(pad=0.3)
    return fig


def _safe_unit_name(unit):
    return (
        str(unit)
        .replace("/", "_per_")
        .replace("*", "_")
        .replace(" ", "")
        .replace("^", "")
    )


def _zip_tmp_from_cache(disp_sed_units, disp_wl_units):
    from lifesimmc.util.spectrum import convert_spectral_units, convert_wavelength_units

    r = _cache
    n_runs = r.get("n_runs", 1)

    sed_unit_name = _safe_unit_name(disp_sed_units)
    wl_unit_name = _safe_unit_name(disp_wl_units)

    wl = convert_wavelength_units(r["wl_raw"], units_in="m", units_out=disp_wl_units)

    sed_true = convert_spectral_units(
        r["sed_true_raw"], r["wl_raw"],
        units_in="ph/s/m3", units_out=disp_sed_units, wavelength_units="m",
    )

    f = np.asarray(convert_spectral_units(
        np.ones(len(r["wl_raw"])), r["wl_raw"],
        units_in="ph/s/m3", units_out=disp_sed_units, wavelength_units="m",
    ))

    if n_runs > 1 and r.get("all_sed_raws") is not None:
        sed_est = np.stack([
            convert_spectral_units(s, r["wl_raw"], units_in="ph/s/m3",
                                   units_out=disp_sed_units, wavelength_units="m")
            for s in r["all_sed_raws"]
        ])
        std_out = np.stack([
            convert_spectral_units(s, r["wl_raw"], units_in="ph/s/m3",
                                   units_out=disp_sed_units, wavelength_units="m")
            for s in r["all_std_raws"]
        ])
        cov_out = np.stack([np.outer(f, f) * c for c in r["all_cov_raws"]])
        fmap_out = r["all_filt_maps"]
    else:
        sed_est = convert_spectral_units(
            r["sed_raw"], r["wl_raw"],
            units_in="ph/s/m3", units_out=disp_sed_units, wavelength_units="m",
        )
        std_out = convert_spectral_units(
            r["std_raw"], r["wl_raw"],
            units_in="ph/s/m3", units_out=disp_sed_units, wavelength_units="m",
        )
        cov_out = np.outer(f, f) * r["cov_raw"]
        fmap_out = r["filt_map"]

    path = Path(tempfile.gettempdir()) / "lifesimmc_results.zip"

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for filename, arr in [
            (f"wavelengths_{wl_unit_name}.npy", wl),
            (f"sed_estimated_{sed_unit_name}.npy", sed_est),
            (f"sed_true_{sed_unit_name}.npy", sed_true),
            (f"std_{sed_unit_name}.npy", std_out),
            (f"covariance_{sed_unit_name}2.npy", cov_out),
            ("matched_filter.npy", fmap_out),
        ]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                np.save(tmp, np.asarray(arr))
                tmp_path = tmp.name
            zf.write(tmp_path, arcname=filename)
            Path(tmp_path).unlink(missing_ok=True)

    return str(path)


# ── Replot (unit / scale changes after run) ────────────────────────────────────
def replot(disp_sed_units, disp_wl_units, ylim_min, ylim_max, cov_scale, linthresh):
    if not _cache:
        return None, None, gr.update()
    from lifesimmc.util.spectrum import convert_spectral_units, convert_wavelength_units
    r = _cache
    wl = convert_wavelength_units(r["wl_raw"], units_in="m", units_out=disp_wl_units)
    sed = convert_spectral_units(r["sed_raw"], r["wl_raw"], units_in="ph/s/m3",
                                 units_out=disp_sed_units, wavelength_units="m")
    std = convert_spectral_units(r["std_raw"], r["wl_raw"], units_in="ph/s/m3",
                                 units_out=disp_sed_units, wavelength_units="m")
    sed_true = convert_spectral_units(r["sed_true_raw"], r["wl_raw"], units_in="ph/s/m3",
                                      units_out=disp_sed_units, wavelength_units="m")
    f = np.asarray(convert_spectral_units(
        np.ones(len(r["wl_raw"])), r["wl_raw"], units_in="ph/s/m3",
        units_out=disp_sed_units, wavelength_units="m"))
    cov = np.outer(f, f) * r["cov_raw"]
    vmax = max(abs(float(np.nanmin(cov))), abs(float(np.nanmax(cov)))) or 1.0
    new_linthresh = f"{vmax * 1e-3:.1e}"
    plt.close("all")
    return (
        _sed_fig(wl, sed, std, sed_true, disp_wl_units, disp_sed_units,
                 ylim_min=ylim_min, ylim_max=ylim_max),
        _cov_fig(cov, scale=cov_scale, linthresh=new_linthresh),
        new_linthresh,
    )


def replot_nolt(disp_sed_units, disp_wl_units, ylim_min, ylim_max, cov_scale, linthresh):
    if not _cache:
        return None, None
    from lifesimmc.util.spectrum import convert_spectral_units, convert_wavelength_units
    r = _cache
    wl = convert_wavelength_units(r["wl_raw"], units_in="m", units_out=disp_wl_units)
    sed = convert_spectral_units(r["sed_raw"], r["wl_raw"], units_in="ph/s/m3",
                                 units_out=disp_sed_units, wavelength_units="m")
    std = convert_spectral_units(r["std_raw"], r["wl_raw"], units_in="ph/s/m3",
                                 units_out=disp_sed_units, wavelength_units="m")
    sed_true = convert_spectral_units(r["sed_true_raw"], r["wl_raw"], units_in="ph/s/m3",
                                      units_out=disp_sed_units, wavelength_units="m")
    f = np.asarray(convert_spectral_units(
        np.ones(len(r["wl_raw"])), r["wl_raw"], units_in="ph/s/m3",
        units_out=disp_sed_units, wavelength_units="m"))
    cov = np.outer(f, f) * r["cov_raw"]
    plt.close("all")
    return (
        _sed_fig(wl, sed, std, sed_true, disp_wl_units, disp_sed_units,
                 ylim_min=ylim_min, ylim_max=ylim_max),
        _cov_fig(cov, scale=cov_scale, linthresh=linthresh),
    )


def update_download_units(disp_sed_units, disp_wl_units):
    if not _cache:
        return gr.update()
    zip_path = _zip_tmp_from_cache(disp_sed_units, disp_wl_units)
    return gr.update(value=zip_path, visible=True)


# ── Simulation generator ───────────────────────────────────────────────────────
def run_simulation(
        star_inc, planet_inc, exozodi_inc, lzodi_inc,
        star_dist, star_mass, star_rad, star_temp, star_ra, star_dec,
        planet_mass, planet_rad, planet_temp, planet_sma,
        planet_ecc, planet_inc_deg, planet_raan, planet_aop, planet_ta,
        sed_mode, sed_file, sed_units_sel, sed_wl_sel, sed_units_custom, sed_wl_custom,
        exozodi_level,
        tot_int_val, tot_int_unit,
        noise_label, spec_res,
        use_det_int, det_int_val, det_int_unit,
        use_mod, mod_val, mod_unit,
        sol_ecl_lat, baseline_mode, custom_bl,
        ap_diam, throughput, qe, templ_fov, bl_min, bl_max, wl_min, wl_max,
        grid_size, use_seed, seed_val, device_str,
        ref_star_dist, ref_star_rad, ref_star_temp, ref_star_mass, ref_star_ra, ref_star_dec,
        disp_sed_units, disp_wl_units,
        ylim_min, ylim_max,
        cov_scale, linthresh,
        avg_runs_enabled, avg_runs_count,
):
    _UNIT = {"d": u.d, "h": u.hour, "s": u.s}
    _NA = (None, None, None, None)
    _hide_download = (gr.update(visible=False),)
    log = []
    _t0 = time.time()

    n_runs = max(2, int(avg_runs_count or 2)) if avg_runs_enabled else 1

    def _log(msg, level="INFO"):
        log.append(f"[{level}] {msg}")

    def _log_text():
        return "\n".join(log)

    def _status(msg, color="#f4a227", spin=True):
        elapsed = time.time() - _t0
        icon = '<span class="spin">↻</span>&nbsp; ' if spin else ""
        return (f'<div class="status-bar">'
                f'<span class="dot" style="background:{color}"></span>'
                f'<span style="color:{color};font-weight:600">{icon}{msg}</span>'
                f'<span style="color:#3a4460;margin-left:auto;font-size:0.75rem;">{elapsed:.0f}s</span>'
                f'</div>')

    def _mid(msg):
        return (*_NA, _status(msg), _log_text(),
                gr.update(visible=False), *_hide_download,
                gr.update(), gr.update(visible=True), gr.update(visible=True))

    def _run_threaded(msg, fn):
        _exc: list = [None]
        _res: list = [None]

        def _target():
            try:
                _res[0] = fn()
            except Exception as e:
                _exc[0] = e

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=1.0)
            if t.is_alive():
                yield _mid(msg)
        if _exc[0]:
            raise _exc[0]
        return _res[0]

    plt.close("all")
    yield _mid("Initialising…")

    try:
        from phringe.core.scene import Scene
        from phringe.core.sources.exozodi import Exozodi
        from phringe.core.sources.local_zodi import LocalZodi
        from phringe.core.sources.planet import Planet
        from phringe.core.sources.star import Star
        from phringe.lib.baseline import OptimalNullingBaseline
        from phringe.lib.beam_combiner import DoubleBracewell
        from lifesimmc.lib.instrument import InstrumentalNoise
        from lifesimmc.presets.single_epoch_observation.single_epoch_observation import SingleEpochObservation
        from lifesimmc.util.spectrum import convert_spectral_units, convert_wavelength_units

        _NOISE = {
            "None (ideal)": InstrumentalNoise.NONE,
            "Optimistic": InstrumentalNoise.OPTIMISTIC,
            "Pessimistic": InstrumentalNoise.PESSIMISTIC,
        }

        all_det_sigs: list = []
        all_sed_raws: list = []
        all_std_raws: list = []
        all_cov_raws: list = []
        all_filt_maps: list = []
        wl_raw_ref = None
        sed_true_raw_ref = None
        seo = None

        for run_idx in range(n_runs):
            _pfx = f"[{run_idx + 1}/{n_runs}] " if n_runs > 1 else ""

            _log(f"{_pfx}Building scene…")
            scene = Scene()

            if star_inc:
                scene.add_source(Star(
                    name="",
                    distance=f"{star_dist} pc",
                    mass=f"{star_mass} Msun",
                    radius=f"{star_rad} Rsun",
                    temperature=f"{star_temp} K",
                    right_ascension=f"{star_ra} hourangle",
                    declination=f"{star_dec} deg",
                ))
                _log(f"{_pfx}Star added.")

            if lzodi_inc:
                scene.add_source(LocalZodi())
                _log(f"{_pfx}Local zodi added.")

            if exozodi_inc:
                scene.add_source(Exozodi(
                    level=exozodi_level
                ))
                _log(f"{_pfx}Exozodi (level {int(exozodi_level)}) added.")

            if planet_inc:
                sed_loader_obj = None
                if sed_mode == "Custom" and sed_file is not None:
                    from phringe.io.sed_loader import SEDLoader
                    su = sed_units_custom if sed_units_sel == "Custom…" else sed_units_sel
                    wu = sed_wl_custom if sed_wl_sel == "Custom…" else sed_wl_sel
                    _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
                    with open(sed_file, "rb") as f:
                        _tmp.write(f.read())
                    _tmp.flush()
                    sed_loader_obj = SEDLoader(path_to_file=_tmp.name, sed_units=su, wavelength_units=wu)
                    _log(f"{_pfx}SED file loaded ({su}, {wu}).")
                scene.add_source(Planet(
                    name="",
                    propagate_orbit=False,
                    sed_loader=sed_loader_obj,
                    mass=f"{planet_mass} Mearth",
                    radius=f"{planet_rad} Rearth",
                    temperature=f"{planet_temp} K",
                    semi_major_axis=f"{planet_sma} au",
                    eccentricity=planet_ecc,
                    inclination=f"{planet_inc_deg} deg",
                    raan=f"{planet_raan} deg",
                    argument_of_periapsis=f"{planet_aop} deg",
                    true_anomaly=f"{planet_ta} deg"
                ))
                _log(f"{_pfx}Planet added.")

            yield _mid(f"{_pfx}Configuring preset…")

            nulling_baseline = (
                custom_bl * u.m if baseline_mode == "Custom (m)"
                else OptimalNullingBaseline(
                    angular_star_separation="habitable-zone",
                    wavelength="15 um",
                    sep_at_max_mod_eff=DoubleBracewell.sep_at_max_mod_eff[0],
                )
            )

            seed_this_run = int(seed_val) + run_idx if use_seed else None

            seo = SingleEpochObservation(
                scene=scene,
                total_integration_time=tot_int_val * _UNIT[tot_int_unit],
                detector_integration_time=det_int_val * _UNIT[det_int_unit] if use_det_int else None,
                modulation_period=mod_val * _UNIT[mod_unit] if use_mod else None,
                solar_ecliptic_latitude=f"{sol_ecl_lat} deg",
                nulling_baseline=nulling_baseline,
                aperture_diameter=ap_diam * u.m,
                nulling_baseline_min=bl_min * u.m,
                nulling_baseline_max=bl_max * u.m,
                spectral_resolving_power=int(spec_res),
                wavelength_min=wl_min * u.um,
                wavelength_max=wl_max * u.um,
                throughput=throughput,
                quantum_efficiency=qe,
                instrumental_noise=_NOISE[noise_label],
                template_fov_rad=templ_fov,
                grid_size=int(grid_size),
                seed=seed_this_run,
                device=torch.device(device_str.lower()),
                **({} if star_inc else {
                    "host_star_radius": f"{ref_star_rad} Rsun",
                    "host_star_temperature": f"{ref_star_temp} K",
                    "host_star_mass": f"{ref_star_mass} Msun",
                    "host_star_distance": f"{ref_star_dist} pc",
                    "host_star_declination": f"{ref_star_dec} deg",
                    "host_star_right_ascension": f"{ref_star_ra} deg",
                })
            )
            _log(f"{_pfx}Preset v{seo.version} created.")

            yield _mid(f"{_pfx}Running data pipeline…")
            yield from _run_threaded(f"{_pfx}Running data pipeline…", seo.run)
            _log(f"{_pfx}Pipeline complete.")

            yield _mid(f"{_pfx}Extracting SED…")
            sed_raw, std_raw, cov_raw = yield from _run_threaded(
                f"{_pfx}Extracting SED…", lambda: seo.extract_sed(units="ph/s/m3"))
            _log(f"{_pfx}SED extracted.")

            # num_reps in SEO is 1, so always use 0th element here
            sed_raw = sed_raw[0]
            std_raw = std_raw[0]
            cov_raw = cov_raw[0]

            yield _mid(f"{_pfx}Computing matched filter…")
            filt_map = yield from _run_threaded(
                f"{_pfx}Computing matched filter…", seo.get_matched_filter)
            _log(f"{_pfx}Matched filter computed.")

            # num_reps in SEO is 1, so always use 0th element here
            filt_map = filt_map[0]

            yield _mid(f"{_pfx}Neyman-Pearson test…")
            det_sig, wl_raw, sed_true_raw = yield from _run_threaded(
                f"{_pfx}Neyman-Pearson test…", lambda: (
                    seo.get_detection_significance()[0],  # num_reps in SEO is 1, so always use 0th element here
                    seo.get_wavelength_bin_centers(units="m"),
                    seo.get_input_sed(units="ph/s/m3"),
                ))
            _log(f"{_pfx}Detection significance: {det_sig:.2f} σ")

            all_det_sigs.append(float(det_sig))
            all_sed_raws.append(np.asarray(sed_raw))
            all_std_raws.append(np.asarray(std_raw))
            all_cov_raws.append(np.asarray(cov_raw))
            all_filt_maps.append(np.asarray(filt_map))
            if run_idx == 0:
                wl_raw_ref = np.asarray(wl_raw)
                sed_true_raw_ref = np.asarray(sed_true_raw)
            if n_runs > 1:
                _log(f"Run {run_idx + 1}/{n_runs} complete.")

        # ── Aggregate ─────────────────────────────────────────────────────────
        det_sig_final = float(np.mean(all_det_sigs))
        sed_raw_final = np.mean(all_sed_raws, axis=0)
        std_raw_final = np.mean(all_std_raws, axis=0)
        cov_arr_final = np.mean(all_cov_raws, axis=0)
        filt_map_final = np.mean(all_filt_maps, axis=0)

        _cache.update({
            "sed_raw": sed_raw_final,
            "std_raw": std_raw_final,
            "cov_raw": cov_arr_final,
            "wl_raw": wl_raw_ref,
            "sed_true_raw": sed_true_raw_ref,
            "filt_map": filt_map_final,
            "det_sig": det_sig_final,
            "n_runs": n_runs,
            "all_sed_raws": np.stack(all_sed_raws) if n_runs > 1 else None,
            "all_std_raws": np.stack(all_std_raws) if n_runs > 1 else None,
            "all_cov_raws": np.stack(all_cov_raws) if n_runs > 1 else None,
            "all_filt_maps": np.stack(all_filt_maps) if n_runs > 1 else None,
        })

        yield _mid("Rendering plots…")

        wl = convert_wavelength_units(wl_raw_ref, units_in="m", units_out=disp_wl_units)
        sed = convert_spectral_units(sed_raw_final, wl_raw_ref, units_in="ph/s/m3",
                                     units_out=disp_sed_units, wavelength_units="m")
        std = convert_spectral_units(std_raw_final, wl_raw_ref, units_in="ph/s/m3",
                                     units_out=disp_sed_units, wavelength_units="m")
        sed_true = convert_spectral_units(sed_true_raw_ref, wl_raw_ref, units_in="ph/s/m3",
                                          units_out=disp_sed_units, wavelength_units="m")

        fig_sed = _sed_fig(wl, sed, std, sed_true, disp_wl_units, disp_sed_units,
                           ylim_min=ylim_min, ylim_max=ylim_max)
        f_cov = np.asarray(convert_spectral_units(
            np.ones(len(wl_raw_ref)), wl_raw_ref, units_in="ph/s/m3",
            units_out=disp_sed_units, wavelength_units="m"))
        cov_display = np.outer(f_cov, f_cov) * cov_arr_final
        vmax_cov = max(abs(float(np.nanmin(cov_display))), abs(float(np.nanmax(cov_display)))) or 1.0
        linthresh_default = vmax_cov * 1e-3
        fig_cov = _cov_fig(cov_display, scale=cov_scale, linthresh=linthresh_default)
        fig_map = _map_fig(np.asarray(filt_map_final))
        _log("Plots rendered. Done.")

        snr = np.sqrt(np.sum((np.array(sed_true) / (np.array(std) + 1e-30)) ** 2))
        avg_label = f" · avg of {n_runs} runs" if n_runs > 1 else ""
        sig_html = (f'<div class="metric-card">'
                    f'<div class="metric-val">{det_sig_final:.2f} σ</div>'
                    f'<div class="metric-label">Detection Significance</div>'
                    f'<div class="metric-sub">SNR ≈ {snr:.1f} · preset v{seo.version}'
                    f' · {len(wl)} channels{avg_label}</div>'
                    f'</div>')
        done_status = ('<div class="status-bar">'
                       '<span class="dot" style="background:#3ddc84"></span>'
                       '<span style="color:#3ddc84;font-weight:600">Simulation complete</span>'
                       '</div>')

        yield (
            fig_sed, fig_cov, fig_map,
            sig_html, done_status, _log_text(),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=f"{linthresh_default:.1e}"),
            gr.update(visible=False),
            gr.update(visible=False),
        )

        zip_path = _zip_tmp_from_cache(disp_sed_units, disp_wl_units)

        yield (
            fig_sed, fig_cov, fig_map,
            sig_html, done_status, _log_text(),
            gr.update(visible=True),
            gr.update(value=zip_path, visible=True),
            gr.update(value=f"{linthresh_default:.1e}"),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    except Exception:
        _log(traceback.format_exc(), "ERROR")
        yield (*_NA,
               _status("Failed — check Log tab", "#f44336", spin=False), _log_text(),
               gr.update(visible=False), *_hide_download,
               gr.update(), gr.update(visible=False), gr.update(visible=True))


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
_READY = ('<div class="status-bar">'
          '<span class="dot" style="background:#333"></span>'
          '<span style="color:#444">Ready</span></div>')

_logo_path = next((p for p in [
    "docs/_static/lifesimmc_light.png",
] if Path(p).exists()), None)

with gr.Blocks(title="LIFEsimMC") as demo:
    # ── Header ────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=160):
            if _logo_path:
                def _img_to_base64(path):
                    with open(path, "rb") as fh:
                        return base64.b64encode(fh.read()).decode()


                img_b64 = _img_to_base64(_logo_path)
                gr.HTML(
                    f'<img src="data:image/png;base64,{img_b64}" '
                    'style="height:56px; width:auto; object-fit:contain;">'
                )
        with gr.Column(scale=5):
            gr.HTML('<h2 style="margin:0;font-size:1.45rem;font-weight:700;color:#e8e8e8;">'
                    'Single-Epoch Observation Simulator</h2>'
                    '<p style="margin:2px 0 0;font-size:0.8rem;color:#444;">'
                    'Preset Version 1</p>')

    gr.HTML('<hr style="border:none;border-top:1px solid #181818;margin:0.5rem 0 0;">')

    # ── Run bar ───────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        run_btn = gr.Button("▶  Run Simulation", variant="primary",
                            scale=0, size="lg", min_width=200)
        stop_btn = gr.Button("⏹  Stop", variant="secondary",
                             scale=0, size="lg", min_width=100, visible=False)
        status_html = gr.HTML(_READY, scale=3)
    with gr.Row(equal_height=True):
        avg_runs_enabled = gr.Checkbox(value=False, label="Average Over Runs",
                                       elem_classes="avg-runs", container=False,
                                       min_width=0, elem_id="p-avg-runs")
        avg_runs_count = gr.Number(value=10, minimum=2, show_label=False,
                                   container=False, visible=False, scale=0,
                                   min_width=70, elem_id="p-avg-runs-count")
        gr.HTML("<div style='flex:1'></div>")

    gr.HTML('<hr style="border:none;border-top:1px solid #181818;margin:0.4rem 0 0.6rem;">')

    # ── Main panels ───────────────────────────────────────────────────────────
    with gr.Row(elem_classes="main-panels", equal_height=False):

        # ════════════════════════════════════════════════════════════════════
        # LEFT — Configuration
        # ════════════════════════════════════════════════════════════════════
        with gr.Column(scale=6, min_width=420, elem_classes="panel-config"):
            with gr.Row(elem_classes="config-cols", equal_height=False):

                # ── Scene ────────────────────────────────────────────────
                with gr.Column(scale=1, min_width=260, elem_classes="config-scene"):
                    gr.HTML('<div class="sec-label">Scene</div>')

                    with gr.Group(elem_classes="source-card"):
                        with gr.Row():
                            gr.HTML("<strong>⭐ Star</strong>")
                            star_inc = gr.Checkbox(value=True, label="Include",
                                                   container=False, scale=0, min_width=80)
                        with gr.Group(visible=True) as star_params:
                            with gr.Row():
                                star_dist = gr.Number(value=10.0, label="Distance (pc)",
                                                      minimum=0.001, elem_id="p-star-dist")
                                star_temp = gr.Number(value=5778, label="Temperature (K)",
                                                      minimum=100, elem_id="p-star-temp")
                            with gr.Row():
                                star_mass = gr.Number(value=1.0, label="Mass (M☉)",
                                                      minimum=0.001, elem_id="p-star-mass")
                                star_rad = gr.Number(value=1.0, label="Radius (R☉)",
                                                     minimum=0.001, elem_id="p-star-rad")
                            with gr.Accordion("More", open=False):
                                with gr.Row():
                                    star_ra = gr.Number(value=10.0, label="RA (h)",
                                                        elem_id="p-star-ra")
                                    star_dec = gr.Number(value=45.0, label="Dec (°)",
                                                         elem_id="p-star-dec")

                    with gr.Group(visible=False, elem_classes="ref-card") as host_star_ref:
                        gr.HTML('<div style="font-size:0.7rem;font-weight:700;color:#5b8af5;'
                                'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;">'
                                '⭐ Host Star Reference</div>'
                                '<div style="font-size:0.72rem;color:#3a4460;margin-bottom:0.5rem;">'
                                'Needed for scene geometry when star is excluded.</div>')
                        with gr.Row():
                            ref_star_rad = gr.Number(value=1.0, label="Radius (R☉)",
                                                     minimum=0.01, elem_id="p-ref-rad")
                            ref_star_mass = gr.Number(value=1.0, label="Mass (M☉)",
                                                      minimum=0.01, elem_id="p-ref-mass")
                            ref_star_temp = gr.Number(value=5780.0, label="Temperature (K)",
                                                      minimum=2000, elem_id="p-ref-temp")
                        with gr.Row():
                            ref_star_dist = gr.Number(value=10.0, label="Distance (pc)",
                                                      minimum=0.1, elem_id="p-ref-dist")
                            ref_star_ra = gr.Number(value=10.0, label="RA (h)",
                                                    elem_id="p-ref-ra")
                            ref_star_dec = gr.Number(value=45.0, label="Dec (°)",
                                                     elem_id="p-ref-dec")

                    with gr.Group(elem_classes="source-card"):
                        with gr.Row():
                            gr.HTML("<strong>🪐 Planet</strong>")
                            planet_inc = gr.Checkbox(value=True, label="Include",
                                                     container=False, scale=0, min_width=80)
                        with gr.Group(visible=True) as planet_params:
                            with gr.Row():
                                planet_sma = gr.Number(value=1.0, label="Semi-Major Axis (au)",
                                                       minimum=0.001, elem_id="p-planet-sma")
                                planet_temp = gr.Number(value=254, label="Temperature (K)",
                                                        minimum=10, elem_id="p-planet-temp")
                            with gr.Row():
                                planet_mass = gr.Number(value=1.0, label="Mass (M⊕)",
                                                        minimum=0.001, elem_id="p-planet-mass")
                                planet_rad = gr.Number(value=1.0, label="Radius (R⊕)",
                                                       minimum=0.001, elem_id="p-planet-rad")
                            sed_mode = gr.Radio(["Blackbody", "Custom"], value="Blackbody",
                                                label="SED", elem_id="p-sed-mode")
                            with gr.Group(visible=False) as sed_custom_group:
                                sed_file = gr.File(label="SED file (.txt / .dat / .csv)",
                                                   file_types=[".txt", ".dat", ".csv"])
                                with gr.Row():
                                    _SED_P = ["W/sr/m2/um", "W/m2/um", "ph/s/m2/um", "ph/s/m3",
                                              "erg/s/cm2/AA", "erg/s/cm2/Hz", "Custom…"]
                                    _WL_P = ["um", "nm", "m", "Custom…"]
                                    sed_units_sel = gr.Dropdown(_SED_P, value="W/sr/m2/um",
                                                                label="SED Units")
                                    sed_wl_sel = gr.Dropdown(_WL_P, value="um",
                                                             label="Wavelength Units")
                                sed_units_custom = gr.Textbox(label="Custom SED units",
                                                              placeholder="e.g. erg/s/cm2/AA",
                                                              visible=False)
                                sed_wl_custom = gr.Textbox(label="Custom Wavelength Units",
                                                           placeholder="e.g. AA", visible=False)
                            with gr.Accordion("More", open=False):
                                with gr.Row():
                                    planet_ecc = gr.Number(value=0.0, label="Eccentricity",
                                                           minimum=0.0, maximum=0.9999,
                                                           elem_id="p-planet-ecc")
                                    planet_inc_deg = gr.Number(value=180.0, label="Inclination (°)",
                                                               elem_id="p-planet-incl")
                                with gr.Row():
                                    planet_raan = gr.Number(value=90.0, label="RAAN (°)",
                                                            elem_id="p-planet-raan")
                                    planet_aop = gr.Number(value=0.0, label="Arg. Periapsis (°)",
                                                           elem_id="p-planet-aop")
                                with gr.Row():
                                    planet_ta = gr.Number(value=45.0, label="True Anomaly (°)",
                                                          elem_id="p-planet-ta")

                    with gr.Group(elem_classes="source-card"):
                        with gr.Row():
                            gr.HTML("<strong>☁ Exozodi</strong>")
                            exozodi_inc = gr.Checkbox(value=True, label="Include",
                                                      container=False, scale=0, min_width=80)
                        with gr.Group(visible=True) as exozodi_params:
                            exozodi_level = gr.Number(value=3, label="Exozodi Level",
                                                      minimum=0.1, elem_id="p-exozodi-level")

                    with gr.Group(elem_classes="source-card"):
                        with gr.Row():
                            gr.HTML("<strong>🌌 Local Zodi</strong>")
                            lzodi_inc = gr.Checkbox(value=True, label="Include",
                                                    container=False, scale=0, min_width=80)
                        gr.HTML('<span style="color:#444;font-size:0.78rem;">No configurable parameters.</span>')

                # ── Observation / Instrument / Simulation ─────────────────
                with gr.Column(scale=1, min_width=240, elem_classes="config-obs"):
                    gr.HTML('<div class="sec-label">Observation</div>')

                    with gr.Group(elem_classes="source-card"):
                        with gr.Row():
                            tot_int_val = gr.Number(value=1.0, minimum=0.001,
                                                    label="Integration Time",
                                                    elem_id="p-tot-int")
                            tot_int_unit = gr.Dropdown(["d", "h", "s"], value="d", label="Unit")
                        spec_res = gr.Number(value=20, label="Spectral Resolving Power",
                                             minimum=1, maximum=10000, elem_id="p-spec-res")
                        with gr.Accordion("More", open=False):
                            use_det_int = gr.Checkbox(value=False,
                                                      label="Override Detector Integration Time")
                            with gr.Row(visible=False) as det_int_row:
                                det_int_val = gr.Number(value=432.0,
                                                        label="Detector Integration Time",
                                                        minimum=0.001, elem_id="p-det-int")
                                det_int_unit = gr.Dropdown(["d", "h", "s"], value="s", label="Unit")
                            use_mod = gr.Checkbox(value=False, label="Override Modulation Period")
                            with gr.Row(visible=False) as mod_row:
                                mod_val = gr.Number(value=1.0, label="Modulation Period",
                                                    minimum=0.001, elem_id="p-mod-val")
                                mod_unit = gr.Dropdown(["d", "h", "s"], value="d", label="Unit")
                            sol_ecl_lat = gr.Number(value=0.0,
                                                    label="Solar Ecliptic Latitude (°)",
                                                    elem_id="p-sol-ecl-lat")
                            baseline_mode = gr.Radio(["Optimize for HZ", "Custom (m)"],
                                                     value="Optimize for HZ",
                                                     label="Nulling Baseline Length")
                            custom_bl = gr.Number(value=10.0,
                                                  label="Custom Nulling Baseline Length (m)",
                                                  visible=False, minimum=1.0,
                                                  elem_id="p-custom-bl")

                    gr.HTML('<div class="sec-label">Instrument</div>')

                    with gr.Group(elem_classes="source-card"):
                        noise_label = gr.Dropdown(
                            ["None (ideal)", "Optimistic", "Pessimistic"],
                            value="Optimistic", label="Instrumental Noise",
                            elem_id="p-noise")
                        with gr.Accordion("More", open=False):
                            with gr.Row():
                                ap_diam = gr.Number(value=3.5, label="Aperture Diameter (m)",
                                                    minimum=0.1, elem_id="p-ap-diam")
                                throughput = gr.Number(value=0.15, label="Throughput",
                                                       minimum=0.0, maximum=1.0,
                                                       elem_id="p-throughput")
                            with gr.Row():
                                qe = gr.Number(value=0.6, label="Quantum Efficiency",
                                               minimum=0.0, maximum=1.0, elem_id="p-qe")
                            with gr.Row():
                                bl_min = gr.Number(value=10.0, label="Min. Null. Baseline (m)",
                                                   minimum=0.1, elem_id="p-bl-min")
                                bl_max = gr.Number(value=100.0, label="Max. Null. Baseline (m)",
                                                   minimum=1.0, elem_id="p-bl-max")
                            with gr.Row():
                                wl_min = gr.Number(value=4.0, label="Min. Wavelength (µm)",
                                                   minimum=0.1, elem_id="p-wl-min")
                                wl_max = gr.Number(value=18.5, label="Max. Wavelength (µm)",
                                                   minimum=0.1, elem_id="p-wl-max")

                    gr.HTML('<div class="sec-label">Simulation</div>')

                    with gr.Group(elem_classes="source-card"):
                        def _available_devices():
                            devs = ["cpu"]
                            if torch.cuda.is_available():
                                devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                                devs.append("mps")
                            if hasattr(torch, "xpu") and torch.xpu.is_available():
                                devs += [f"xpu:{i}" for i in range(torch.xpu.device_count())]
                            return devs


                        _devs = _available_devices()
                        device_str = gr.Radio([s.upper() for s in _devs], value="CPU",
                                              label="Compute Device", elem_id="p-device")
                        with gr.Accordion("More", open=False):
                            grid_size = gr.Number(value=40, label="Grid Size",
                                                  minimum=4, maximum=512, elem_id="p-grid-size")
                            templ_fov = gr.Number(value=1e-6, label="Template FOV (rad)",
                                                  elem_id="p-templ-fov")
                            with gr.Row():
                                use_seed = gr.Checkbox(value=False, label="Fixed Seed")
                                seed_val = gr.Number(value=42, label="Seed", minimum=0,
                                                     elem_id="p-seed")

        # ════════════════════════════════════════════════════════════════════
        # RIGHT — Results
        # ════════════════════════════════════════════════════════════════════
        with gr.Column(scale=4, min_width=320, elem_classes="panel-results"):
            gr.HTML('<div class="sec-label">Results</div>')

            with gr.Group(elem_classes="source-card"):
                with gr.Row():
                    disp_sed_units = gr.Dropdown(
                        ["ph/s/m2/um", "ph/s/m3", "W/m2/um", "W/m3", "erg/s/cm2/AA", "erg/s/cm2/Hz"],
                        value="ph/s/m2/um", label="SED Units", scale=1)
                    disp_wl_units = gr.Dropdown(
                        ["um", "nm", "m"], value="um", label="Wavelength Units", scale=1)

                sig_html = gr.HTML()

                with gr.Column(visible=False) as results_col:
                    gr.HTML('<div class="sec-label">Spectral Energy Distribution</div>')
                    plot_sed = gr.Plot(label="", container=False)
                    with gr.Row():
                        ylim_min = gr.Number(value=None, label="Y min", precision=4,
                                             scale=1, min_width=90)
                        ylim_max = gr.Number(value=None, label="Y max", precision=4,
                                             scale=1, min_width=90)

                    gr.HTML('<hr style="border:none;border-top:1px solid #1e2538;margin:0.6rem 0;">')
                    gr.HTML('<div class="sec-label">Covariance Matrix</div>')
                    with gr.Row():
                        cov_scale = gr.Radio(["Linear", "SymLog"], value="Linear",
                                             label="Scale", scale=2)
                        linthresh = gr.Textbox(value="", label="Linear Threshold",
                                               placeholder="e.g. 1.2e-4",
                                               scale=2, visible=False)
                    plot_cov = gr.Plot(label="", container=False)

                    gr.HTML('<hr style="border:none;border-top:1px solid #1e2538;margin:0.6rem 0;">')
                    gr.HTML('<div class="sec-label">Matched Filter Map</div>')
                    plot_map = gr.Plot(label="", container=False)

                    gr.HTML('<hr style="border:none;border-top:1px solid #1e2538;margin:0.6rem 0;">')
                    gr.HTML('<div class="sec-label">Downloads</div>')
                    download_status = gr.HTML("")
                    prepare_downloads_btn = gr.DownloadButton(
                        "Download Results (.zip)",
                        visible=False,
                        variant="secondary",
                    )

                no_results_html = gr.HTML(
                    '<div id="no-results" style="text-align:center;color:#fff;padding:4rem 0;">'
                    'No results yet — run a simulation first.</div>'
                )

    # ── Bottom tabs ───────────────────────────────────────────────────────────
    with gr.Tabs(elem_classes="bottom-tabs"):
        with gr.TabItem("📋  Log"):
            log_box = gr.Textbox(label="Simulation Log", lines=24,
                                 interactive=False, elem_classes="log-box")
            gr.Button("Clear", variant="secondary", size="sm").click(
                fn=lambda: "", outputs=log_box)

        with gr.TabItem("❓  Help"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
## Getting Started
1. Toggle **Include** for each scene source.
2. Set scene parameters (star, planet, exozodi, local zodi).
3. Set **Integration Time** and **Spectral Resolving Power**.
4. Expand *More parameters* sections for advanced control.
5. Click **▶ Run Simulation**.
6. Results appear on the right as the simulation progresses.

---

## Pipeline Steps
| # | Module |
|---|--------|
| 1 | Setup (instrument + observation + scene) |
| 2 | Data generation |
| 3 | Template generation |
| 4 | ZCA whitening |
| 5 | ML SED estimation |
| 6 | Matched filter correlation |
| 7 | Neyman-Pearson hypothesis test |

---

## Custom Planetary SED
Upload a plain-text file with wavelength and flux density columns.
Select units from the preset list, or enter a custom [astropy unit string](https://docs.astropy.org/en/stable/units/).
""")
                with gr.Column():
                    gr.Markdown("""
## Contact & Support

| | |
|--|--|
| **GitHub** | [github.com/pahuber/lifesimmc](https://github.com/pahuber/lifesimmc) |
| **Docs** | [lifesimmc.readthedocs.io](https://lifesimmc.readthedocs.io) |
| **Issues** | Open a ticket on GitHub |

---

## Troubleshooting
- **Errors**: check the **📋 Log** tab for the full traceback.
- **CUDA**: verify your device string (`cuda:0`, `cuda:1`) and driver.
- **Slow runs**: reduce *Grid Size* (e.g. 20) or shorten integration time.
- **Unit errors**: custom strings must be valid astropy expressions.

---
GPL-3.0 License · © Philipp A. Huber
""")

    # ── Event wiring ──────────────────────────────────────────────────────────
    star_inc.change(
        lambda x: [gr.update(visible=x), gr.update(visible=not x)],
        star_inc, [star_params, host_star_ref], queue=False)
    planet_inc.change(
        lambda x: gr.update(visible=x), planet_inc, planet_params, queue=False)
    exozodi_inc.change(
        lambda x: gr.update(visible=x), exozodi_inc, exozodi_params, queue=False)

    sed_mode.change(
        lambda x: gr.update(visible=x == "Custom"),
        sed_mode, sed_custom_group, queue=False)
    sed_units_sel.change(
        lambda x: gr.update(visible=x == "Custom…"), sed_units_sel, sed_units_custom, queue=False)
    sed_wl_sel.change(
        lambda x: gr.update(visible=x == "Custom…"), sed_wl_sel, sed_wl_custom, queue=False)

    use_det_int.change(
        lambda x: gr.update(visible=x), use_det_int, det_int_row, queue=False)
    use_mod.change(
        lambda x: gr.update(visible=x), use_mod, mod_row, queue=False)
    baseline_mode.change(
        lambda x: gr.update(visible=x == "Custom (m)"), baseline_mode, custom_bl, queue=False)

    cov_scale.change(
        lambda x: gr.update(visible=x == "SymLog"), cov_scale, linthresh, queue=False)

    avg_runs_enabled.change(
        lambda x: gr.update(visible=x), avg_runs_enabled, avg_runs_count,
        queue=False, show_progress="hidden")

    # ── Run / Stop ────────────────────────────────────────────────────────────
    _all_inputs = [
        star_inc, planet_inc, exozodi_inc, lzodi_inc,
        star_dist, star_mass, star_rad, star_temp, star_ra, star_dec,
        planet_mass, planet_rad, planet_temp, planet_sma,
        planet_ecc, planet_inc_deg, planet_raan, planet_aop, planet_ta,
        sed_mode, sed_file, sed_units_sel, sed_wl_sel, sed_units_custom, sed_wl_custom,
        exozodi_level,
        tot_int_val, tot_int_unit,
        noise_label, spec_res,
        use_det_int, det_int_val, det_int_unit,
        use_mod, mod_val, mod_unit,
        sol_ecl_lat, baseline_mode, custom_bl,
        ap_diam, throughput, qe, templ_fov, bl_min, bl_max, wl_min, wl_max,
        grid_size, use_seed, seed_val, device_str,
        ref_star_dist, ref_star_rad, ref_star_temp, ref_star_mass, ref_star_ra, ref_star_dec,
        disp_sed_units, disp_wl_units,
        ylim_min, ylim_max,
        cov_scale, linthresh,
        avg_runs_enabled, avg_runs_count,
    ]

    # 11 outputs — every yield in run_simulation must return exactly 11 values
    _all_outputs = [
        plot_sed, plot_cov, plot_map,
        sig_html, status_html, log_box,
        results_col,
        prepare_downloads_btn,
        linthresh,
        stop_btn,
        no_results_html,
    ]

    run_event = run_btn.click(
        fn=run_simulation, inputs=_all_inputs, outputs=_all_outputs)

    stop_btn.click(
        fn=lambda: (_READY, gr.update(visible=False)),
        outputs=[status_html, stop_btn],
        cancels=[run_event])

    # ── Replot triggers ───────────────────────────────────────────────────────
    _rp_in = [disp_sed_units, disp_wl_units, ylim_min, ylim_max, cov_scale, linthresh]
    for _w in [disp_sed_units, disp_wl_units]:
        _w.change(replot, _rp_in, [plot_sed, plot_cov, linthresh], queue=False)
    for _w in [ylim_min, ylim_max, cov_scale]:
        _w.change(replot_nolt, _rp_in, [plot_sed, plot_cov], queue=False)
    linthresh.blur(replot_nolt, _rp_in, [plot_sed, plot_cov], queue=False)
    linthresh.submit(replot_nolt, _rp_in, [plot_sed, plot_cov], queue=False)

    _dl_inputs = [disp_sed_units, disp_wl_units]
    for _w in [disp_sed_units, disp_wl_units]:
        _w.change(update_download_units, _dl_inputs, prepare_downloads_btn, queue=True)

    # ── Info-button injection (tooltips from docstrings) ──────────────────────
    gr.HTML(f"""<script>
(function () {{
  "use strict";
  const TIPS = {json.dumps(_TIPS, ensure_ascii=False)};

  function findLabelSpan(el) {{
    // Radio / fieldset group label
    var s = el.querySelector("legend span") || el.querySelector("legend");
    if (s) return s;
    // Label without an embedded input (Number, Dropdown, Textbox, Radio group)
    var labels = el.querySelectorAll("label");
    for (var i = 0; i < labels.length; i++) {{
      if (!labels[i].querySelector("input")) {{
        var sp = labels[i].querySelector("span");
        if (sp) return sp;
      }}
    }}
    // Checkbox / inline radio: label with input + text span
    var chk = el.querySelector("label > span");
    if (chk) return chk;
    // Gradio Radio standalone label-wrap span
    var lw = el.querySelector(".label-wrap span") || el.querySelector(".label-wrap");
    if (lw) return lw;
    return null;
  }}

  function injectAll() {{
    for (var id in TIPS) {{
      var el = document.getElementById(id);
      if (!el || el.dataset.tipDone) continue;
      var span = findLabelSpan(el);
      if (!span) continue;

      (function (tipText, anchorEl) {{
        var btn = document.createElement("i");
        btn.className = "info-btn";
        btn.textContent = "i";

        var tipEl = document.createElement("span");
        tipEl.innerHTML = tipText.replace(/\n/g, "<br>");
        tipEl.style.cssText = [
          "display:none", "position:fixed", "z-index:99999",
          "background:#111828", "border:1px solid #2d4070", "border-radius:8px",
          "padding:8px 11px", "font-size:0.7rem", "font-weight:400", "font-style:normal",
          "color:#8898b8", "white-space:normal", "width:220px", "line-height:1.5",
          "box-shadow:0 4px 24px rgba(0,0,0,0.75)", "pointer-events:none",
          "letter-spacing:0", "text-transform:none"
        ].join(";");
        document.body.appendChild(tipEl);

        btn.addEventListener("mouseenter", function () {{
          var r = btn.getBoundingClientRect();
          tipEl.style.display = "block";
          tipEl.style.left = (r.right + 8) + "px";
          tipEl.style.right = "auto";
          tipEl.style.top = (r.top + r.height / 2) + "px";
          tipEl.style.transform = "translateY(-50%)";
          requestAnimationFrame(function () {{
            var tr = tipEl.getBoundingClientRect();
            if (tr.right > window.innerWidth - 8) {{
              tipEl.style.left = "auto";
              tipEl.style.right = (window.innerWidth - r.left + 8) + "px";
            }}
          }});
        }});
        btn.addEventListener("mouseleave", function () {{
          tipEl.style.display = "none";
        }});

        span.after(btn);
        anchorEl.dataset.tipDone = "1";
      }})(TIPS[id], el);
    }}
  }}

  var attempts = 0;
  (function retry() {{
    injectAll();
    if (++attempts < 20) setTimeout(retry, 500);
  }})();
}})();
</script>""")


def main():
    demo.launch(theme=theme, css=CSS)


if __name__ == "__main__":
    main()
