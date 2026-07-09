"""Lightcurve plotting -- analyze_and_plot_lightcurves,
plot_individual_lightcurve, generate_stats_text, save_lightcurve_data,
create_lightcurve_summary, create_summary_grid_plot. self.lightcurve_dir/
self.config/self.logger are explicit parameters rather than instance state.
"""

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


def analyze_and_plot_lightcurves(lightcurves: Dict, lightcurve_dir, config=None, logger=None, final_candidates=None):
    """Generate lightcurve plots and analysis.

    Only the top-N candidates by quality_score (from frontend max_candidates
    config) get PNG plots and ECSV data files — the frontend never accesses
    the rest, and generating all of them causes a pipeline timeout.
    """
    max_outputs = 500  # safe fallback
    if config is not None:
        max_outputs = config.frontend.max_candidates

    # Build ordered set of transient_ids to process, ranked by quality_score.
    # final_candidates is already sorted descending by quality_score.
    process_ids: set = set()
    if final_candidates is not None and len(final_candidates) > 0 and 'transient_id' in final_candidates.colnames:
        top = final_candidates[:max_outputs]
        process_ids = set(str(tid) for tid in top['transient_id'])

    n_saved = 0
    for transient_id, lightcurve in lightcurves.items():
        if process_ids and transient_id not in process_ids:
            continue
        save_lightcurve_data(transient_id, lightcurve, lightcurve_dir)
        plot_individual_lightcurve(transient_id, lightcurve, lightcurve_dir)
        n_saved += 1

    if logger:
        logger.info(f"Generated {n_saved} lightcurve outputs (out of {len(lightcurves)} total)")


def plot_individual_lightcurve(transient_id: str, lightcurve: Table, lightcurve_dir):
    """Create detailed lightcurve plot."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    times = lightcurve['obs_time']
    time_hours = (times - times[0]) / 3600.0

    if 'MAG_CALIB' in lightcurve.colnames:
        mags = lightcurve['MAG_CALIB']
        mag_errs = lightcurve['MAGERR_CALIB']
        ax.set_ylabel('Calibrated Magnitude')
    else:
        mags = lightcurve['MAG_ISO']
        mag_errs = lightcurve['MAGERR_ISO']
        ax.set_ylabel('Instrumental Magnitude')

    ax.errorbar(time_hours, mags, yerr=mag_errs, fmt='o-', capsize=3, markersize=6)
    ax.invert_yaxis()
    ax.set_xlabel('Time since first detection (hours)')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Lightcurve: {transient_id}')

    # Add statistics
    stats_text = generate_stats_text(lightcurve)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plot_filename = lightcurve_dir / f"{transient_id}_lightcurve.png"
    plt.savefig(str(plot_filename), dpi=150, bbox_inches='tight')
    plt.close()


def generate_stats_text(lightcurve: Table) -> str:
    """Generate statistics text for lightcurve plot."""
    stats = []
    stats.append(f"N detections: {len(lightcurve)}")
    stats.append(f"N epochs: {len(np.unique(lightcurve['epoch_id']))}")

    time_span = (np.max(lightcurve['obs_time']) - np.min(lightcurve['obs_time'])) / 3600.0
    stats.append(f"Time span: {time_span:.1f} hours")

    if 'MAG_CALIB' in lightcurve.colnames:
        mags = lightcurve['MAG_CALIB']
        stats.append(f"Mag range: {np.max(mags) - np.min(mags):.2f}")
        stats.append(f"Mag std: {np.std(mags):.3f}")

    ra_std = np.std(lightcurve['ALPHA_J2000']) * 3600
    dec_std = np.std(lightcurve['DELTA_J2000']) * 3600
    pos_scatter = np.sqrt(ra_std**2 + dec_std**2)
    stats.append(f"Pos scatter: {pos_scatter:.2f}\"")

    return '\n'.join(stats)


def save_lightcurve_data(transient_id: str, lightcurve: Table, lightcurve_dir):
    """Save lightcurve data to file."""
    lightcurve.meta['transient_id'] = transient_id
    lightcurve.meta['n_detections'] = len(lightcurve)
    lightcurve.meta['n_epochs'] = len(np.unique(lightcurve['epoch_id']))

    filename = lightcurve_dir / f"{transient_id}_lightcurve.ecsv"
    lightcurve.write(str(filename), format='ascii.ecsv', overwrite=True)


def create_lightcurve_summary(lightcurves: Dict, final_candidates: Table, lightcurve_dir):
    """Create summary plots and analysis."""

    # Summary grid plot
    create_summary_grid_plot(lightcurves, final_candidates, lightcurve_dir)


def create_summary_grid_plot(lightcurves: Dict, final_candidates, lightcurve_dir, max_plots=32):
    """
    Enhanced grid plot with color coding for different candidate types.
    """

    n_plots = min(len(lightcurves), max_plots)
    if n_plots == 0:
        return

    cols = int(np.ceil(np.sqrt(n_plots)))
    rows = int(np.ceil(n_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    axes_flat = axes.flatten() if n_plots > 1 else axes

    # Define colors for different candidate types
    type_colors = {
        'new': 'blue',
        'brightening': 'red',
        'fading': 'orange',
        'trail': 'purple',
        'unknown': 'gray'
    }

    type_symbols = {
        'new': 'o',
        'brightening': '^',  # Triangle up
        'fading': 'v',       # Triangle down
        'trail': 'D',        # Diamond
        'unknown': 's'       # Square
    }

    # Sort lightcurves by quality score (highest first)
    lightcurve_items = []
    for transient_id, lc in lightcurves.items():
        candidate_mask = final_candidates['transient_id'] == transient_id
        if np.any(candidate_mask):
            quality_score = final_candidates[candidate_mask][0]['quality_score']
            lightcurve_items.append((quality_score, transient_id, lc))
        else:
            lightcurve_items.append((0.0, transient_id, lc))

    # Sort by quality score (descending)
    lightcurve_items.sort(key=lambda x: x[0], reverse=True)

    for i, (quality_score, transient_id, lc) in enumerate(lightcurve_items[:max_plots]):
        ax = axes_flat[i]

        times = lc['obs_time']
        time_hours = (times - times[0]) / 3600.0

        if 'MAG_CALIB' in lc.colnames:
            mags = lc['MAG_CALIB']
            mag_errs = lc['MAGERR_CALIB']
        else:
            mags = lc['MAG_ISO']
            mag_errs = lc['MAGERR_ISO']

        # Get candidate type for color coding
        candidate_mask = final_candidates['transient_id'] == transient_id
        if np.any(candidate_mask):
            candidate = final_candidates[candidate_mask][0]
            candidate_type = candidate['candidate_type'] if 'candidate_type' in candidate.colnames else 'unknown'
        else:
            candidate_type = 'unknown'

        # Plot with appropriate color and symbol
        color = type_colors.get(candidate_type, 'gray')
        symbol = type_symbols.get(candidate_type, 'o')

        ax.errorbar(time_hours, mags, yerr=mag_errs,
                fmt=f'{symbol}-', color=color, markersize=4, capsize=2,
                label=candidate_type.capitalize())

        ax.set_title(f"{transient_id[:15]}...\nN={len(lc)} ({candidate_type})",
                    fontsize=8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Add legend
    if n_plots > 0:
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = []
        for ctype, color in type_colors.items():
            symbol = type_symbols[ctype]
            legend_elements.append(Line2D([0], [0], marker=symbol, color=color,
                                        linestyle='-', markersize=6, label=ctype.capitalize()))

        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    save_path = lightcurve_dir / 'lightcurves_summary.png'
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
