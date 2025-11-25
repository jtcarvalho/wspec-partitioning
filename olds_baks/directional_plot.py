import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable

__all__ = ["plot_directional_spectrum"]

def plot_directional_spectrum(
    E2d, freq, dirs_deg, hs, tp, dp, selected_time, source_type,
    save_dir=None, show_plot=True, spectrum_unit="m2_s_rad",
    vmin=None, vmax=None, levels=None, cmap="jet", dpi=300,
    title_override=None, max_density_2d=None, lon=None, lat=None,
    file_tag=None, index=None
):
    """Unified directional spectrum plot (polar, period radial) for WW3, SAR, NDBC.

    Parameters
    ----------
    E2d : 2D np.ndarray (nf, nd)
        Directional spectrum values in the chosen spectrum_unit.
    freq : 1D np.ndarray (Hz)
    dirs_deg : 1D array of directions in degrees (meteorological/oceanographic consistent with calling code).
    hs, tp, dp : float
        Significant wave height (m), peak period (s), peak direction (deg).
    selected_time : datetime-like
    source_type : str ("ww3", "sar", "ndbc")
    spectrum_unit : str
        One of: m2_s_rad, m2_Hz_rad, m2_s_deg, m2_Hz_deg (others allowed with fallback scaling).
    vmin, vmax, levels : optional color scale specification.
    cmap : str or Colormap
    dpi : int for saving figure.
    title_override : str optional custom title.
    """
    if E2d is None or freq is None or dirs_deg is None:
        raise ValueError("E2d, freq and dirs_deg must be provided")

    Eplot = np.nan_to_num(E2d, nan=0.0, neginf=0.0, posinf=0.0)

    # Ensure 1D arrays
    freq = np.asarray(freq).flatten()
    dirs_deg = np.asarray(dirs_deg).flatten()

    # Convert directions to radians & sort
    dirs_rad = np.radians(dirs_deg)
    sort_idx = np.argsort(dirs_rad)
    dirs_sorted = dirs_rad[sort_idx]
    Eplot_sorted = Eplot[:, sort_idx]

    # Guarantee periodic wrap (append 2pi)
    if not np.isclose(dirs_sorted[0], 0.0):
        dirs_sorted = np.insert(dirs_sorted, 0, 0.0)
        Eplot_sorted = np.insert(Eplot_sorted, 0, Eplot_sorted[:, 0], axis=1)
    if not np.isclose(dirs_sorted[-1], 2*np.pi):
        dirs_sorted = np.append(dirs_sorted, 2*np.pi)
        Eplot_sorted = np.concatenate([Eplot_sorted, Eplot_sorted[:, 0:1]], axis=1)

    # Radial = period (s)
    with np.errstate(divide='ignore', invalid='ignore'):
        period = np.where(freq > 0, 1.0 / freq, 0)

    theta, r = np.meshgrid(dirs_sorted, period)

    # Default color scales per unit (matches leWW3Spec conventions)
    # Escala comum ajustável (vmax configurável)
    if vmin is None:
        vmin = 2.5
    if vmax is None:
        vmax = max_density_2d if (max_density_2d is not None) else 25.0
    if levels is None:
        step = max((vmax - vmin)/50.0, 0.5)  # granularidade adaptativa
        levels = np.arange(vmin + step, vmax + step*0.51, step)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')

    cs = ax.contour(theta, r, Eplot_sorted, levels, cmap=cmap, vmin=vmin, vmax=vmax)

    # Axes style
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rticks([5, 10, 15, 20])
    ax.set_yticklabels(['5s', '10s', '15s', '20s'], color='gray', fontsize=7.5)
    ax.set_rlim(0, 25)
    ax.set_rlabel_position(30)
    ax.tick_params(axis='y', colors='gray', labelsize=16)
    ticks = ['N','NE','E','SE','S','SW','W','NW']
    tick_angles = np.deg2rad(np.linspace(0, 315, 8))
    ax.set_xticks(tick_angles)
    ax.set_xticklabels(ticks)
    ax.tick_params(axis='x', colors='k', labelsize=16)
    title = title_override or 'Directional Spectrum'
    ax.set_title(title, fontsize=16, color='k', pad=30)

    # Stats box
    stats_ax = fig.add_axes([0.75, 0.7, 0.2, 0.15], facecolor='white')
    stats_ax.patch.set_alpha(0.8)
    stats_ax.patch.set_edgecolor('black')
    stats_ax.patch.set_linewidth(1.5)
    stats_ax.axis('off')

    stats_ax.text(0.7, 1.9, 'Statistics', fontsize=14, color='k', ha='center', va='center', weight='bold')
    try:
        if isinstance(selected_time, np.datetime64):
            import pandas as pd
            selected_time = pd.to_datetime(selected_time)
        date_str = selected_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(selected_time, 'strftime') else str(selected_time)
    except Exception:
        date_str = str(selected_time)
    stats_ax.text(0.7, 1.7, f'Date: {date_str}', fontsize=12, color='k', ha='center', va='center')
    
    # Adicionar coordenadas se disponíveis
    if lon is not None and lat is not None:
        stats_ax.text(0.7, 1.55, f'Lon: {lon:.3f}°', fontsize=12, color='k', ha='center', va='center')
        stats_ax.text(0.7, 1.4, f'Lat: {lat:.3f}°', fontsize=12, color='k', ha='center', va='center')
        y_offset = 1.25
    else:
        y_offset = 1.55
    
    stats_ax.text(0.7, y_offset, f'Hs: {hs:.2f} m', fontsize=12, color='k', ha='center', va='center')
    if tp is not None and not (isinstance(tp, float) and np.isnan(tp)):
        stats_ax.text(0.7, y_offset-0.15, f'Tp: {tp:.1f} s', fontsize=12, color='k', ha='center', va='center')
    if dp is not None and not (isinstance(dp, float) and np.isnan(dp)):
        stats_ax.text(0.7, y_offset-0.3, f'Dp: {dp:.1f}°', fontsize=12, color='k', ha='center', va='center')

    unit_labels = {
        'm2_s_rad': 'm²·s·rad⁻¹',
        'm2_Hz_rad': 'm²·Hz⁻¹·rad⁻¹',
        'm2_s_deg': 'm²·s·deg⁻¹',
        'm2_Hz_deg': 'm²·Hz⁻¹·deg⁻¹'
    }
    colorbar_label = unit_labels.get(spectrum_unit, spectrum_unit)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.025, pad=0.1, ax=ax, extend='both')
    cbar.set_label(colorbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    tick_interval = (vmax - vmin) / 5 if vmax > vmin else 1
    cbar.set_ticks(np.arange(vmin, vmax + 0.5 * tick_interval, tick_interval))

    # Ajuste manual ao invés de tight_layout para evitar warning
    fig.subplots_adjust(left=0.06, right=0.86, top=0.9, bottom=0.05)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # Formato simplificado: source_type_YYYYMMDD_HHMM_spectrum_2D.png
        # source_type já contém o point_name se fornecido (ex: "sar1" para ref=1)
        date_str_formatted = selected_time.strftime("%Y%m%d_%H%M")
        base_filename = f"{source_type}_{date_str_formatted}"
        fig_path = os.path.join(save_dir, f"{base_filename}_spectrum_2D.png")
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        print(f"Figura salva em: {fig_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig
