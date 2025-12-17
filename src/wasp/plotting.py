"""
Functions for plotting of spectra directional e visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable


def plot_directional_spectrum(E2d, freq, dirs, selected_time=None, hs=None, tp=None, dp=None):
    """
    Plot 2D directional spectrum in polar coordinates.
    
    Parameters:
    -----------
    E2d : ndarray
        Spectrum directional 2D [m²·s·rad⁻¹]
    freq : ndarray
        Array of frequencies [Hz]
    dirs : ndarray
        Array of directions [degrees]
    selected_time : datetime, optional
        Timestamp dos data
    hs : float, optional
        Height significant [m]
    tp : float, optional
        Period of peak [s]
    dp : float, optional
        Direction of peak [degrees]
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figura gerada
    ax : matplotlib.axes.Axes
        Eixos polares of the figura
    """
    Eplot = np.nan_to_num(E2d, nan=0.0, neginf=0.0, posinf=0.0)

    # Garantir arrays 1D
    freq_plot = np.asarray(freq).flatten()
    dirs_plot = np.asarray(dirs).flatten()

    # Convert directions for radians e ordenar
    dirs_rad_plot = np.radians(dirs_plot)
    sort_idx = np.argsort(dirs_rad_plot)
    dirs_sorted = dirs_rad_plot[sort_idx]
    Eplot_sorted = Eplot[:, sort_idx]

    # Ensure periodic continuity (0 to 2π)
    if not np.isclose(dirs_sorted[0], 0.0):
        dirs_sorted = np.insert(dirs_sorted, 0, 0.0)
        Eplot_sorted = np.insert(Eplot_sorted, 0, Eplot_sorted[:, 0], axis=1)
    if not np.isclose(dirs_sorted[-1], 2*np.pi):
        dirs_sorted = np.append(dirs_sorted, 2*np.pi)
        Eplot_sorted = np.concatenate([Eplot_sorted, Eplot_sorted[:, 0:1]], axis=1)

    # Eixo radial = period (s)
    with np.errstate(divide='ignore', invalid='ignore'):
        period = np.where(freq_plot > 0, 1.0 / freq_plot, 0)

    theta, r = np.meshgrid(dirs_sorted, period)

    # Escalas of cor
    vmin = 2.
    vmax = 66.
    levels = np.arange(vmin, vmax+2, 2)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')

    cs = ax.contour(theta, r, Eplot_sorted, levels, cmap='rainbow', vmin=vmin, vmax=vmax)

    # Estilo dos axes
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
    title = 'Directional Spectrum'
    ax.set_title(title, fontsize=16, color='k', pad=30)

    # Statistics box (if there is data)
    show_stats = selected_time is not None or hs is not None or tp is not None or dp is not None
    if show_stats:
        stats_ax = fig.add_axes([0.75, 0.7, 0.2, 0.15], facecolor='white')
        stats_ax.patch.set_alpha(0.8)
        stats_ax.patch.set_edgecolor('black')
        stats_ax.patch.set_linewidth(1.5)
        stats_ax.axis('off')

        stats_ax.text(0.7, 1.9, 'Statistics', fontsize=14, color='k', ha='center', va='center', weight='bold')
        
        y_offset = 1.7
        if selected_time is not None:
            date_str = selected_time.strftime('%Y-%m-%d %H:%M:%S')
            stats_ax.text(0.7, y_offset, f'Date: {date_str}', fontsize=12, color='k', ha='center', va='center')
            y_offset -= 0.15
        
        if hs is not None:
            stats_ax.text(0.7, y_offset, f'Hs: {hs:.2f} m', fontsize=12, color='k', ha='center', va='center')
            y_offset -= 0.15
        
        if tp is not None:
            stats_ax.text(0.7, y_offset, f'Tp: {tp:.1f} s', fontsize=12, color='k', ha='center', va='center')
            y_offset -= 0.15
        
        if dp is not None:
            stats_ax.text(0.7, y_offset, f'Dp: {dp:.1f}°', fontsize=12, color='k', ha='center', va='center')

    # Barra of cores
    colorbar_label = 'm²·s·rad⁻¹'
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap='rainbow', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.025, pad=0.1, ax=ax, extend='both')
    cbar.set_label(colorbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    tick_interval = 8
    cbar.set_ticks(np.arange(vmin, vmax + 0.5 * tick_interval, tick_interval))

    # Ajuste manual
    fig.subplots_adjust(left=0.06, right=0.86, top=0.9, bottom=0.05)

    plt.show()
    
    return fig, ax
