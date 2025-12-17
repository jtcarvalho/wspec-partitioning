import numpy as np
import pandas as pd
from collections import Counter


def convert_meteorological_to_oceanographic(met_dir):
    """Converts direction meteorological (coming from) for oceanographic (going to).
    
    Args:
        met_dir: Direction meteorological in degrees (of where o vento/wave vem)
        
    Returns:
        Direction oceanographic in degrees (for where a wave vai)
    """
    return (met_dir + 180) % 360

def convert_sar_energy_units(E_sar, k, phi):
    """Convert SAR spectrum from wavenumber to frequency in m²·s·rad⁻¹ (same as WW3).
    
    Conversion correct using only o Jacobian of the relation of dispersion:
    E(f,θ) [m²·s·rad⁻¹] = E(k,θ) [m⁴] × |dk/df| × (π/180)
    
    Where:
    - |dk/df| = 8π²f/g  (Jacobian of the relation of dispersion ω² = gk)
    - π/180 converts of θ[degrees] for θ[radians]
    """
    
    g = 9.81
    omega = np.sqrt(g * k)
    freq = omega / (2 * np.pi)
    
    # Jacobian of k -> f transformation
    # From dispersion relation: ω² = gk, where ω = 2πf
    # Diferenciando: 2ω dω = g dk
    # Portanto: dk/dω = 2ω/g = 2√(gk)/g
    # Como dω = 2π df, temos: dk/df = dk/dω × dω/df = (2ω/g) × 2π = 4πω/g
    # Substituindo ω = √(gk): dk/df = 4π√(gk)/g
    # But for f: ω = 2πf, so √(gk) = 2πf, thus: dk/df = 8π²f/g
    dkdf = 8 * np.pi**2 * freq / g
    dkdf_matrix = dkdf.reshape(-1, 1)
    
    # Conversion of direction: degrees -> radians in the densidade espectral
    # Como o SAR fornece E in bins of degrees, precisamos convert for rad⁻¹
    # A integral ∫E dθ deve ser invariante: ∫E(θ_deg) dθ_deg = ∫E(θ_rad) dθ_rad
    # Como dθ_rad = (π/180) dθ_deg, temos: E(θ_rad) = E(θ_deg) × (π/180)
    deg_to_rad_factor = np.pi / 180.0
    
    # E(k,θ) [m⁴] -> E(f,θ) [m²·s·rad⁻¹]
    # Correct formula: E(f,θ) = E(k,θ) × |dk/df| × (π/180)
    E_m2_s_rad = E_sar * dkdf_matrix * deg_to_rad_factor
    
    # Ajuste shape for (NF, ND)
    if E_m2_s_rad.shape[0] != len(freq):
        E_m2_s_rad = E_m2_s_rad.T
    
    # SAR data already in oceanographic convention (direction going to)
    phi_oceanographic = phi
    dirs_rad = np.radians(phi_oceanographic)
    
    # Calculation of m0 and Hs for diagnostic (using trapezoidal integration)
    ddir = 2 * np.pi / len(phi)
    m0 = 0
    for j in range(len(phi)):
        m0 += np.trapezoid(np.where(np.isfinite(E_m2_s_rad[:, j]) & (E_m2_s_rad[:, j] >= 0), E_m2_s_rad[:, j], 0), freq) * ddir
    hs = 4 * np.sqrt(m0)
    
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║         SAR CONVERSION: m⁴ → m²·s·rad⁻¹ (WW3 units)          ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║ Shape: {str(E_sar.shape):>52} ║")
    print(f"║ Frequencies: {len(freq):>2d} bins | Directions: {len(phi):>2d} bins              ║")
    print(f"║ Freq range: {freq[0]:.4f} - {freq[-1]:.4f} Hz                       ║")
    print(f"║ Dir range: {phi[0]:.1f}° - {phi[-1]:.1f}°                            ║")
    print(f"╟──────────────────────────────────────────────────────────────╢")
    print(f"║ Jacobiano dk/df: {np.min(dkdf):.4f} - {np.max(dkdf):.4f}                   ║")
    print(f"║ Fator angular (π/180): {deg_to_rad_factor:.6f}                      ║")
    print(f"╟──────────────────────────────────────────────────────────────╢")
    print(f"║ Parameters integrated:                                       ║")
    print(f"║   m0 = {m0:>10.6f} m²                                          ║")
    print(f"║   Hs = {hs:>10.6f} m                                           ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    
    return E_m2_s_rad, freq, phi_oceanographic, dirs_rad

def convert_spectrum_units(E2d, freq, dirs, from_unit, to_unit):
    """Converts spectrum between diferentes unidades of energy."""
    # If units are equal, return copy of original
    if from_unit == to_unit:
        return E2d.copy()
    
    result = E2d.copy()
    
    # Conversions between units
    if from_unit == "m2_s_rad" and to_unit == "m2_Hz_rad":
        result = result / (2 * np.pi)
    elif from_unit == "m2_Hz_rad" and to_unit == "m2_s_rad":
        result = result * (2 * np.pi)
    elif from_unit == "m2_s_rad" and to_unit == "m2_Hz_deg":
        result = result / (2 * np.pi) * (180 / np.pi)
    elif from_unit == "m2_Hz_rad" and to_unit == "m2_Hz_deg":
        result = result * (180 / np.pi)
    
    return result

def load_sar_spectrum(ds, date_time=None, index=0):
    """Load SAR spectrum for specific date/time, compatible with preprocessed Sentinel-1A/B (CMEMS) files.
    
    Automatically converts from SAR (m⁴) to m²·s·rad⁻¹ (same as WW3).
    """
    print("Available variables in SAR file:", list(ds.variables.keys()))
    # Search variable by multiple possible names
    def get_var(ds, varnames):
        for var in varnames:
            if var in ds.variables:
                return ds[var].values
        raise ValueError(f"None of variables {varnames} found in the file SAR.")

    # Nomes possible for each variable
    wave_spec_names = ['wave_spec', 'obs_params/wave_spec', 'wave_spectrum', 'obs_params/wave_spectrum']
    k_names = ['wavenumber_spec', 'obs_params/wavenumber_spec']
    phi_names = ['direction_spec', 'obs_params/direction_spec']
    time_names = ['time', 'obs_params/time', 'TIME', 'obs_time', 'acquisition_time', 'time_center', 'valid_time', 't']

    try:
        E_sar = get_var(ds, wave_spec_names)  # (NF, ND, Nobs)
        k = get_var(ds, k_names)  # (NF,)
        phi = get_var(ds, phi_names)  # (ND,)
        # Time
        times = None
        for tname in time_names:
            if tname in ds.variables:
                times = ds[tname].values
                break
        nobs = E_sar.shape[2] if E_sar.ndim == 3 else 1
        if nobs > 1:
            if date_time is not None and times is not None:
                if isinstance(date_time, str):
                    date_time = pd.to_datetime(date_time)
                idx = abs(times - np.datetime64(date_time)).argmin()
                actual_time = pd.to_datetime(times[idx])
            else:
                idx = index
                actual_time = pd.to_datetime(times[idx]) if times is not None else None
            E_sar = np.squeeze(E_sar[:, :, idx])
        else:
            E_sar = np.squeeze(E_sar)
            actual_time = pd.to_datetime(times[0]) if times is not None else None
        print(f"Usando file preprocessado (CMEMS), shape E_sar: {E_sar.shape}")
    except Exception as e:
        # Tenta formato antigo
        if 'oswPolSpec' in ds.variables:
            if 'time' in ds and len(ds.time) > 1:
                if date_time is not None:
                    if isinstance(date_time, str):
                        date_time = pd.to_datetime(date_time)
                    idx = abs(ds.time.values - np.datetime64(date_time)).argmin()
                    actual_time = pd.to_datetime(ds.time.values[idx])
                else:
                    idx = index
                    actual_time = pd.to_datetime(ds.time.values[idx])
                E_sar = np.squeeze(ds.oswPolSpec.values[idx])
            else:
                E_sar = np.squeeze(ds.oswPolSpec.values)
                if 'time' in ds:
                    timestamp = ds.time.values[0]
                    actual_time = pd.to_datetime(timestamp)
                else:
                    actual_time = None
            k = ds.oswK.values
            phi = ds.oswPhi.values
            print(f"Usando file SAR antigo, shape E_sar: {E_sar.shape}")
        else:
            raise ValueError("File SAR not has variables recognized (nor wave_spec nor oswPolSpec)")

    print(f"Shape k: {k.shape}")
    print(f"Shape phi: {phi.shape}")
    # Convert SAR (m⁴) to m²·s·rad⁻¹ (same as WW3)
    E2d, freq, dirs, dirs_rad = convert_sar_energy_units(E_sar, k, phi)
    return E2d, freq, dirs, dirs_rad, actual_time

def spectrum1d_from_2d(E2d, dirs_rad):
    """Integrates spectrum 2D for obtain spectrum 1D E(f) using integration trapezoidal in direction."""
    E2d = np.where(np.isfinite(E2d) & (E2d >= 0), E2d, 0)
    ddir = 2 * np.pi / len(dirs_rad)
    spec1d = np.sum(E2d, axis=1) * ddir
    return spec1d, ddir

def calculate_wave_parameters(E2d, freq, dirs_rad):
    """Calculates Hs, Tp, Dp e outros parameters of spectrum using integration trapezoidal."""
    # Clean data invalid
    E2d_clean = np.where(np.isfinite(E2d) & (E2d >= 0), E2d, 0)
    
    # Calculate increment directional
    ddir = 2 * np.pi / len(dirs_rad)
    
    # Calculate frequency increments for compatibility with old code
    delf = np.zeros_like(freq)
    for i in range(len(freq)-1):
        delf[i] = freq[i+1] - freq[i]
    delf[-1] = delf[-2]
    
    # Calculate moment espectral m0 using integration trapezoidal in frequency
    m0 = 0
    for j in range(len(dirs_rad)):
        m0 += np.trapezoid(E2d_clean[:, j], freq) * ddir
    
    # Calculate spectrum 1D for find peak
    spec1d = np.sum(E2d_clean, axis=1) * ddir
    # Calculate height significant
    hs = 4 * np.sqrt(m0) if m0 > 0 else 0.0
    
    # Findr peak in the spectrum of frequency
    i_peak = np.argmax(spec1d) if np.max(spec1d) > 0 else 0
    tp = 1.0 / freq[i_peak] if i_peak < len(freq) and freq[i_peak] > 0 else np.nan
    
    # Findr direction in the peak
    j_peak = np.argmax(E2d[i_peak, :]) if i_peak < len(freq) else 0
    
    # Calculate direction mean weighted in the peak
    if np.any(E2d[i_peak, :] > 0):
        weighted_dir = np.sum(E2d[i_peak, :] * dirs_rad) / np.sum(E2d[i_peak, :])
        dp = np.degrees(weighted_dir) % 360
    else:
        dp = np.nan
    
    return hs, tp, dp, m0, delf, ddir, i_peak, j_peak


def plot_directional_spectrum(E2d, freq, dirs, selected_time=None, hs=None, tp=None, dp=None):
    """
    Plot 2D polar directional spectrum.
    
    Parameters:
    -----------
    E2d : array-like
        2D directional spectrum energy
    freq : array-like
        Frequency array
    dirs : array-like
        Direction array (degrees)
    selected_time : datetime, optional
        Timestamp for the data. If None, date will not be shown in statistics box
    hs : float, optional
        Significant wave height (m). If None, will not be shown in statistics box
    tp : float, optional
        Peak period (s). If None, will not be shown in statistics box
    dp : float, optional
        Peak direction (degrees). If None, will not be shown in statistics box
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.cm import ScalarMappable
    
    Eplot = np.nan_to_num(E2d, nan=0.0, neginf=0.0, posinf=0.0)

    # Ensure 1D arrays
    freq_plot = np.asarray(freq).flatten()
    dirs_plot = np.asarray(dirs).flatten()

    # Convert directions to radians & sort
    dirs_rad_plot = np.radians(dirs_plot)
    sort_idx = np.argsort(dirs_rad_plot)
    dirs_sorted = dirs_rad_plot[sort_idx]
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
        period = np.where(freq_plot > 0, 1.0 / freq_plot, 0)

    theta, r = np.meshgrid(dirs_sorted, period)

    # Default color scales
    vmin = 2.
    vmax = 66.
    # step = max((vmax - vmin)/50.0, 0.5)
   #levels = np.arange(vmin + step, vmax + step*0.51, step)
    levels = np.arange(vmin, vmax+2, 2)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')

    cs = ax.contour(theta, r, Eplot_sorted, levels, cmap='rainbow', vmin=vmin, vmax=vmax)

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
    title = 'Directional Spectrum'
    ax.set_title(title, fontsize=16, color='k', pad=30)

    # Stats box - only show if we have at least one parameter to display
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

    colorbar_label = 'm²·s·rad⁻¹'

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap='rainbow', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.025, pad=0.1, ax=ax, extend='both')
    cbar.set_label(colorbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    #tick_interval = (vmax - vmin) / 5
    tick_interval = 8
    cbar.set_ticks(np.arange(vmin, vmax + 0.5 * tick_interval, tick_interval))

    # Manual adjustment instead of tight_layout
    fig.subplots_adjust(left=0.06, right=0.86, top=0.9, bottom=0.05)

    plt.show()
    
    return fig, ax
