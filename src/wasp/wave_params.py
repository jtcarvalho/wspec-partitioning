"""
Functions for calculating wave parameters (Hs, Tp, Dp) from 2D spectra
"""

import numpy as np


def calculate_wave_parameters(E2d, freq, dirs_rad):
    """
    Calculate Hs, Tp, Dp and other spectral parameters using trapezoidal integration.
    
    Parameters:
    -----------
    E2d : ndarray (NF, ND)
        Spectrum directional 2D [m²·s·rad⁻¹]
    freq : ndarray (NF,)
        Frequencies [Hz]
    dirs_rad : ndarray (ND,)
        Directions [radians]
    
    Returns:
    --------
    hs : float
        Height significant [m]
    tp : float
        Period of peak [s]
    dp : float
        Direction of peak [degrees]
    m0 : float
        Momento espectral of order 0 [m²]
    delf : ndarray (NF,)
        Incrementos of frequency [Hz]
    ddir : float
        Incremento directional [rad]
    i_peak : int
        Index of the frequency of peak
    j_peak : int
        Index of the direction of peak
    """
    # Clean data invalid
    E2d_clean = np.where(np.isfinite(E2d) & (E2d >= 0), E2d, 0)
    
    # Calculate increment directional
    ddir = 2 * np.pi / len(dirs_rad)
    
    # Calculate incrementos of frequency
    delf = np.zeros_like(freq)
    for i in range(len(freq)-1):
        delf[i] = freq[i+1] - freq[i]
    delf[-1] = delf[-2]
    
    # Calculate moment espectral m0 using integration trapezoidal
    m0 = 0
    for j in range(len(dirs_rad)):
        m0 += np.trapezoid(E2d_clean[:, j], freq) * ddir
    
    # Calculate spectrum 1D for find peak
    spec1d = np.sum(E2d_clean, axis=1) * ddir
    
    # Height significant
    hs = 4 * np.sqrt(m0) if m0 > 0 else 0.0
    
    # Peak period
    i_peak = np.argmax(spec1d) if np.max(spec1d) > 0 else 0
    tp = 1.0 / freq[i_peak] if i_peak < len(freq) and freq[i_peak] > 0 else np.nan
    
    # Direction of peak (mean weighted by energy)
    j_peak = np.argmax(E2d[i_peak, :]) if i_peak < len(freq) else 0
    
    if np.any(E2d[i_peak, :] > 0):
        weighted_dir = np.sum(E2d[i_peak, :] * dirs_rad) / np.sum(E2d[i_peak, :])
        dp = np.degrees(weighted_dir) % 360
    else:
        dp = np.nan
    
    return hs, tp, dp, m0, delf, ddir, i_peak, j_peak


def spectrum1d_from_2d(E2d, dirs_rad):
    """
    Integrates spectrum 2D for obtain spectrum 1D E(f).
    
    Parameters:
    -----------
    E2d : ndarray (NF, ND)
        Spectrum directional 2D
    dirs_rad : ndarray (ND,)
        Directions [radians]
    
    Returns:
    --------
    spec1d : ndarray (NF,)
        Spectrum 1D integrated
    ddir : float
        Incremento directional used in the integration
    """
    E2d_clean = np.where(np.isfinite(E2d) & (E2d >= 0), E2d, 0)
    ddir = 2 * np.pi / len(dirs_rad)
    spec1d = np.sum(E2d_clean, axis=1) * ddir
    return spec1d, ddir


def convert_meteorological_to_oceanographic(met_dir):
    """
    Converts direction meteorological (coming from) for oceanographic (going to).
    
    Parameters:
    -----------
    met_dir : float or ndarray
        Direction meteorological in degrees (of where o vento/wave vem)
        
    Returns:
    --------
    float or ndarray
        Direction oceanographic in degrees (for where a wave vai)
    """
    return (met_dir + 180) % 360


def convert_spectrum_units(E2d, freq, dirs, from_unit, to_unit):
    """
    Converts spectrum between diferentes unidades of energy.
    
    Parameters:
    -----------
    E2d : ndarray
        Spectrum 2D
    freq : ndarray
        Frequencies
    dirs : ndarray
        Directions
    from_unit : str
        Unidade of origem: 'm2_s_rad', 'm2_Hz_rad', 'm2_Hz_deg'
    to_unit : str
        Unidade of destino: 'm2_s_rad', 'm2_Hz_rad', 'm2_Hz_deg'
    
    Returns:
    --------
    ndarray
        Spectrum convertido
    """
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
