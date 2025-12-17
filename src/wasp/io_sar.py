"""
Functions for leitura e conversion of data SAR (Sentinel-1)
"""

import numpy as np
import pandas as pd


def convert_sar_energy_units(E_sar, k, phi):
    """
    Converts spectrum SAR of wavenumber for frequency in m²·s·rad⁻¹ (standard WW3).
    
    Apply conversion using o Jacobian of the relation of dispersion:
    E(f,θ) [m²·s·rad⁻¹] = E(k,θ) [m⁴] × |dk/df| × (π/180)
    
    Where:
    - |dk/df| = 8π²f/g  (Jacobian of the relation of dispersion ω² = gk)
    - π/180 converts of θ[degrees] for θ[radians]
    
    Parameters:
    -----------
    E_sar : ndarray
        Spectrum SAR in wavenumber [m⁴]
    k : ndarray
        Numbers of wave [rad/m]
    phi : ndarray
        Directions [degrees]
    
    Returns:
    --------
    E_m2_s_rad : ndarray (NF, ND)
        Spectrum convertido [m²·s·rad⁻¹]
    freq : ndarray (NF,)
        Frequencies [Hz]
    phi_oceanographic : ndarray (ND,)
        Directions oceanographic [degrees]
    dirs_rad : ndarray (ND,)
        Directions in radians
    """
    g = 9.81
    omega = np.sqrt(g * k)
    freq = omega / (2 * np.pi)
    
    # Jacobian of k -> f transformation
    # From dispersion relation: ω² = gk, where ω = 2πf
    # dk/df = 8π²f/g
    dkdf = 8 * np.pi**2 * freq / g
    dkdf_matrix = dkdf.reshape(-1, 1)
    
    # Conversion of direction: degrees -> radians in the densidade espectral
    # E(θ_rad) = E(θ_deg) × (π/180)
    deg_to_rad_factor = np.pi / 180.0
    
    # E(k,θ) [m⁴] -> E(f,θ) [m²·s·rad⁻¹]
    E_m2_s_rad = E_sar * dkdf_matrix * deg_to_rad_factor
    
    # Ajuste shape for (NF, ND)
    if E_m2_s_rad.shape[0] != len(freq):
        E_m2_s_rad = E_m2_s_rad.T
    
    # SAR data already in oceanographic convention (direction going to)
    phi_oceanographic = phi
    dirs_rad = np.radians(phi_oceanographic)
    
    # Diagnostic: calculate m0 and Hs
    ddir = 2 * np.pi / len(phi)
    m0 = 0
    for j in range(len(phi)):
        E_clean = np.where(np.isfinite(E_m2_s_rad[:, j]) & (E_m2_s_rad[:, j] >= 0), 
                          E_m2_s_rad[:, j], 0)
        m0 += np.trapezoid(E_clean, freq) * ddir
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


def load_sar_spectrum(ds, date_time=None, index=0):
    """
    Load SAR spectrum for specific date/time.
    
    Compatible with preprocessed Sentinel-1A/B files (CMEMS).
    Automatically converts from SAR (m⁴) to m²·s·rad⁻¹ (WW3 standard).
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Opened SAR dataset
    date_time : str or pd.Timestamp, optional
        Specific date/time to search for. If None, uses index.
    index : int
        Index of the observation to load (used if date_time=None)
    
    Returns:
    --------
    E2d : ndarray (NF, ND)
        Spectrum directional 2D [m²·s·rad⁻¹]
    freq : ndarray (NF,)
        Frequencies [Hz]
    dirs : ndarray (ND,)
        Directions [degrees]
    dirs_rad : ndarray (ND,)
        Directions [radians]
    actual_time : pd.Timestamp
        Timestamp of the observation loaded
    """
    print("Available variables in SAR file:", list(ds.variables.keys()))
    
    # Auxiliary function to search for variable by multiple possible names
    def get_var(ds, varnames):
        for var in varnames:
            if var in ds.variables:
                return ds[var].values
        raise ValueError(f"None of variables {varnames} found in the file SAR.")

    # Nomes possible for each variable
    wave_spec_names = ['wave_spec', 'obs_params/wave_spec', 'wave_spectrum', 
                       'obs_params/wave_spectrum']
    k_names = ['wavenumber_spec', 'obs_params/wavenumber_spec']
    phi_names = ['direction_spec', 'obs_params/direction_spec']
    time_names = ['time', 'obs_params/time', 'TIME', 'obs_time', 
                  'acquisition_time', 'time_center', 'valid_time', 't']

    try:
        # Tentar formato preprocessado CMEMS
        E_sar = get_var(ds, wave_spec_names)  # (NF, ND, Nobs)
        k = get_var(ds, k_names)  # (NF,)
        phi = get_var(ds, phi_names)  # (ND,)
        
        # Buscar time
        times = None
        for tname in time_names:
            if tname in ds.variables:
                times = ds[tname].values
                break
        
        # Select specific observation
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
        # Tentar formato antigo (oswPolSpec)
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
            raise ValueError("File SAR not has variables recognized "
                           "(nor wave_spec nor oswPolSpec)")

    print(f"Shape k: {k.shape}")
    print(f"Shape phi: {phi.shape}")
    
    # Convert SAR (m⁴) to m²·s·rad⁻¹ (WW3 standard)
    E2d, freq, dirs, dirs_rad = convert_sar_energy_units(E_sar, k, phi)
    
    return E2d, freq, dirs, dirs_rad, actual_time
