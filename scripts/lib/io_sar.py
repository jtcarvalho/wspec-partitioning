"""
Funções para leitura e conversão de dados SAR (Sentinel-1)
"""

import numpy as np
import pandas as pd


def convert_sar_energy_units(E_sar, k, phi):
    """
    Converte espectro SAR de número de onda para frequência em m²·s·rad⁻¹ (padrão WW3).
    
    Aplica conversão usando o jacobiano da relação de dispersão:
    E(f,θ) [m²·s·rad⁻¹] = E(k,θ) [m⁴] × |dk/df| × (π/180)
    
    Onde:
    - |dk/df| = 8π²f/g  (jacobiano da relação de dispersão ω² = gk)
    - π/180 converte de θ[graus] para θ[radianos]
    
    Parameters:
    -----------
    E_sar : ndarray
        Espectro SAR em número de onda [m⁴]
    k : ndarray
        Números de onda [rad/m]
    phi : ndarray
        Direções [graus]
    
    Returns:
    --------
    E_m2_s_rad : ndarray (NF, ND)
        Espectro convertido [m²·s·rad⁻¹]
    freq : ndarray (NF,)
        Frequências [Hz]
    phi_oceanographic : ndarray (ND,)
        Direções oceanográficas [graus]
    dirs_rad : ndarray (ND,)
        Direções em radianos
    """
    g = 9.81
    omega = np.sqrt(g * k)
    freq = omega / (2 * np.pi)
    
    # Jacobiano da transformação k -> f
    # Da relação de dispersão: ω² = gk, onde ω = 2πf
    # dk/df = 8π²f/g
    dkdf = 8 * np.pi**2 * freq / g
    dkdf_matrix = dkdf.reshape(-1, 1)
    
    # Conversão de direção: graus -> radianos na densidade espectral
    # E(θ_rad) = E(θ_deg) × (π/180)
    deg_to_rad_factor = np.pi / 180.0
    
    # E(k,θ) [m⁴] -> E(f,θ) [m²·s·rad⁻¹]
    E_m2_s_rad = E_sar * dkdf_matrix * deg_to_rad_factor
    
    # Ajuste shape para (NF, ND)
    if E_m2_s_rad.shape[0] != len(freq):
        E_m2_s_rad = E_m2_s_rad.T
    
    # Dados SAR já estão em convenção oceanográfica (direção PARA onde vai)
    phi_oceanographic = phi
    dirs_rad = np.radians(phi_oceanographic)
    
    # Diagnóstico: calcular m0 e Hs
    ddir = 2 * np.pi / len(phi)
    m0 = 0
    for j in range(len(phi)):
        E_clean = np.where(np.isfinite(E_m2_s_rad[:, j]) & (E_m2_s_rad[:, j] >= 0), 
                          E_m2_s_rad[:, j], 0)
        m0 += np.trapezoid(E_clean, freq) * ddir
    hs = 4 * np.sqrt(m0)
    
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║         CONVERSÃO SAR: m⁴ → m²·s·rad⁻¹ (WW3 units)          ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║ Shape: {str(E_sar.shape):>52} ║")
    print(f"║ Frequências: {len(freq):>2d} bins | Direções: {len(phi):>2d} bins              ║")
    print(f"║ Freq range: {freq[0]:.4f} - {freq[-1]:.4f} Hz                       ║")
    print(f"║ Dir range: {phi[0]:.1f}° - {phi[-1]:.1f}°                            ║")
    print(f"╟──────────────────────────────────────────────────────────────╢")
    print(f"║ Jacobiano dk/df: {np.min(dkdf):.4f} - {np.max(dkdf):.4f}                   ║")
    print(f"║ Fator angular (π/180): {deg_to_rad_factor:.6f}                      ║")
    print(f"╟──────────────────────────────────────────────────────────────╢")
    print(f"║ Parâmetros integrados:                                       ║")
    print(f"║   m0 = {m0:>10.6f} m²                                          ║")
    print(f"║   Hs = {hs:>10.6f} m                                           ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    
    return E_m2_s_rad, freq, phi_oceanographic, dirs_rad


def load_sar_spectrum(ds, date_time=None, index=0):
    """
    Carrega espectro SAR para data/hora específica.
    
    Compatível com arquivos preprocessados Sentinel-1A/B (CMEMS).
    Converte automaticamente de SAR (m⁴) para m²·s·rad⁻¹ (padrão WW3).
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset SAR aberto
    date_time : str or pd.Timestamp, optional
        Data/hora específica para buscar. Se None, usa index.
    index : int
        Índice da observação a carregar (usado se date_time=None)
    
    Returns:
    --------
    E2d : ndarray (NF, ND)
        Espectro direcional 2D [m²·s·rad⁻¹]
    freq : ndarray (NF,)
        Frequências [Hz]
    dirs : ndarray (ND,)
        Direções [graus]
    dirs_rad : ndarray (ND,)
        Direções [radianos]
    actual_time : pd.Timestamp
        Timestamp da observação carregada
    """
    print("Variáveis disponíveis no arquivo SAR:", list(ds.variables.keys()))
    
    # Função auxiliar para buscar variável por múltiplos nomes possíveis
    def get_var(ds, varnames):
        for var in varnames:
            if var in ds.variables:
                return ds[var].values
        raise ValueError(f"Nenhuma das variáveis {varnames} encontrada no arquivo SAR.")

    # Nomes possíveis para cada variável
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
        
        # Buscar tempo
        times = None
        for tname in time_names:
            if tname in ds.variables:
                times = ds[tname].values
                break
        
        # Selecionar observação específica
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
        
        print(f"Usando arquivo preprocessado (CMEMS), shape E_sar: {E_sar.shape}")
    
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
            print(f"Usando arquivo SAR antigo, shape E_sar: {E_sar.shape}")
        else:
            raise ValueError("Arquivo SAR não possui variáveis reconhecidas "
                           "(nem wave_spec nem oswPolSpec)")

    print(f"Shape k: {k.shape}")
    print(f"Shape phi: {phi.shape}")
    
    # Converter SAR (m⁴) para m²·s·rad⁻¹ (padrão WW3)
    E2d, freq, dirs, dirs_rad = convert_sar_energy_units(E_sar, k, phi)
    
    return E2d, freq, dirs, dirs_rad, actual_time
