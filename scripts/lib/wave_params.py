"""
Funções para cálculo de parâmetros de onda (Hs, Tp, Dp) a partir de espectros 2D
"""

import numpy as np


def calculate_wave_parameters(E2d, freq, dirs_rad):
    """
    Calcula Hs, Tp, Dp e outros parâmetros do espectro usando integração trapezoidal.
    
    Parameters:
    -----------
    E2d : ndarray (NF, ND)
        Espectro direcional 2D [m²·s·rad⁻¹]
    freq : ndarray (NF,)
        Frequências [Hz]
    dirs_rad : ndarray (ND,)
        Direções [radianos]
    
    Returns:
    --------
    hs : float
        Altura significativa [m]
    tp : float
        Período de pico [s]
    dp : float
        Direção de pico [graus]
    m0 : float
        Momento espectral de ordem 0 [m²]
    delf : ndarray (NF,)
        Incrementos de frequência [Hz]
    ddir : float
        Incremento direcional [rad]
    i_peak : int
        Índice da frequência de pico
    j_peak : int
        Índice da direção de pico
    """
    # Limpar dados inválidos
    E2d_clean = np.where(np.isfinite(E2d) & (E2d >= 0), E2d, 0)
    
    # Calcular incremento direcional
    ddir = 2 * np.pi / len(dirs_rad)
    
    # Calcular incrementos de frequência
    delf = np.zeros_like(freq)
    for i in range(len(freq)-1):
        delf[i] = freq[i+1] - freq[i]
    delf[-1] = delf[-2]
    
    # Calcular momento espectral m0 usando integração trapezoidal
    m0 = 0
    for j in range(len(dirs_rad)):
        m0 += np.trapezoid(E2d_clean[:, j], freq) * ddir
    
    # Calcular espectro 1D para encontrar pico
    spec1d = np.sum(E2d_clean, axis=1) * ddir
    
    # Altura significativa
    hs = 4 * np.sqrt(m0) if m0 > 0 else 0.0
    
    # Período de pico
    i_peak = np.argmax(spec1d) if np.max(spec1d) > 0 else 0
    tp = 1.0 / freq[i_peak] if i_peak < len(freq) and freq[i_peak] > 0 else np.nan
    
    # Direção de pico (média ponderada pela energia)
    j_peak = np.argmax(E2d[i_peak, :]) if i_peak < len(freq) else 0
    
    if np.any(E2d[i_peak, :] > 0):
        weighted_dir = np.sum(E2d[i_peak, :] * dirs_rad) / np.sum(E2d[i_peak, :])
        dp = np.degrees(weighted_dir) % 360
    else:
        dp = np.nan
    
    return hs, tp, dp, m0, delf, ddir, i_peak, j_peak


def spectrum1d_from_2d(E2d, dirs_rad):
    """
    Integra espectro 2D para obter espectro 1D E(f).
    
    Parameters:
    -----------
    E2d : ndarray (NF, ND)
        Espectro direcional 2D
    dirs_rad : ndarray (ND,)
        Direções [radianos]
    
    Returns:
    --------
    spec1d : ndarray (NF,)
        Espectro 1D integrado
    ddir : float
        Incremento direcional usado na integração
    """
    E2d_clean = np.where(np.isfinite(E2d) & (E2d >= 0), E2d, 0)
    ddir = 2 * np.pi / len(dirs_rad)
    spec1d = np.sum(E2d_clean, axis=1) * ddir
    return spec1d, ddir


def convert_meteorological_to_oceanographic(met_dir):
    """
    Converte direção meteorológica (de onde vem) para oceanográfica (para onde vai).
    
    Parameters:
    -----------
    met_dir : float or ndarray
        Direção meteorológica em graus (de onde o vento/onda vem)
        
    Returns:
    --------
    float or ndarray
        Direção oceanográfica em graus (para onde a onda vai)
    """
    return (met_dir + 180) % 360


def convert_spectrum_units(E2d, freq, dirs, from_unit, to_unit):
    """
    Converte espectro entre diferentes unidades de energia.
    
    Parameters:
    -----------
    E2d : ndarray
        Espectro 2D
    freq : ndarray
        Frequências
    dirs : ndarray
        Direções
    from_unit : str
        Unidade de origem: 'm2_s_rad', 'm2_Hz_rad', 'm2_Hz_deg'
    to_unit : str
        Unidade de destino: 'm2_s_rad', 'm2_Hz_rad', 'm2_Hz_deg'
    
    Returns:
    --------
    ndarray
        Espectro convertido
    """
    if from_unit == to_unit:
        return E2d.copy()
    
    result = E2d.copy()
    
    # Conversões entre unidades
    if from_unit == "m2_s_rad" and to_unit == "m2_Hz_rad":
        result = result / (2 * np.pi)
    elif from_unit == "m2_Hz_rad" and to_unit == "m2_s_rad":
        result = result * (2 * np.pi)
    elif from_unit == "m2_s_rad" and to_unit == "m2_Hz_deg":
        result = result / (2 * np.pi) * (180 / np.pi)
    elif from_unit == "m2_Hz_rad" and to_unit == "m2_Hz_deg":
        result = result * (180 / np.pi)
    
    return result
