"""
Funções para leitura e processamento de dados WW3
"""

import numpy as np
import pandas as pd
import xarray as xr


def find_closest_time(file_path, target_time_dt):
    """
    Encontra o timestamp mais próximo no dataset WW3 ao tempo alvo.
    
    Parameters:
    -----------
    file_path : str
        Caminho do arquivo NetCDF do WW3
    target_time_dt : pd.Timestamp
        Tempo alvo para buscar
    
    Returns:
    --------
    itime : int
        Índice do tempo mais próximo
    closest_time : pd.Timestamp
        Timestamp mais próximo encontrado
    time_diff_hours : float
        Diferença temporal em horas
    """
    ds_temp = xr.open_dataset(file_path)
    ww3_times = pd.to_datetime(ds_temp.time.values)
    
    time_diffs = np.abs(ww3_times - target_time_dt)
    itime = np.argmin(time_diffs)
    closest_time = ww3_times[itime]
    time_diff_hours = time_diffs[itime].total_seconds() / 3600
    
    ds_temp.close()
    
    return itime, closest_time, time_diff_hours


def load_ww3_spectrum(file_path, time_index):
    """
    Carrega espectro direcional 2D do WW3 e coordenadas.
    
    Parameters:
    -----------
    file_path : str
        Caminho do arquivo NetCDF do WW3
    time_index : int
        Índice temporal a carregar
    
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
    lon : float
        Longitude do ponto
    lat : float
        Latitude do ponto
    """
    ds = xr.open_dataset(file_path)
    
    E2d = ds.efth[time_index, 0, :, :].values
    freq = ds.frequency.values
    dirs = ds.direction.values
    dirs_rad = np.radians(dirs)
    lon = ds.longitude.values[0]
    lat = ds.latitude.values[0]
    
    ds.close()
    
    return E2d, freq, dirs, dirs_rad, lon, lat
