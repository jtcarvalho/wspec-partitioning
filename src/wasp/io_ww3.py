"""
Functions for leitura e processing of data WW3
"""

import numpy as np
import pandas as pd
import xarray as xr


def find_closest_time(file_path, target_time_dt):
    """
    Find the closest timestamp in the WW3 dataset to the target time.
    
    Parameters:
    -----------
    file_path : str
        Path to WW3 NetCDF file
    target_time_dt : pd.Timestamp
        Target time to search for
    
    Returns:
    --------
    itime : int
        Index of closest time
    closest_time : pd.Timestamp
        Closest timestamp found
    time_diff_hours : float
        Temporal difference in hours
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
    Load 2D directional spectrum from WW3 and coordinates.
    
    Parameters:
    -----------
    file_path : str
        Path to WW3 NetCDF file
    time_index : int
        Temporal index to load
    
    Returns:
    --------
    E2d : ndarray (NF, ND)
        2D directional spectrum [m²·s·rad⁻¹]
    freq : ndarray (NF,)
        Frequencies [Hz]
    dirs : ndarray (ND,)
        Directions [degrees]
    dirs_rad : ndarray (ND,)
        Directions [radians]
    lon : float
        Point longitude
    lat : float
        Point latitude
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
