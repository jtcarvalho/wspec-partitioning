"""
Functions for matching SAR observations with NDBC buoys and WW3 model data
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate great circle distance between two points in km.
    
    Parameters:
    -----------
    lon1, lat1 : float or array
        Longitude and latitude of first point(s) in degrees
    lon2, lat2 : float or array
        Longitude and latitude of second point(s) in degrees
    
    Returns:
    --------
    float or array
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def check_ndbc_has_spectral_data(ndbc_file):
    """
    Check if NDBC file contains spectral directional data.
    
    Parameters:
    -----------
    ndbc_file : str or Path
        Path to NDBC NetCDF file
    
    Returns:
    --------
    bool
        True if file has required spectral variables
    """
    required_vars = [
        'spectral_wave_density',
        'wave_spectrum_r1',
        'wave_spectrum_r2',
        'mean_wave_dir',
        'principal_wave_dir',
        'frequency'
    ]
    
    try:
        with xr.open_dataset(ndbc_file) as ds:
            has_all = all(var in ds.variables for var in required_vars)
            return has_all
    except:
        return False


def scan_ndbc_stations(ndbc_base_dir, year_range=None):
    """
    Scan NDBC directory and identify stations with spectral data.
    
    Parameters:
    -----------
    ndbc_base_dir : str or Path
        Base directory containing NDBC station folders
    year_range : tuple of int, optional
        (start_year, end_year) to filter files
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: station_id, year, file_path, has_spectral
    """
    ndbc_path = Path(ndbc_base_dir)
    results = []
    
    # List all station directories
    station_dirs = sorted([d for d in ndbc_path.iterdir() if d.is_dir()])
    
    print(f"Scanning {len(station_dirs)} NDBC stations...")
    
    for station_dir in station_dirs:
        station_id = station_dir.name
        
        # Find all NetCDF files for this station
        nc_files = sorted(station_dir.glob('*.nc'))
        
        for nc_file in nc_files:
            # Extract year from filename (e.g., 41010w2020.nc)
            try:
                year = int(nc_file.stem[-4:])
            except:
                continue
            
            # Filter by year range if specified
            if year_range and (year < year_range[0] or year > year_range[1]):
                continue
            
            # Check if has spectral data
            has_spectral = check_ndbc_has_spectral_data(nc_file)
            
            results.append({
                'station_id': station_id,
                'year': year,
                'file_path': str(nc_file),
                'has_spectral': has_spectral
            })
    
    df = pd.DataFrame(results)
    
    # Get station coordinates from first file of each station
    coords = []
    for station_id in df['station_id'].unique():
        first_file = df[df['station_id'] == station_id].iloc[0]['file_path']
        try:
            with xr.open_dataset(first_file) as ds:
                lon = float(ds.longitude.values.item())
                lat = float(ds.latitude.values.item())
                coords.append({'station_id': station_id, 'lon': lon, 'lat': lat})
        except:
            coords.append({'station_id': station_id, 'lon': np.nan, 'lat': np.nan})
    
    df_coords = pd.DataFrame(coords)
    df = df.merge(df_coords, on='station_id', how='left')
    
    return df


def find_sar_ndbc_matches(sar_dir, ndbc_info_df, max_distance_km=50, 
                          max_time_diff_hours=3, year_range=None, limit_files=None):
    """
    Find SAR observations within distance and time of NDBC stations.
    
    Parameters:
    -----------
    sar_dir : str or Path
        Directory containing SAR NetCDF files
    ndbc_info_df : pd.DataFrame
        DataFrame from scan_ndbc_stations() with spectral stations
    max_distance_km : float
        Maximum distance for a match (km)
    max_time_diff_hours : float
        Maximum time difference for a match (hours)
    year_range : tuple of int, optional
        (start_year, end_year) to filter SAR files
    limit_files : int, optional
        Limit processing to first N files (for testing)
    
    Returns:
    --------
    pd.DataFrame
        Matches with columns: station_id, sar_file, sar_index, sar_lon, sar_lat,
        sar_time, ndbc_lon, ndbc_lat, distance_km, ndbc_file
    """
    sar_path = Path(sar_dir)
    
    # Filter only stations with spectral data
    spectral_stations = ndbc_info_df[ndbc_info_df['has_spectral']].copy()
    
    if len(spectral_stations) == 0:
        print("⚠️  No NDBC stations with spectral data found!")
        return pd.DataFrame()
    
    # Get unique stations with coordinates
    stations = spectral_stations[['station_id', 'lon', 'lat']].drop_duplicates()
    stations = stations.dropna(subset=['lon', 'lat'])
    
    print(f"\nSearching matches for {len(stations)} stations with spectral data")
    print(f"Max distance: {max_distance_km} km")
    print(f"Max time diff: {max_time_diff_hours} hours")
    
    # Get all SAR files
    sar_files = sorted(sar_path.glob('*.nc'))
    
    # Filter by year if specified
    if year_range:
        sar_files = [f for f in sar_files 
                     if year_range[0] <= int(f.stem.split('_')[1][:4]) <= year_range[1]]
    
    # Limit files if requested (for testing)
    if limit_files:
        sar_files = sar_files[:limit_files]
    
    print(f"Processing {len(sar_files)} SAR files...")
    
    matches = []
    
    for sar_idx, sar_file in enumerate(sar_files):
        if (sar_idx + 1) % 100 == 0:
            print(f"  Processed {sar_idx + 1}/{len(sar_files)} files, found {len(matches)} matches")
        
        try:
            with xr.open_dataset(sar_file) as ds_sar:
                sar_lons = ds_sar['longitude'].values
                sar_lats = ds_sar['latitude'].values
                sar_times = pd.to_datetime(ds_sar['time'].values)
                
                # Handle both 1D and 2D coordinate arrays
                # SAR files can have shape (nobs,) or (nobs, ndir)
                if sar_lons.ndim == 2:
                    # Take first direction column for coordinates
                    sar_lons = sar_lons[:, 0]
                    sar_lats = sar_lats[:, 0]
                
                # Check each SAR observation
                for obs_idx in range(len(sar_lons)):
                    sar_lon = sar_lons[obs_idx]
                    sar_lat = sar_lats[obs_idx]
                    sar_time = sar_times[obs_idx]
                    
                    # Skip invalid coordinates
                    if np.isnan(sar_lon) or np.isnan(sar_lat):
                        continue
                    
                    # Calculate distance to all stations
                    distances = haversine_distance(
                        sar_lon, sar_lat,
                        stations['lon'].values, stations['lat'].values
                    )
                    
                    # Find stations within max distance
                    close_stations = stations[distances <= max_distance_km].copy()
                    close_stations['distance_km'] = distances[distances <= max_distance_km]
                    
                    if len(close_stations) == 0:
                        continue
                    
                    # For each close station, check if we have data at this time
                    for _, station in close_stations.iterrows():
                        station_id = station['station_id']
                        year = sar_time.year
                        
                        # Find NDBC file for this year
                        ndbc_files = spectral_stations[
                            (spectral_stations['station_id'] == station_id) &
                            (spectral_stations['year'] == year)
                        ]
                        
                        if len(ndbc_files) == 0:
                            continue
                        
                        ndbc_file = ndbc_files.iloc[0]['file_path']
                        
                        # Check if NDBC has data within time window
                        try:
                            with xr.open_dataset(ndbc_file) as ds_ndbc:
                                ndbc_times = pd.to_datetime(ds_ndbc.time.values)
                                time_diffs = np.abs(ndbc_times - sar_time)
                                min_time_diff = time_diffs.min()
                                
                                if min_time_diff <= pd.Timedelta(hours=max_time_diff_hours):
                                    closest_idx = np.argmin(time_diffs)
                                    ndbc_time = ndbc_times[closest_idx]
                                    
                                    matches.append({
                                        'station_id': station_id,
                                        'station_lon': station['lon'],
                                        'station_lat': station['lat'],
                                        'sar_file': sar_file.name,
                                        'sar_index': obs_idx,
                                        'sar_lon': sar_lon,
                                        'sar_lat': sar_lat,
                                        'sar_time': sar_time,
                                        'ndbc_file': Path(ndbc_file).name,
                                        'ndbc_time': ndbc_time,
                                        'ndbc_time_index': closest_idx,
                                        'distance_km': station['distance_km'],
                                        'time_diff_hours': min_time_diff.total_seconds() / 3600
                                    })
                        except Exception as e:
                            continue
                            
        except Exception as e:
            print(f"  ⚠ Error processing {sar_file.name}: {e}")
            continue
    
    print(f"\n✓ Found {len(matches)} total matches")
    
    return pd.DataFrame(matches)


def add_ww3_info(matches_df, ww3_dir):
    """
    Add WW3 file information to matches DataFrame.
    
    Parameters:
    -----------
    matches_df : pd.DataFrame
        DataFrame with SAR-NDBC matches
    ww3_dir : str or Path
        Directory containing WW3 NetCDF files (format: ww3_STATION.nc)
    
    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with ww3_file and ww3_available columns
    """
    ww3_path = Path(ww3_dir)
    
    # Create copy to avoid modifying original
    df = matches_df.copy()
    
    # Add WW3 columns
    df['ww3_file'] = None
    df['ww3_available'] = False
    
    # Get list of available WW3 files
    ww3_files = {f.stem.split('_')[1]: f.name for f in ww3_path.glob('ww3_*.nc')}
    
    print(f"\nChecking WW3 data availability...")
    print(f"Found WW3 files for {len(ww3_files)} stations")
    
    # Check each match
    for idx, row in df.iterrows():
        station_id = str(row['station_id'])
        
        if station_id in ww3_files:
            ww3_file = ww3_files[station_id]
            
            # Verify WW3 has data at this time
            try:
                ww3_path_full = ww3_path / ww3_file
                with xr.open_dataset(ww3_path_full) as ds_ww3:
                    ww3_times = pd.to_datetime(ds_ww3.time.values)
                    sar_time = pd.to_datetime(row['sar_time'])
                    
                    # Check if time exists in WW3 data
                    time_diffs = np.abs(ww3_times - sar_time)
                    min_diff = time_diffs.min()
                    
                    if min_diff <= pd.Timedelta(hours=3):
                        df.at[idx, 'ww3_file'] = ww3_file
                        df.at[idx, 'ww3_available'] = True
            except Exception as e:
                continue
    
    n_with_ww3 = df['ww3_available'].sum()
    print(f"✓ {n_with_ww3}/{len(df)} matches have WW3 data available")
    
    return df
