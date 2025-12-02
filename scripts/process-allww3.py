"""
WW3 Spectral Partitioning Processing Script

This script processes WW3 spectral data, applies partitioning algorithm,
and saves results to CSV files.
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
from utils import calculate_wave_parameters
from partition import partition_spectrum

#case = 'lee'
#case = 'freddy'
case = 'surigae'
#case = 'all'

# Configuration
OUTPUT_DIR = f'../data/{case}/partition'
CSV_PATH = f'../auxdata/sar_matches_{case}_track_3day.csv'
WW3_DATA_PATH = f'/Users/jtakeo/data/ww3/{case}'
MIN_ENERGY_THRESHOLD_FRACTION = 0.01  # 1% of total energy
PEAK_DETECTION_SENSITIVITY = 0.5

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_closest_time(file_path, target_time_dt):
    """
    Find the closest time index in WW3 dataset to target time
    
    Returns:
    --------
    tuple: (time_index, closest_time, time_diff_hours)
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
    Load WW3 spectral data and coordinates
    
    Returns:
    --------
    tuple: (E2d, freq, dirs, dirs_rad, lon, lat)
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


def count_significant_partitions(results, min_energy_threshold):
    """Count partitions with energy above threshold"""
    count = 0
    for i in range(1, len(results['Hs'])):
        if results['energy'][i] > min_energy_threshold:
            count += 1
    return count


# ============================================================================
# PRINTING FUNCTIONS
# ============================================================================

def print_case_header(idx, total_cases, ref, target_time_dt):
    """Print processing case header"""
    print(f"\n{'='*60}")
    print(f"Processing case {idx + 1}/{total_cases}")
    print(f"{'='*60}")
    print(f"Reference ID: {ref}")
    print(f"Target SAR time: {target_time_dt}")


def print_time_match_info(closest_time, itime, time_diff_hours):
    """Print time matching information"""
    print(f"Closest WW3 time: {closest_time}")
    print(f"Time index (itime): {itime}")
    print(f"Time difference: {time_diff_hours:.2f} hours")


def print_partitioning_summary(n_peaks_initial, n_partitions_final):
    """Print partitioning process summary"""
    print("\n" + "="*70)
    print(" SPECTRAL PARTITIONING - PROCESS SUMMARY")
    print("="*70)
    print(f"🔍 Spectral peaks initially identified: {n_peaks_initial}")
    print(f"🔗 After merging nearby systems: {n_partitions_final} partition(s)")
    print("="*70)


def print_partitioning_results(results, min_energy_threshold):
    """Print detailed partitioning results"""
    n_partitions = count_significant_partitions(results, min_energy_threshold)
    
    print("\n" + "="*70)
    print(" PARTITIONING RESULTS")
    print("="*70)
    print(f"Number of partitions found: {n_partitions}")
    print("─"*70)
    
    # Display each partition
    partition_count = 0
    for i in range(1, len(results['Hs'])):
        if results['energy'][i] > min_energy_threshold:
            partition_count += 1
            energy_fraction = (results['energy'][i] / results['total_m0']) * 100
            
            print(f"\nPartition {partition_count}:")
            print(f"  Hs = {results['Hs'][i]:.2f} m")
            print(f"  Tp = {results['Tp'][i]:.2f} s")
            print(f"  Dp = {results['Dp'][i]:.0f}°")
            print(f"  Energy: {results['energy'][i]:.4f} m²")
            print(f"  Energy fraction: {energy_fraction:.1f}%")
    
    # Display total
    print("\n" + "─"*70)
    print(f"Integrated total:")
    print(f"  Hs = {results['total_Hs']:.2f} m")
    print(f"  Tp = {results['total_Tp']:.2f} s")
    print(f"  Dp = {results['total_Dp']:.0f}°")
    print("="*70)


def print_save_confirmation(output_path, df_results):
    """Print save confirmation and preview"""
    print(f"\n✓ Results saved to: {output_path}")
    print(f"\nColumns in CSV: {list(df_results.columns)}")
    print(f"\nPreview:")
    print(df_results.T)


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def create_partition_data_dict(ref, selected_time, lon, lat, file_path, 
                                results, min_energy_threshold):
    """
    Create data dictionary with partition results
    
    Returns:
    --------
    dict: Data dictionary ready for DataFrame conversion
    """
    moments = results['moments']
    m0_total = moments['total'][0]
    m1_total = moments['total'][1]
    m2_total = moments['total'][2]
    
    data = {
        'reference_id': ref,
        'obs_time': selected_time,
        'longitude': float(lon),
        'latitude': float(lat),
        'source_file': os.path.basename(file_path),
        
        # Total spectrum
        'total_energy': results['total_m0'],
        'total_Hs': results['total_Hs'],
        'total_Tp': results['total_Tp'],
        'total_Dp': results['total_Dp'],
        'total_m0': m0_total,
        'total_m1': m1_total,
        'total_m2': m2_total,
    }
    
    # Add partition data (up to 3 partitions)
    for p in range(1, 4):
        if p < len(results['Hs']) and results['energy'][p] > min_energy_threshold:
            data[f'P{p}_energy'] = results['energy'][p]
            data[f'P{p}_Hs'] = results['Hs'][p]
            data[f'P{p}_Tp'] = results['Tp'][p]
            data[f'P{p}_Dp'] = results['Dp'][p]
            data[f'P{p}_m0'] = moments['m0'][p]
            data[f'P{p}_m1'] = moments['m1'][p]
            data[f'P{p}_m2'] = moments['m2'][p]
        else:
            # Fill with zeros if partition doesn't exist
            data[f'P{p}_energy'] = 0.0
            data[f'P{p}_Hs'] = 0.0
            data[f'P{p}_Tp'] = 0.0
            data[f'P{p}_Dp'] = 0.0
            data[f'P{p}_m0'] = 0.0
            data[f'P{p}_m1'] = 0.0
            data[f'P{p}_m2'] = 0.0
    
    return data


def save_partition_results(ref, selected_time, data, output_dir):
    """
    Save partition results to CSV file
    
    Returns:
    --------
    tuple: (output_path, df_results)
    """
    # Create filename
    date_time_formatted = selected_time.strftime('%Y%m%d-%H%M%S')
    output_filename = f'ww3_{ref:03d}_{date_time_formatted}.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    # Create DataFrame and save
    df_results = pd.DataFrame([data])
    df_results.to_csv(output_path, index=False, float_format='%.6f')
    
    return output_path, df_results


def process_single_case(row, idx, total_cases, output_dir):
    """
    Process a single WW3 case
    
    Parameters:
    -----------
    row : pandas.Series
        Row from the input CSV with case information
    idx : int
        Current case index
    total_cases : int
        Total number of cases to process
    output_dir : str
        Output directory for results
    """
    # Extract case information
    ref = int(row['ref'])
    target_time_str = row['time']
    target_time_dt = pd.to_datetime(target_time_str)
    
    # Construct file path
    file_path = f'{WW3_DATA_PATH}/ww3_sar{ref:03d}.nc'
    
    # Print header
    print_case_header(idx, total_cases, ref, target_time_dt)
    
    # Find closest time
    itime, closest_time, time_diff_hours = find_closest_time(file_path, target_time_dt)
    print_time_match_info(closest_time, itime, time_diff_hours)
    
    # Load spectrum
    E2d, freq, dirs, dirs_rad, lon, lat = load_ww3_spectrum(file_path, itime)
    
    # Apply partitioning
    results = partition_spectrum(E2d, freq, dirs_rad, PEAK_DETECTION_SENSITIVITY, 5)
    
    if results is None:
        print("⚠ No spectral peaks identified!")
        return
    
    # Calculate threshold and count partitions
    min_energy_threshold = MIN_ENERGY_THRESHOLD_FRACTION * results['total_m0']
    n_peaks_initial = len(results['peaks'])
    n_partitions_final = count_significant_partitions(results, min_energy_threshold)
    
    # Print results
    print_partitioning_summary(n_peaks_initial, n_partitions_final)
    print_partitioning_results(results, min_energy_threshold)
    
    # Create and save results
    data = create_partition_data_dict(ref, closest_time, lon, lat, file_path,
                                      results, min_energy_threshold)
    output_path, df_results = save_partition_results(ref, closest_time, data, output_dir)
    
    # Print confirmation
    print_save_confirmation(output_path, df_results)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load cases
    df = pd.read_csv(CSV_PATH)
    total_cases = len(df)
    
    print(f"{'='*60}")
    print(f"WW3 SPECTRAL PARTITIONING PROCESSOR")
    print(f"{'='*60}")
    print(f"Total cases to process: {total_cases}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Process each case
    for idx, row in df.iterrows():
        try:
            process_single_case(row, idx, total_cases, OUTPUT_DIR)
        except Exception as e:
            print(f"\n❌ Error processing case {idx + 1}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
