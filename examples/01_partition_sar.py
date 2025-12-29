"""
STEP 1: Partition SAR Spectra

This script processes 2D SAR (Sentinel-1) spectra, applies the partitioning 
algorithm, and saves the results to CSV.

Workflow:
1. Read SAR file (NetCDF) with 2D spectra
2. Convert from wavenumber (k) to frequency (f)
3. Apply watershed partitioning algorithm
4. Save Hs, Tp, Dp for each identified partition
"""

import os
import pandas as pd
import xarray as xr
import numpy as np

# Import from wasp package
from wasp.io_sar import load_sar_spectrum
from wasp.wave_params import calculate_wave_parameters
from wasp.partition import partition_spectrum

# ============================================================================
# CONFIGURATION
# ============================================================================

# case = 'surigae'
# case = 'lee'
# case = 'freddy'
case = 'all'

# Directories
OUTPUT_DIR = f'../data/{case}/partition'
SAR_DATA_PATH = f'/Users/jtakeo/data/sentinel1ab/{case}'
CSV_PATH = f'../auxdata/sar_matches_{case}_track.csv'

# Partitioning parameters
MIN_ENERGY_THRESHOLD_FRACTION = 0.01  # 1% of the energy total (post-processing filter)
MAX_PARTITIONS = 5

# Parameters for partition_spectrum() function
THRESHOLD_PERCENTILE = 99.5       # SAR: Conservative threshold for high-resolution data
MERGE_FACTOR = 0.3                # SAR: Preserves distinct systems due to clear directional separation

# NetCDF group name (CMEMS structure)
GROUP_NAME = "obs_params"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_corrected_total_hs(results, min_energy_threshold):
    """
    Calculates Hs total corrected a from of the sum of energies of partitions
    
    Returns:
    --------
    energy_sum : float
        Sum of energies of partitions [m²]
    Hs : float
        Height significant total [m]
    """
    energy_sum = sum(results['energy'][i] for i in range(len(results['energy'])) 
                     if i == 0 or results['energy'][i] > min_energy_threshold)
    Hs = 4 * np.sqrt(energy_sum)
    return energy_sum, Hs


def count_significant_partitions(results, min_energy_threshold):
    """Count partitions with energy above of threshold"""
    return sum(1 for i in range(1, len(results['Hs'])) 
               if results['energy'][i] > min_energy_threshold)


# ============================================================================
# I/O FUNCTIONS
# ============================================================================

def load_sar_data(file_path, index):
    """
    Load dataset SAR e extracts spectrum
    
    Parameters:
    -----------
    file_path : str
        Path of file NetCDF SAR
    index : int
        Index of the observation a load
    
    Returns:
    --------
    ds : xarray.Dataset
        Dataset SAR opened
    E2d : ndarray
        Spectrum 2D [m²·s·rad⁻¹]
    freq : ndarray
        Frequencies [Hz]
    dirs : ndarray
        Directions [degrees]
    dirs_rad : ndarray
        Directions [radians]
    selected_time : pd.Timestamp
        Timestamp of the observation
    """
    ds = xr.open_dataset(file_path, group=GROUP_NAME)
    E2d, freq, dirs, dirs_rad, selected_time = load_sar_spectrum(
        ds, date_time=None, index=index
    )
    return ds, E2d, freq, dirs, dirs_rad, selected_time


def extract_sar_metadata(ds, index):
    """
    Extracts metadata of dataset SAR
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset SAR
    index : int
        Index of the observation
    
    Returns:
    --------
    quality_flag : int
        Flag of quality (0 = best)
    actual_time : pd.Timestamp
        Timestamp of the observation
    lon : float
        Longitude
    lat : float
        Latitude
    """
    quality_flag = ds.L2_partition_quality_flag[index].values
    actual_time = pd.to_datetime(ds.time[index].values)
    lon = ds.longitude[index].values if 'longitude' in ds else None
    lat = ds.latitude[index].values if 'latitude' in ds else None
    return quality_flag, actual_time, lon, lat


# ============================================================================
# PROCESSAMENTO DE DADOS
# ============================================================================

def create_partition_data_dict(ref, index, quality_flag, date_time, lon, lat, 
                                file_path, energy_sum, Hs, results, min_energy_threshold):
    """
    Create dictionary with results of partitioning
    
    Returns:
    --------
    dict: Dictionary ready for conversion in DataFrame
    """
    moments = results['moments']
    
    data = {
        'reference_id': ref,
        'obs_index': index,
        'quality_flag': quality_flag,
        'obs_time': date_time,
        'longitude': float(lon),
        'latitude': float(lat),
        'source_file': os.path.basename(file_path),
        
        # Spectrum total
        'total_energy': energy_sum,
        'total_Hs': Hs,
        'total_Tp': results['total_Tp'],
        'total_Dp': results['total_Dp'],
        'total_m0': moments['total'][0],
        'total_m1': moments['total'][1],
        'total_m2': moments['total'][2],
    }
    
    # Add partition data (up to 3 partitions)
    for p in range(1, 4):
        if p < len(results['Hs']) and results['energy'][p] > min_energy_threshold:
            data.update({
                f'P{p}_energy': results['energy'][p],
                f'P{p}_Hs': results['Hs'][p],
                f'P{p}_Tp': results['Tp'][p],
                f'P{p}_Dp': results['Dp'][p],
                f'P{p}_m0': moments['m0'][p],
                f'P{p}_m1': moments['m1'][p],
                f'P{p}_m2': moments['m2'][p],
            })
        else:
            # Fill with zeros if partition doesn't exist
            data.update({
                f'P{p}_energy': 0.0, 
                f'P{p}_Hs': 0.0, 
                f'P{p}_Tp': 0.0,
                f'P{p}_Dp': 0.0, 
                f'P{p}_m0': 0.0, 
                f'P{p}_m1': 0.0, 
                f'P{p}_m2': 0.0,
            })
    
    return data


def save_partition_results(ref, index, date_time, data, output_dir):
    """
    Save results of partitioning in file CSV
    
    Returns:
    --------
    tuple: (output_path, df_results)
    """
    dt = pd.to_datetime(date_time)
    date_time_formatted = dt.strftime('%Y%m%d-%H%M%S')
    output_filename = f'sar_{ref:03d}_{index}_{date_time_formatted}.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    df_results = pd.DataFrame([data])
    df_results.to_csv(output_path, index=False, float_format='%.6f')
    
    return output_path, df_results


# ============================================================================
# PRINTING FUNCTIONS
# ============================================================================

def print_case_header(idx, total_cases, ref, index, file_name):
    """Print header of case being processed"""
    print(f"\n{'='*60}")
    print(f"Processing case {idx + 1}/{total_cases}")
    print(f"{'='*60}")
    print(f"File: {file_name}")
    print(f"Reference ID: {ref}, Observation index: {index}")


def print_location_info(lon, lat, quality_flag, hs, tp, dp):
    """Print location information and integrated parameters"""
    print(f"Location: ({lon:.2f}°E, {lat:.2f}°N)")
    print(f"Quality flag: {quality_flag}")
    print(f"Integrated spectrum: Hs={hs:.2f}m, Tp={tp:.2f}s, Dp={dp:.0f}°")


def print_partitioning_summary(n_peaks_initial, n_partitions_final):
    """Print summary of process of partitioning"""
    print("\n" + "="*70)
    print(" SPECTRAL PARTITIONING - PROCESS SUMMARY")
    print("="*70)
    print(f"🔍 Spectral peaks initially identified: {n_peaks_initial}")
    print(f"🔗 After merging nearby systems: {n_partitions_final} partition(s)")
    print("="*70)


def print_partitioning_results(results, min_energy_threshold):
    """Print results detailed of partitioning"""
    n_partitions = count_significant_partitions(results, min_energy_threshold)
    
    print("\n" + "="*70)
    print(" PARTITIONING RESULTS")
    print("="*70)
    print(f"Number of partitions found: {n_partitions}")
    print("─"*70)
    
    # Show each partition
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
    
    # Show integrated total
    print("\n" + "─"*70)
    print(f"Integrated total:")
    print(f"  Hs = {results['total_Hs']:.2f} m")
    print(f"  Tp = {results['total_Tp']:.2f} s")
    print(f"  Dp = {results['total_Dp']:.0f}°")
    print("="*70)


def print_save_confirmation(output_path, energy_sum, Hs):
    """Print confirmation of saving"""
    print(f"\n✓ Results saved to: {output_path}")
    print(f"✓ Corrected total: energy={energy_sum:.6f}m², Hs={Hs:.3f}m")


# ============================================================================
# PROCESSAMENTO PRINCIPAL
# ============================================================================

def process_single_case(row, idx, total_cases, output_dir):
    """
    Process a single case SAR
    
    Parameters:
    -----------
    row : pandas.Series
        Row of CSV with information of case
    idx : int
        Index of case current
    total_cases : int
        Number total of cases a process
    output_dir : str
        Directory of output for results
    """
    # Extract parameters
    file_path = os.path.join(SAR_DATA_PATH, row['filename'])
    index = int(row['obs_index'])
    date_time = row['time']
    ref = int(row['ref'])
    
    # Print header
    print_case_header(idx, total_cases, ref, index, os.path.basename(file_path))
    
    # Load data
    ds, E2d, freq, dirs, dirs_rad, selected_time = load_sar_data(file_path, index)
    
    # Calculate integrated parameters
    hs, tp, dp, m0, _, _, _, _ = calculate_wave_parameters(E2d, freq, dirs_rad)
    quality_flag, actual_time, lon, lat = extract_sar_metadata(ds, index)
    
    # Print information
    print_location_info(lon, lat, quality_flag, hs, tp, dp)
    
    # Apply partitioning with SAR-specific parameters
    # SAR has high resolution, so we use:
    # - High threshold (99.5%) to avoid detecting noise peaks
    # - Conservative merge (0.3) to preserve distinct systems
    results = partition_spectrum(
        E2d, freq, dirs_rad,
        threshold_mode='adaptive',
        threshold_percentile=THRESHOLD_PERCENTILE,
        merge_factor=MERGE_FACTOR,
        max_partitions=MAX_PARTITIONS
    )
    
    if results is None:
        print("⚠ No spectral peaks identified!")
        ds.close()
        return
    
    # Calculate threshold and count partitions
    min_energy_threshold = MIN_ENERGY_THRESHOLD_FRACTION * results['total_m0']
    n_peaks_initial = len(results['peaks'])
    n_partitions_final = count_significant_partitions(results, min_energy_threshold)
    
    # Print results
    print_partitioning_summary(n_peaks_initial, n_partitions_final)
    print_partitioning_results(results, min_energy_threshold)
    
    # Calculate corrected Hs
    energy_sum, Hs = calculate_corrected_total_hs(results, min_energy_threshold)
    
    # Create and save results
    data = create_partition_data_dict(
        ref, index, quality_flag, date_time, lon, lat,
        file_path, energy_sum, Hs, results, min_energy_threshold
    )
    output_path, _ = save_partition_results(ref, index, date_time, data, output_dir)
    
    # Print confirmation
    print_save_confirmation(output_path, energy_sum, Hs)
    
    # Close dataset
    ds.close()


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
    print(f"SAR SPECTRAL PARTITIONING PROCESSOR")
    print(f"{'='*60}")
    print(f"Case: {case}")
    print(f"Total cases to process: {total_cases}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Process each case
    for idx, row in df.iterrows():
        try:
            process_single_case(row, idx, total_cases, OUTPUT_DIR)
        except Exception as e:
            print(f"\n❌ Error processing case {idx + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
