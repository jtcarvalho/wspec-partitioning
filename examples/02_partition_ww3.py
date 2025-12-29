"""
STEP 2: Partition WW3 Spectra

This script processes 2D WW3 model spectra, applies the partitioning algorithm,
and saves the results to CSV.

Workflow:
1. Read coordinates/timestamps from auxiliary file
2. For each point, load WW3 spectrum at closest time or multiple times
3. Apply watershed partitioning algorithm
4. Save Hs, Tp, Dp for each identified partition

Data Type Support (automatically detected):
- SAR/NDBC with timestamp column: Processes closest WW3 time to specified timestamp
  - CSV must have: 'ref'+'sar_time' (SAR) or 'station_id'+'ndbc_time/obs_time' (NDBC)
  - One output file per row in CSV
  - Format: ww3_XXX_YYYYMMDD-HHMMSS.csv
  
- NDBC without timestamp: Processes multiple WW3 times based on TIME_INTERVAL_HOURS
  - CSV must have: 'station_id' (without time column)
  - Multiple output files per station
  - Format: ww3_STATION_YYYYMMDD-HHMMSS.csv
  
The script automatically detects the data type and processing mode based on CSV columns.
"""

import os
import xarray as xr
import pandas as pd
import numpy as np

# Import from wasp package
from wasp.io_ww3 import find_closest_time, load_ww3_spectrum
from wasp.wave_params import calculate_wave_parameters
from wasp.partition import partition_spectrum

# ============================================================================
# CONFIGURATION
# ============================================================================

#case = 'lee'
# case = 'freddy'
# case = 'surigae'
case = 'all'

# Directories
OUTPUT_DIR = f'../data/{case}/partition-ndbc'

#NDBC options
# CSV_PATH = f'../auxdata/ndbc_ww3_matches_{case}.csv'
# WW3_DATA_PATH = f'/work/cmcc/jc11022/simulations/uGlobWW3/highResExperiments/exp_02-st4-uost-psi-400s-era5-b143-ic5-noref/2020/ww3-ndbc/'

#SAR options
# CSV_PATH = f'../auxdata/sar_ww3_matches_{case}.csv'
# WW3_DATA_PATH = f'/work/cmcc/jc11022/simulations/uGlobWW3/highResExperiments/exp_02-st4-uost-psi-400s-era5-b143-ic5-noref/2020/sar-spec/'


# OUTPUT_DIR = f'../data/{case}/partition'
CSV_PATH = f'../auxdata/ndbc_ww3_matches_{case}.csv'
WW3_DATA_PATH = f'/User/data/ww3/ndbc/'



# Partitioning parameters
MIN_ENERGY_THRESHOLD_FRACTION = 0.01  # 1% of the energy total (post-processing filter)

# Parameters for partition_spectrum() function
THRESHOLD_PERCENTILE = 99.0  # WW3: Conservative threshold for model data
MERGE_FACTOR = 0.6           # WW3: Merge nearby systems more aggressively

# Time sampling for NDBC (hours between outputs)
# Options: 1 (hourly), 3 (3-hourly), 6 (6-hourly), 12 (12-hourly), 24 (daily)
TIME_INTERVAL_HOURS = 6  # Process data every N hours

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_data_type(df):
    """
    Detect if CSV is for SAR or NDBC based on column names
    
    Returns:
    --------
    str: 'sar' or 'ndbc'
    """
    if 'station_id' in df.columns:
        return 'ndbc'
    elif 'ref' in df.columns:
        return 'sar'
    else:
        raise ValueError("Cannot determine data type. Expected 'station_id' (NDBC) or 'ref' (SAR) column.")


def count_significant_partitions(results, min_energy_threshold):
    """Count partitions with energy above of threshold"""
    count = 0
    for i in range(1, len(results['Hs'])):
        if results['energy'][i] > min_energy_threshold:
            count += 1
    return count


# ============================================================================
# PRINTING FUNCTIONS
# ============================================================================

def print_case_header(idx, total_cases, ref_id, target_time_dt, data_type='sar'):
    """Print header of case being processed"""
    print(f"\n{'='*60}")
    print(f"Processing case {idx + 1}/{total_cases}")
    print(f"{'='*60}")
    if data_type == 'ndbc':
        print(f"Station ID: {ref_id}")
        print(f"Target time: {target_time_dt}")
    else:
        print(f"Reference ID: {ref_id}")
        print(f"Target SAR time: {target_time_dt}")


def print_time_match_info(closest_time, itime, time_diff_hours):
    """Print information of matching temporal"""
    print(f"Closest WW3 time: {closest_time}")
    print(f"Time index (itime): {itime}")
    print(f"Time difference: {time_diff_hours:.2f} hours")


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


def print_save_confirmation(output_path, df_results):
    """Print confirmation of saving"""
    print(f"\n✓ Results saved to: {output_path}")
    print(f"\nColumns in CSV: {list(df_results.columns)}")
    print(f"\nPreview:")
    print(df_results.T)


# ============================================================================
# PROCESSAMENTO DE DADOS
# ============================================================================

def create_partition_data_dict(ref, selected_time, lon, lat, file_path, 
                                results, min_energy_threshold):
    """
    Create dictionary with results of partitioning
    
    Returns:
    --------
    dict: Dictionary ready for conversion in DataFrame
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
        
        # Spectrum total
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


def save_partition_results(ref_id, selected_time, data, output_dir, data_type='sar'):
    """
    Save results of partitioning in file CSV
    
    Returns:
    --------
    tuple: (output_path, df_results)
    """
    # Create nome of file
    date_time_formatted = selected_time.strftime('%Y%m%d-%H%M%S')
    
    if data_type == 'ndbc':
        # For NDBC: use station_id as string
        output_filename = f'ww3_{ref_id}_{date_time_formatted}.csv'
    else:
        # For SAR: use reference_id as integer with padding
        output_filename = f'ww3_{ref_id:03d}_{date_time_formatted}.csv'
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Create DataFrame e save
    df_results = pd.DataFrame([data])
    df_results.to_csv(output_path, index=False, float_format='%.6f')
    
    return output_path, df_results


def process_single_case(row, idx, total_cases, output_dir, data_type='sar'):
    """
    Process a single case WW3
    
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
    data_type : str
        Type of data: 'sar' or 'ndbc'
    """
    # Extract information of case based on data type
    if data_type == 'ndbc':
        ref_id = str(row['station_id'])
        # Check if CSV has time column (ndbc_time, ww3_time, obs_time, etc.)
        time_col = None
        for col in ['ndbc_time', 'ww3_time', 'obs_time', 'time']:
            if col in row.index and pd.notna(row[col]):
                time_col = col
                break
        
        if time_col:
            target_time_dt = pd.to_datetime(row[time_col])
        else:
            target_time_dt = None  # Will process multiple times
        
        file_path = f"{WW3_DATA_PATH}/ww3_{ref_id}.nc"
    else:
        ref_id = int(row['ref'])
        target_time_str = row['sar_time']
        target_time_dt = pd.to_datetime(target_time_str)
        file_path = f'{WW3_DATA_PATH}/ww3_sar{ref_id:04d}_2020_spec.nc'

    # Print header
    print_case_header(idx, total_cases, ref_id, target_time_dt if target_time_dt else "Multiple times", data_type)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"⚠ WW3 file not found: {file_path}")
        return
    
    # If target_time_dt is None, process multiple times based on TIME_INTERVAL_HOURS
    # Otherwise, find and process the closest time
    if target_time_dt is None:
        # Open file to get all times
        with xr.open_dataset(file_path) as ds:
            times = pd.to_datetime(ds.time.values)
            
            # Filter times based on configured interval
            # Keep times where hour is divisible by TIME_INTERVAL_HOURS
            filtered_times = [t for t in times if t.hour % TIME_INTERVAL_HOURS == 0]
            filtered_indices = [i for i, t in enumerate(times) if t.hour % TIME_INTERVAL_HOURS == 0]
            
            interval_str = f"{TIME_INTERVAL_HOURS}-hourly" if TIME_INTERVAL_HOURS > 1 else "hourly"
            print(f"Processing {len(filtered_times)} time steps ({interval_str}) for {data_type.upper()} {ref_id}")
            print(f"Total available: {len(times)} (sampling every {TIME_INTERVAL_HOURS}h)")
            
            # Process each filtered time
            for idx, (itime, obs_time) in enumerate(zip(filtered_indices, filtered_times)):
                print(f"\n  Time {idx+1}/{len(filtered_times)}: {obs_time}")
                
                # Load spectrum for this time
                E2d, freq, dirs, dirs_rad, lon, lat = load_ww3_spectrum(file_path, itime)
                
                # Process this time step
                process_time_step(ref_id, obs_time, E2d, freq, dirs, dirs_rad, 
                                lon, lat, file_path, output_dir, data_type)
        return
    else:
        # Find closest time to target
        itime, closest_time, time_diff_hours = find_closest_time(file_path, target_time_dt)
        print_time_match_info(closest_time, itime, time_diff_hours)
        
        # Load spectrum
        E2d, freq, dirs, dirs_rad, lon, lat = load_ww3_spectrum(file_path, itime)
        
        # Process single time step
        process_time_step(ref_id, closest_time, E2d, freq, dirs, dirs_rad,
                        lon, lat, file_path, output_dir, data_type)


def process_time_step(ref_id, selected_time, E2d, freq, dirs, dirs_rad,
                     lon, lat, file_path, output_dir, data_type='sar'):
    """
    Process a single time step (common for SAR and NDBC)
    """
    
    # Apply partitioning with WW3-specific parameters
    # WW3 has moderate resolution, so we use:
    # - Conservative threshold (99.0%) to avoid detecting noise peaks
    # - Higher merge factor (0.6) to merge nearby systems more aggressively
    results = partition_spectrum(
        E2d, freq, dirs_rad,
        threshold_mode='adaptive',
        threshold_percentile=THRESHOLD_PERCENTILE,
        merge_factor=MERGE_FACTOR,
        max_partitions=5
    )
    
    if results is None:
        print("    ⚠ No spectral peaks identified!")
        return
    
    # Calculate threshold and count partitions
    min_energy_threshold = MIN_ENERGY_THRESHOLD_FRACTION * results['total_m0']
    n_peaks_initial = len(results['peaks'])
    n_partitions_final = count_significant_partitions(results, min_energy_threshold)
    
    # Print results (condensed for NDBC multi-time processing)
    if data_type == 'ndbc':
        print(f"    Peaks: {n_peaks_initial} → Partitions: {n_partitions_final} | "
              f"Hs={results['total_Hs']:.2f}m Tp={results['total_Tp']:.1f}s")
    else:
        print_partitioning_summary(n_peaks_initial, n_partitions_final)
        print_partitioning_results(results, min_energy_threshold)
    
    # Create and save results
    data = create_partition_data_dict(ref_id, selected_time, lon, lat, file_path,
                                      results, min_energy_threshold)
    output_path, df_results = save_partition_results(ref_id, selected_time, data, output_dir, data_type)
    
    # Print confirmation (condensed for NDBC)
    if data_type == 'sar':
        print_save_confirmation(output_path, df_results)
    else:
        print(f"    ✓ Saved: {os.path.basename(output_path)}")


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
    
    # Detect data type (SAR or NDBC)
    data_type = detect_data_type(df)
    
    print(f"{'='*60}")
    print(f"WW3 SPECTRAL PARTITIONING PROCESSOR")
    print(f"{'='*60}")
    print(f"Data type: {data_type.upper()}")
    print(f"Total cases to process: {total_cases}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"CSV file: {CSV_PATH}")
    
    # Process each case
    for idx, row in df.iterrows():
        try:
            process_single_case(row, idx, total_cases, OUTPUT_DIR, data_type)
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
