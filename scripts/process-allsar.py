"""
SAR Spectral Partitioning Processing Script

This script processes SAR spectral data, applies partitioning algorithm,
and saves results to CSV files.
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
from utils import load_sar_spectrum, calculate_wave_parameters
from partition import partition_spectrum
case = 'surigae'
case = 'lee'
case = 'freddy'
#case = 'all'

# Configuration
OUTPUT_DIR = f'../data/{case}/partition'
SAR_DATA_PATH = f'/Users/jtakeo/data/sentinel1ab/{case}'
CSV_PATH = f'../auxdata/sar_matches_{case}_track_3day.csv'
MIN_ENERGY_THRESHOLD_FRACTION = 0.01  # 1% of total energy
GROUP_NAME = "obs_params"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_corrected_total_hs(results, min_energy_threshold):
    """Calculate corrected total Hs from sum of partition energies"""
    energy_sum = sum(results['energy'][i] for i in range(len(results['energy'])) 
                     if i == 0 or results['energy'][i] > min_energy_threshold)
    Hs = 4 * np.sqrt(energy_sum)
    return energy_sum, Hs


def count_significant_partitions(results, min_energy_threshold):
    """Count partitions with energy above threshold"""
    return sum(1 for i in range(1, len(results['Hs'])) 
               if results['energy'][i] > min_energy_threshold)


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def load_sar_data(file_path, index):
    """Load SAR dataset and extract spectrum"""
    ds = xr.open_dataset(file_path, group=GROUP_NAME)
    E2d, freq, dirs, dirs_rad, selected_time = load_sar_spectrum(ds, date_time=None, index=index)
    return ds, E2d, freq, dirs, dirs_rad, selected_time


def extract_sar_metadata(ds, index):
    """Extract metadata from SAR dataset"""
    quality_flag = ds.L2_partition_quality_flag[index].values
    actual_time = pd.to_datetime(ds.time[index].values)
    lon = ds.longitude[index].values if 'longitude' in ds else None
    lat = ds.latitude[index].values if 'latitude' in ds else None
    return quality_flag, actual_time, lon, lat


def create_partition_data_dict(ref, index, quality_flag, date_time, lon, lat, 
                                file_path, energy_sum, Hs, results, min_energy_threshold):
    """Create data dictionary with partition results"""
    moments = results['moments']
    
    data = {
        'reference_id': ref,
        'obs_index': index,
        'quality_flag': quality_flag,
        'obs_time': date_time,
        'longitude': float(lon),
        'latitude': float(lat),
        'source_file': os.path.basename(file_path),
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
            data.update({
                f'P{p}_energy': 0.0, f'P{p}_Hs': 0.0, f'P{p}_Tp': 0.0,
                f'P{p}_Dp': 0.0, f'P{p}_m0': 0.0, f'P{p}_m1': 0.0, f'P{p}_m2': 0.0,
            })
    
    return data


def save_partition_results(ref, index, date_time, data, output_dir):
    """Save partition results to CSV file"""
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

def print_progress(idx, total, ref, index, file_name):
    """Print progress header"""
    print(f"\n{'='*60}")
    print(f"Processing case {idx + 1}/{total}")
    print(f"{'='*60}")
    print(f"File: {file_name}")
    print(f"Index: {index}, Ref: {ref}")


def print_partitioning_results(results, min_energy_threshold):
    """Print partitioning results"""
    n_partitions = count_significant_partitions(results, min_energy_threshold)
    
    print(f"\n{'='*70}")
    print(" PARTITIONING RESULTS")
    print(f"{'='*70}")
    print(f"Partitions found: {n_partitions}")
    
    for i in range(1, len(results['Hs'])):
        if results['energy'][i] > min_energy_threshold:
            frac = (results['energy'][i] / results['total_m0']) * 100
            print(f"\nPartition {i}:")
            print(f"  Hs={results['Hs'][i]:.2f}m, Tp={results['Tp'][i]:.2f}s, "
                  f"Dp={results['Dp'][i]:.0f}°, Energy={results['energy'][i]:.4f}m² ({frac:.1f}%)")
    
    print(f"\nTotal: Hs={results['total_Hs']:.2f}m, Tp={results['total_Tp']:.2f}s, "
          f"Dp={results['total_Dp']:.0f}°")
    print(f"{'='*70}")


def print_save_info(output_path, energy_sum, Hs):
    """Print save confirmation"""
    print(f"\n✓ Saved: {output_path}")
    print(f"✓ Corrected total_energy={energy_sum:.6f}m², total_Hs={Hs:.3f}m")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_single_case(row, idx, total_cases, output_dir):
    """Process a single SAR case"""
    # Extract parameters
    file_path = os.path.join(SAR_DATA_PATH, row['filename'])
    index = int(row['obs_index'])
    date_time = row['time']
    ref = int(row['ref'])
    
    print_progress(idx, total_cases, ref, index, os.path.basename(file_path))
    
    # Load data
    ds, E2d, freq, dirs, dirs_rad, selected_time = load_sar_data(file_path, index)
    
    # Calculate parameters
    hs, tp, dp, m0, _, _, _, _ = calculate_wave_parameters(E2d, freq, dirs_rad)
    quality_flag, actual_time, lon, lat = extract_sar_metadata(ds, index)
    
    print(f"Location: ({lon:.2f}°E, {lat:.2f}°N), Quality: {quality_flag}")
    print(f"Integrated: Hs={hs:.2f}m, Tp={tp:.2f}s, Dp={dp:.0f}°")
    
    # Apply partitioning
    results = partition_spectrum(E2d, freq, dirs_rad, 0.05, 5)
    
    if results is None:
        print("⚠ No spectral peaks identified!")
        ds.close()
        return
    
    # Process results
    min_energy_threshold = MIN_ENERGY_THRESHOLD_FRACTION * results['total_m0']
    print_partitioning_results(results, min_energy_threshold)
    
    energy_sum, Hs = calculate_corrected_total_hs(results, min_energy_threshold)
    
    # Save results
    data = create_partition_data_dict(ref, index, quality_flag, date_time, lon, lat,
                                      file_path, energy_sum, Hs, results, min_energy_threshold)
    output_path, _ = save_partition_results(ref, index, date_time, data, output_dir)
    
    print_save_info(output_path, energy_sum, Hs)
    ds.close()


def main():
    """Main execution function"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    
    print(f"{'='*60}")
    print("SAR SPECTRAL PARTITIONING PROCESSOR")
    print(f"{'='*60}")
    print(f"Total cases: {len(df)}")
    
    for idx, row in df.iterrows():
        try:
            process_single_case(row, idx, len(df), OUTPUT_DIR)
        except Exception as e:
            print(f"\n❌ Error in case {idx + 1}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("✓ Processing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
