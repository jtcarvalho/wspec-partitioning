"""
NDBC Buoy Spectral Partitioning

Process NDBC buoy spectral data from SAR-NDBC matches and apply partitioning.
Use 00_create_matches.py first to create the matches CSV file.
"""

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import from wasp package
from wasp.wave_params import calculate_wave_parameters
from wasp.partition import partition_spectrum
from wasp.io_ndbc import find_closest_time, load_ndbc_spectrum


# ============================================================================
# CONFIGURATION
# ============================================================================

CASE_NAME = 'all'  # Must match 00_create_matches.py

# Directories
NDBC_BASE_DIR = '/Users/jtakeo/data/ndbc-all'  # Base directory with station folders
OUTPUT_DIR = f'../data/{CASE_NAME}/partition'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV with SAR-NDBC matches (created by 00_create_matches.py)
MATCHES_CSV = f'../auxdata/sar_ndbc_ww3_matches_{CASE_NAME}.csv'

# Partitioning parameters
MAX_PARTITIONS = 5                # Maximum number of partitions
MIN_ENERGY_FRACTION = 0.01        # Minimum energy fraction (1% of total)
MAX_TIME_DIFF_HOURS = 3.0         # Maximum time difference for matching

# Parameters for partition_spectrum() function
THRESHOLD_PERCENTILE = 95.0       # NDBC: Lower threshold for low directional resolution
MERGE_FACTOR = 0.5                # NDBC: Moderate merging to combine MEM artifacts


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_ndbc_match(match_row):
    """
    Process a single SAR-NDBC match and apply partitioning.
    
    Parameters:
    -----------
    match_row : pd.Series
        Row from matches CSV with match information
    
    Returns:
    --------
    list of dict
        List of partition results with wave parameters
    """
    station_id = str(match_row['station_id'])
    sar_time = pd.to_datetime(match_row['sar_time'])
    ndbc_file = match_row['ndbc_file']
    
    print(f"\n{'='*70}")
    print(f"Station: {station_id}")
    print(f"SAR time: {sar_time}")
    print(f"NDBC file: {ndbc_file}")
    print(f"{'='*70}")
    
    # Construct full path to NDBC file
    ndbc_full_path = Path(NDBC_BASE_DIR) / station_id / ndbc_file
    
    if not ndbc_full_path.exists():
        print(f"  ⚠ File not found: {ndbc_full_path}")
        return []
    
    try:
        # Open NDBC dataset
        ds = xr.open_dataset(ndbc_full_path)
        
        # Find closest time
        itime, closest_time, time_diff_hours = find_closest_time(ds, sar_time)
        
        if itime is None:
            print(f"  ⚠ Could not find closest time")
            ds.close()
            return []
        
        print(f"NDBC time: {closest_time}")
        print(f"Time difference: {time_diff_hours:.2f} hours")
        
        # Load spectrum
        spectrum_result = load_ndbc_spectrum(ds, itime)
        ds.close()
        
        if spectrum_result is None:
            print(f"  ⚠ Could not load spectrum")
            return []
        
        E2d, freq, dirs, dirs_rad, lon, lat = spectrum_result
        
        print(f"Spectrum: {E2d.shape[0]} freq x {E2d.shape[1]} dir")
        print(f"Freq range: {freq.min():.3f} - {freq.max():.3f} Hz")
        print(f"Dir range: {dirs.min():.1f} - {dirs.max():.1f}°")
        
        # Apply partitioning with NDBC-specific parameters
        # NDBC has low directional resolution (MEM reconstruction), so we use:
        # - Lower threshold (95%) to capture multiple peaks
        # - Moderate merge (0.5) to combine MEM artifacts without over-merging
        results = partition_spectrum(
            E2d, freq, dirs_rad,
            threshold_mode='adaptive',
            threshold_percentile=THRESHOLD_PERCENTILE,
            merge_factor=MERGE_FACTOR,
            max_partitions=MAX_PARTITIONS
        )
        
        if results is None:
            print("  ⚠ No spectral peaks identified")
            return []
        
        # Calculate threshold
        min_energy_threshold = MIN_ENERGY_FRACTION * results['total_m0']
        
        print(f"\nTotal energy: {results['total_m0']:.4f} m²")
        print(f"Total Hs: {results['total_Hs']:.2f} m")
        print(f"Spectral peaks found: {len(results['peaks'])}")
        
        # Process significant partitions
        partition_results = []
        partition_count = 0
        
        for i in range(1, len(results['Hs'])):
            if results['energy'][i] > min_energy_threshold:
                partition_count += 1
                energy_fraction = (results['energy'][i] / results['total_m0']) * 100
                
                result = {
                    'ndbc_time': closest_time,
                    'station_id': station_id,
                    'station_lon': lon,
                    'station_lat': lat,
                    'partition': partition_count,
                    'Hs': results['Hs'][i],
                    'Tp': results['Tp'][i],
                    'Dp': results['Dp'][i],
                    'energy': results['energy'][i],
                    'energy_fraction': energy_fraction,
                    'ndbc_file': ndbc_file,
                }
                partition_results.append(result)
                
                print(f"\n  Partition {partition_count}:")
                print(f"    Hs = {results['Hs'][i]:.2f} m")
                print(f"    Tp = {results['Tp'][i]:.2f} s")
                print(f"    Dp = {results['Dp'][i]:.0f}°")
                print(f"    Energy fraction: {energy_fraction:.1f}%")
        
        print(f"\n{'='*70}")
        print(f"✓ {partition_count} partitions above threshold")
        print(f"{'='*70}")
        
        return partition_results
        
    except Exception as e:
        print(f"  ✖ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Process all SAR-NDBC matches and save partitioned spectra results.
    """
    print("="*70)
    print("NDBC SPECTRA PARTITIONING")
    print("="*70)
    print(f"Case: {CASE_NAME}")
    print(f"NDBC directory: {NDBC_BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Matches CSV: {MATCHES_CSV}")
    print("="*70)
    
    # Check if matches CSV exists
    if not os.path.exists(MATCHES_CSV):
        print(f"\n⚠️  File not found: {MATCHES_CSV}")
        print("\nPlease run 00_create_matches.py first to create the matches CSV.")
        return
    
    # Load matches
    df_matches = pd.read_csv(MATCHES_CSV)
    print(f"\n✓ Loaded {len(df_matches)} SAR-NDBC matches")
    print(f"  Unique stations: {df_matches['station_id'].nunique()}")
    print(f"  Matches with WW3: {df_matches.get('ww3_available', pd.Series([False])).sum()}")
    
    # Process each match
    all_results = []
    success_count = 0
    failed_count = 0
    
    for idx, row in df_matches.iterrows():
        print(f"\n[{idx+1}/{len(df_matches)}]")
        
        try:
            results = process_ndbc_match(row)
            
            # Add match metadata to each partition result
            for result in results:
                result['match_id'] = idx + 1
                result['sar_file'] = row['sar_file']
                result['sar_index'] = row['sar_index']
                result['sar_lon'] = row['sar_lon']
                result['sar_lat'] = row['sar_lat']
                result['sar_time'] = row['sar_time']
                result['distance_km'] = row['distance_km']
                result['sar_ndbc_time_diff_hours'] = row['time_diff_hours']
                
                if 'ww3_file' in row and pd.notna(row['ww3_file']):
                    result['ww3_file'] = row['ww3_file']
                    result['ww3_available'] = row.get('ww3_available', False)
                else:
                    result['ww3_available'] = False
            
            all_results.extend(results)
            
            if len(results) > 0:
                success_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"  ✖ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    # Summary and save results
    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Successful: {success_count}/{len(df_matches)}")
    print(f"✗ Failed: {failed_count}/{len(df_matches)}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = os.path.join(OUTPUT_DIR, 'ndbc_partitions.csv')
        df.to_csv(output_file, index=False, float_format='%.6f')
        
        print(f"\n✓ Results saved to: {output_file}")
        print(f"  Total partitions: {len(all_results)}")
        print(f"  Unique stations: {df['station_id'].nunique()}")
        print(f"  Partitions with WW3: {df.get('ww3_available', pd.Series([False])).sum()}")
        print(f"{'='*70}")
        
        # Show preview
        print("\nPreview (first 5 rows):")
        preview_cols = ['match_id', 'station_id', 'partition', 'Hs', 'Tp', 'Dp', 
                       'distance_km', 'ww3_available']
        available_cols = [c for c in preview_cols if c in df.columns]
        print(df[available_cols].head(5).to_string(index=False))
        
        # Statistics
        print(f"\nPartition statistics:")
        print(f"  Hs range: {df['Hs'].min():.2f} - {df['Hs'].max():.2f} m")
        print(f"  Tp range: {df['Tp'].min():.2f} - {df['Tp'].max():.2f} s")
        print(f"  Distance range: {df['distance_km'].min():.1f} - {df['distance_km'].max():.1f} km")
    else:
        print("\n⚠️  No results to save")
        print("   Check your configuration and data availability.")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
