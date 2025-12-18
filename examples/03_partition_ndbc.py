"""
Template for Partitioning NDBC Buoy Spectra

This is an initial template for you to adapt to your buoy data.
Adjust as needed for the specific format of your files.
"""

import os
import pandas as pd
import numpy as np

# Import from wasp package
from wasp.wave_params import calculate_wave_parameters
from wasp.partition import partition_spectrum


# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
INPUT_DIR = '/Users/jtakeo/data/ndbc' # Adjust to your NDBC data directory
OUTPUT_DIR = '../output/ndbc'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Partitioning parameters
ENERGY_THRESHOLD = 1e-6    # Adjust based on typical energy of your buoy
MAX_PARTITIONS = 5
MIN_PARTITION_POINTS = 5


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_ndbc_spectrum(filepath):
    """
    Load NDBC buoy spectrum.
    
    YOU NEED TO IMPLEMENT THIS FUNCTION based on your data format.
    
    Returns
    -------
    E : ndarray (NF, ND)
        2D spectrum in m²/Hz/rad
    freq : ndarray (NF,)
        Frequencies in Hz
    dirs : ndarray (ND,)
        Directions in degrees (oceanographic convention - going to)
    metadata : dict
        Additional information (timestamp, lat, lon, etc.)
    """
    
    # EXAMPLE - ADJUST FOR YOUR FORMAT
    # Option 1: If your data is in CSV
    # df = pd.read_csv(filepath)
    # E = df.values.reshape(NF, ND)
    
    # Option 2: If your data is in NetCDF
    # import xarray as xr
    # ds = xr.open_dataset(filepath)
    # E = ds['energy'].values  # [freq x dir]
    # freq = ds['frequency'].values
    # dirs = ds['direction'].values
    
    # IMPORTANT: Check units and conventions
    # - Energy must be in m²/Hz/rad
    # - If in m²/Hz/deg, multiply by π/180
    # - Direction must be oceanographic (going to)
    # - If meteorological (coming from), convert:
    #   from wasp.utils import convert_meteorological_to_oceanographic
    #   dirs = convert_meteorological_to_oceanographic(dirs)
    
    raise NotImplementedError(
        "You need to implement the load_ndbc_spectrum() function "
        "based on your buoy data format."
    )
    
    # return E, freq, dirs, metadata


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_ndbc_station(station_file):
    """
    Process a buoy file and return partitions.
    """
    print(f"\nProcessing: {station_file}")
    
    # 1. Load spectrum
    E, freq, dirs, metadata = load_ndbc_spectrum(station_file)
    
    print(f"  Spectrum: {E.shape[0]} freq x {E.shape[1]} dir")
    print(f"  Freq range: {freq.min():.3f} - {freq.max():.3f} Hz")
    print(f"  Dir range: {dirs.min():.1f} - {dirs.max():.1f} deg")
    print(f"  Total energy: {np.sum(E):.2e} m²")
    
    # 2. Apply partitioning
    partitions = partition_spectrum(
        E, freq, dirs,
        energy_threshold=ENERGY_THRESHOLD,
        max_partitions=MAX_PARTITIONS,
        min_partition_points=MIN_PARTITION_POINTS
    )
    
    print(f"  → {len(partitions)} partitions identified")
    
    # 3. Calculate parameters for each partition
    results = []
    
    for i, partition in enumerate(partitions):
        params = calculate_wave_parameters(partition, freq, dirs)
        
        result = {
            'timestamp': metadata.get('timestamp', ''),
            'station_id': metadata.get('station_id', ''),
            'lat': metadata.get('lat', np.nan),
            'lon': metadata.get('lon', np.nan),
            'partition': i + 1,
            'Hs': params['Hs'],
            'Tp': params['Tp'],
            'Dp': params['Dp'],
            'fp': params['fp'],
            'Tm': params['Tm'],
            'E_total': params['E'],
        }
        results.append(result)
        
        print(f"    P{i+1}: Hs={params['Hs']:.2f}m, Tp={params['Tp']:.1f}s, Dp={params['Dp']:.0f}°")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Process all buoy files in the input directory.
    """
    print("="*70)
    print("NDBC SPECTRA PARTITIONING")
    print("="*70)
    
    # List input files
    # ADJUST the search pattern according to your files
    import glob
    buoy_files = glob.glob(os.path.join(INPUT_DIR, '*.nc'))  # or *.csv, *.txt, etc.
    
    if not buoy_files:
        print(f"\n⚠️  No files found in {INPUT_DIR}")
        print("   Adjust INPUT_DIR and search pattern in the code.")
        return
    
    print(f"\nFound {len(buoy_files)} files")
    
    # Process each file
    all_results = []
    
    for buoy_file in buoy_files:
        try:
            results = process_ndbc_station(buoy_file)
            all_results.extend(results)
        except Exception as e:
            print(f"  ❌ Error processing {buoy_file}: {e}")
            continue
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = os.path.join(OUTPUT_DIR, 'ndbc_partitions.csv')
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        print(f"  Total of {len(all_results)} partitions processed")
    else:
        print("\n⚠️  No results to save")


if __name__ == '__main__':
    main()
