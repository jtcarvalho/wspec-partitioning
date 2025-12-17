"""
WASP - Wave Analysis from Sentinel and WaveWatch III Partitioning
==================================================================

A Python package for spectral wave partitioning from SAR, WaveWatch III, and buoy data.

Core modules:
- partition: Spectral partitioning algorithms (Hanson & Phillips 2001)
- wave_params: Wave parameter calculations (Hs, Tp, Dir, etc.)
- io_sar: SAR data input/output
- io_ww3: WaveWatch III data input/output
- plotting: Visualization tools
- utils: Utility functions

Example usage:
-------------
    from wasp.partition import partition_spectrum
    from wasp.wave_params import calculate_wave_parameters
    
    # Partition a 2D spectrum
    partitions = partition_spectrum(
        E, freq, dirs, 
        energy_threshold=1e-6,
        max_partitions=3
    )
    
    # Calculate wave parameters for each partition
    for i, partition in enumerate(partitions):
        params = calculate_wave_parameters(partition, freq, dirs)
        print(f"Partition {i+1}: Hs={params['Hs']:.2f}m, Tp={params['Tp']:.1f}s")
"""

__version__ = "0.1.1"
__author__ = "JT Carvalho"

# Import main functions for easy access
from .partition import partition_spectrum
from .wave_params import calculate_wave_parameters
from .utils import spectrum1d_from_2d

__all__ = [
    'partition_spectrum',
    'calculate_wave_parameters',
    'spectrum1d_from_2d',
]
