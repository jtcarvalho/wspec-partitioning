import numpy as np
from collections import Counter
from .utils import calculate_wave_parameters, spectrum1d_from_2d

def identify_spectral_peaks(E, NF, ND, energy_threshold, max_partitions):
    """
    Identify spectral peaks in the 2D energy spectrum using a 3x3 neighborhood analysis.
    
    This function implements the Hanson & Phillips (2001) peak identification algorithm.
    For each point in the spectrum, it examines the 3x3 neighborhood to determine if the
    point is a local maximum. The algorithm assigns direction codes (ICOD) that indicate
    the direction of the steepest descent from each point.
    
    Parameters
    ----------
    E : ndarray (NF, ND)
        2D spectral energy density matrix [frequency x direction]
    NF : int
        Number of frequency bins
    ND : int
        Number of directional bins
    energy_threshold : float
        Minimum energy threshold for peak identification
    max_partitions : int
        Maximum number of partitions/peaks to identify
    
    Returns
    -------
    ICOD : ndarray (NF, ND)
        Direction code matrix indicating steepest descent direction for each point
        Code format: JY*10 + IX, where IX,JY ∈ {1,2,3} indicate relative position
        Code 22 indicates a local maximum (peak)
    MASK : ndarray (NF, ND)
        Initial mask with peaks marked (1 to nmask), zeros elsewhere
    peaks : ndarray (nmask, 2)
        Array of peak locations [frequency_index, direction_index] (1-based indexing)
    nmask : int
        Number of peaks identified
    
    Notes
    -----
    - Peaks are sorted by energy (highest first) and limited to max_partitions
    - Direction dimension is treated as periodic (wraps around)
    """
    print(f"Identifying spectral peaks with threshold: {energy_threshold:.2e}")
    print(f"Spectrum values: min={np.min(E):.2e}, max={np.max(E):.2e}, mean={np.mean(E):.2e}")

    ICOD = np.zeros((NF, ND), dtype=int)
    peaks_list = []
    
    # For each point in the spectrum
    for II in range(NF):
        for JJ in range(ND):
            if E[II, JJ] < 1e-15:  # Very small value to ignore
                continue
                
            # Check 3x3 neighborhood
            RMAX = 0
            IX = 2
            JY = 2
            i_range = [max(0, II-1), II, min(NF-1, II+1)]
            j_range = [(JJ-1) % ND, JJ, (JJ+1) % ND]
            
            # Find the largest neighbor
            for i_idx, I in enumerate(i_range):
                for j_idx, J in enumerate(j_range):
                    if i_idx == 1 and j_idx == 1:  # Skip the point itself
                        continue
                    RT = E[I, J] - E[II, JJ]
                    if RT > RMAX:
                        IX = i_idx + 1
                        JY = j_idx + 1
                        RMAX = RT
            
            # Assign direction code
            ICOD[II, JJ] = JY * 10 + IX
            
            # Consider as peak only if energy >= threshold
            if ICOD[II, JJ] == 22 and E[II, JJ] >= energy_threshold:
                peaks_list.append((II+1, JJ+1, E[II, JJ]))  # Add energy value as third column
    
    # Sort peaks by energy (highest to lowest)
    peaks_list.sort(key=lambda x: x[2], reverse=True)
    
    # Limit number of peaks to max_partitions
    if len(peaks_list) > max_partitions:
        print(f"Limiting number of peaks from {len(peaks_list)} to {max_partitions}")
        peaks_list = peaks_list[:max_partitions]
    
    # Remove energy from final peak list
    peaks_list = [(p[0], p[1]) for p in peaks_list]
    
    # Create initial mask with peaks
    nmask = len(peaks_list)
    peaks = np.array(peaks_list) if nmask > 0 else np.empty((0, 2))
    MASK = np.zeros((NF, ND), dtype=int)
    
    for im in range(nmask):
        ii = int(peaks[im, 0]) - 1
        jj = int(peaks[im, 1]) - 1
        MASK[ii, jj] = im + 1
    
    print(f"Identified {nmask} spectral peaks")
    
    return ICOD, MASK, peaks, nmask


def generate_mask(ICOD, MASK, NF, ND):
    """
    Generate partition mask by propagating peak labels using ICOD direction codes.
    
    This function implements the watershed algorithm to assign each spectrum point to
    a partition. It uses the ICOD direction codes (computed in identify_spectral_peaks)
    to trace paths from each point to the nearest peak. The algorithm operates in two
    phases: (1) propagation using ICOD directions, and (2) filling remaining zeros
    using nearest neighbor voting.
    
    Parameters
    ----------
    ICOD : ndarray (NF, ND)
        Direction code matrix indicating steepest descent direction
        Format: JY*10 + IX, where IX,JY encode relative position to neighbor
    MASK : ndarray (NF, ND)
        Initial mask with peaks labeled (1 to nmask), zeros elsewhere
    NF : int
        Number of frequency bins
    ND : int
        Number of directional bins
    
    Returns
    -------
    mask_copy : ndarray (NF, ND)
        Complete partition mask with all points assigned to a partition
        Values: 1 to nmask (partition labels)
    
    Algorithm
    ---------
    Phase 1 (ICOD propagation):
        - Iterate 5 times in forward and backward directions (frequency and direction)
        - For each point, follow ICOD code to find which partition it belongs to
        - Propagate partition labels from peaks to surrounding points
    
    Phase 2 (Fill remaining zeros):
        - For any remaining unassigned points (value = 0)
        - Use 8-neighbor voting to assign most common neighbor label
        - Iterate until in the zeros remain or in the changes occur
        - Assign default value (1) if in the neighbors are labeled
    
    Notes
    -----
    - Direction dimension is treated as periodic (wraps around at 0/360°)
    - Frequency dimension has hard boundaries (in the wrapping)
    - Multiple passes ensure all points are assigned even in complex spectra
    """
    mask_copy = MASK.copy()

    print("Generating mask from ICOD...")
    # First pass - propagate values using ICOD directions
    for _ in range(5):
        i_ranges = [(0, NF, 1), (NF-1, -1, -1)]
        for i_start, i_end, i_step in i_ranges:
            j_ranges = [(0, ND, 1), (ND-1, -1, -1)]
            for j_start, j_end, j_step in j_ranges:
                for i in range(i_start, i_end, i_step):
                    for j in range(j_start, j_end, j_step):
                        code = ICOD[i, j]
                        j_dir = (code // 10) - 2 + j
                        i_dir = (code % 10) - 2 + i
                        if i_dir < 0: i_dir = 0
                        elif i_dir >= NF: i_dir = NF - 1
                        if j_dir < 0: j_dir = ND - 1
                        elif j_dir >= ND: j_dir = 0
                        mask_copy[i, j] = mask_copy[i_dir, j_dir]
    
    # Second pass - handle remaining zeros
    while 0 in mask_copy:
        zero_changed = False
        for i in range(NF):
            for j in range(ND):
                if mask_copy[i, j] == 0:
                    # Check 8 neighboring cells
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni = i + di
                            nj = (j + dj) % ND  # Handle circular direction
                            if 0 <= ni < NF:
                                if mask_copy[ni, nj] > 0:
                                    neighbors.append(mask_copy[ni, nj])
                    
                    if neighbors:
                        # Assign most common neighbor value
                        most_common = Counter(neighbors).most_common(1)[0][0]
                        mask_copy[i, j] = most_common
                        zero_changed = True
        
        if not zero_changed:
            # If in the zeros were changed, assign default value
            for i in range(NF):
                for j in range(ND):
                    if mask_copy[i, j] == 0:
                        mask_copy[i, j] = 1  # Default value
            break
            
    return mask_copy


def calculate_peak_distances(peaks, frequencies, directions_rad, nmask):
    """
    Calculate squared Euclidean distances between spectral peaks in frequency-direction space.
    
    This function computes pairwise distances between all identified peaks using their
    positions in the 2D frequency-direction spectrum. Peaks are converted from spectral
    coordinates (frequency, direction) to Cartesian coordinates (x, y) where:
        x = frequency * cos(direction)
        y = frequency * sin(direction)
    
    The distance metric is used later to determine which partitions should be merged
    based on their proximity in spectral space.
    
    Parameters
    ----------
    peaks : ndarray (nmask, 2)
        Array of peak locations with 1-based indices [frequency_index, direction_index]
    frequencies : ndarray (NF,)
        Frequency array in Hz
    directions_rad : ndarray (ND,)
        Direction array in radians (oceanographic convention)
    nmask : int
        Number of peaks
    
    Returns
    -------
    distances : ndarray (nmask, nmask)
        Symmetric matrix of squared distances between peaks
        distances[i,j] = ||peak_i - peak_j||²
    
    Notes
    -----
    - Returns squared distances (in the square root) for computational efficiency
    - Distance matrix is symmetric with zeros on diagonal
    - Used in merge_overlapping_systems to identify peaks that should be combined
    """
    print("Calculating distances between peaks...")
    if nmask == 0:
        return np.zeros((0, 0))
    
    # Convert from 1-based to 0-based indices
    i_indices = (peaks[:, 0] - 1).astype(int)
    j_indices = (peaks[:, 1] - 1).astype(int)
    
    # Get frequencies and directions for all peaks
    freqs = frequencies[i_indices]
    dirs = directions_rad[j_indices]
    
    # Calculate x, y coordinates for all peaks
    x_coords = freqs[:, np.newaxis] * np.cos(dirs)[:, np.newaxis]
    y_coords = freqs[:, np.newaxis] * np.sin(dirs)[:, np.newaxis]
    
    # Calculate squared distances
    dx = x_coords - x_coords.T
    dy = y_coords - y_coords.T
    
    return dx**2 + dy**2

def calculate_peak_spreading(E, MASK, frequencies, directions_rad, NF, ND, nmask, Etot, delf, ddir):
    """
    Calculate individual peak spreading parameter for each partition.
    
    This function computes the spectral spreading (variance) of each partition in
    frequency-direction space. The spreading parameter Eip is used as a measure of
    how concentrated or diffuse each wave system is, which helps determine appropriate
    merging thresholds in merge_overlapping_systems.
    
    The spreading is calculated as the variance in Cartesian coordinates:
        Eip[i] = Var(x) + Var(y)
    where x = f*cos(θ), y = f*sin(θ) for partition i
    
    Parameters
    ----------
    E : ndarray (NF, ND)
        2D spectral energy density matrix
    MASK : ndarray (NF, ND)
        Partition mask (1 to nmask for partitions, 0 for unclassified)
    frequencies : ndarray (NF,)
        Frequency array in Hz
    directions_rad : ndarray (ND,)
        Direction array in radians
    NF : int
        Number of frequency bins
    ND : int
        Number of directional bins
    nmask : int
        Number of partitions
    Etot : float
        Total spectral energy (m0)
    delf : ndarray (NF,)
        Frequency bin widths
    ddir : float
        Directional bin width in radians
    
    Returns
    -------
    Eip : ndarray (nmask,)
        Spreading parameter for each partition
        Eip[i] = variance of partition i in frequency-direction space
    
    Notes
    -----
    - Spreading is computed using weighted moments: fx, fy (first), fxx, fyy (second)
    - Variance formula: Var(x) = E[x²] - E[x]² applied to both x and y components
    - Normalized by total energy Etot for dimensionless measure
    - Smaller Eip indicates more concentrated (narrower) partition
    """
    print("Calculating peak spreading...")
    if nmask == 0:
        return np.zeros(0)
    
    # Create frequency and direction meshgrids
    freq_grid, dir_grid = np.meshgrid(frequencies, directions_rad, indexing='ij')
    
    # Pre-compute common terms
    cos_dir = np.cos(dir_grid)
    sin_dir = np.sin(dir_grid)
    freq_cos = freq_grid * cos_dir
    freq_sin = freq_grid * sin_dir
    freq2_cos2 = freq_grid**2 * cos_dir**2
    freq2_sin2 = freq_grid**2 * sin_dir**2
    
    # Create weight grid
    delf_grid = delf[:, np.newaxis] * np.ones(ND)
    weights = E * delf_grid * ddir
    
    # Initialize arrays
    fx = np.zeros(nmask + 2)
    fy = np.zeros(nmask + 2)
    fxx = np.zeros(nmask + 2)
    fyy = np.zeros(nmask + 2)
    
    # Calculate values for each partition
    for idx in range(1, nmask + 1):
        mask = (MASK == idx)
        fx[idx] = np.sum(weights[mask] * freq_cos[mask])
        fy[idx] = np.sum(weights[mask] * freq_sin[mask])
        fxx[idx] = np.sum(weights[mask] * freq2_cos2[mask])
        fyy[idx] = np.sum(weights[mask] * freq2_sin2[mask])
    
    # Calculate spreading (variance in Cartesian space)
    Eip = np.zeros(nmask)
    for i in range(nmask):
        idx = i + 1  # Adjust for 1-based indexing
        Eip[i] = fxx[idx]/Etot - (fx[idx]/Etot)**2 + fyy[idx]/Etot - (fy[idx]/Etot)**2
    
    return Eip




def merge_overlapping_systems(MASK, dist, Eip, peaks, nmask):
    """
    Merge overlapping wave systems based on proximity and spreading criteria.
    
    This function implements the partition merging step of the Hanson & Phillips algorithm.
    Two partitions are merged if their peaks are close relative to their spreading parameters,
    indicating they likely represent the same wave system that was incorrectly split.
    
    The merging criterion is:
        dist[i,j] ≤ 0.5 * Eip[i] AND dist[i,j] ≤ 0.5 * Eip[j]
    
    Where:
        - dist[i,j] is the squared distance between peaks i and j
        - Eip[i] is the spreading parameter of partition i
        - Factor 0.5 is an empirical threshold (can be tuned)
    
    Parameters
    ----------
    MASK : ndarray (NF, ND)
        Current partition mask with labels 1 to nmask
    dist : ndarray (nmask, nmask)
        Squared distance matrix between all peak pairs
    Eip : ndarray (nmask,)
        Spreading parameter for each partition
    peaks : ndarray (nmask, 2)
        Peak locations [frequency_index, direction_index] with 1-based indexing
    nmask : int
        Number of partitions before merging
    
    Returns
    -------
    M : ndarray (NF, ND)
        Updated partition mask after merging
        Merged partitions take the label of the lower-numbered partition
    
    Notes
    -----
    - Merging is asymmetric: partition j is merged into partition i
    - Only affects partitions that meet both distance criteria
    - Does not renumber partitions (may leave gaps in numbering)
    - Subsequent renumbering by energy is recommended
    """
    print("Checking for overlapping systems...")
    print(f"Number of masks: {nmask}")

    if nmask <= 1:
        return MASK.copy()
    
    M = MASK.copy()
    
    # Create threshold matrices for comparison
    thresholds_i = Eip * 0.5  # Merging factor (increase distance threshold)
    
    # Find pairs to merge
    for i in range(nmask):
        for j in range(i+1, nmask):
            # Check if systems should be merged
            if dist[i, j] <= thresholds_i[i] and dist[i, j] <= thresholds_i[j]:
                print(f"Distance {dist[i, j]} <= Thresholds ({thresholds_i[i]}, {thresholds_i[j]})")
                print("Merging systems!")
                
                # Get indices for peaks i and j
                i_idx = int(peaks[i, 0]) - 1
                j_i_idx = int(peaks[i, 1]) - 1
                j_idx = int(peaks[j, 0]) - 1
                j_j_idx = int(peaks[j, 1]) - 1
                
                # Update mask to merge systems
                system_j_mask = (M == j + 1)
                M[system_j_mask] = i + 1
                M[j_idx, j_j_idx] = i + 1
    
    return M
def calculate_partitioned_energy(E, M, delf, ddir, NF, ND, nmask):
    """
    Calculate energy and significant wave height for each partition.
    
    Integrates the spectral energy density over each partition's spectral domain
    to obtain the total energy (m0) and derived significant wave height (Hs) for
    each wave system identified by the partitioning algorithm.
    
    Parameters
    ----------
    E : ndarray (NF, ND)
        2D spectral energy density matrix in m²·s·rad⁻¹
    M : ndarray (NF, ND)
        Partition mask with integer labels (0=unclassified, 1 to nmask=partitions)
    delf : ndarray (NF,)
        Frequency bin widths in Hz
    ddir : float
        Directional bin width in radians
    NF : int
        Number of frequency bins
    ND : int
        Number of directional bins
    nmask : int
        Number of partitions
    
    Returns
    -------
    e : ndarray (nmask+2,)
        Energy for each partition in m²
        Index 0: unclassified energy
        Indices 1 to nmask: partition energies
        Index nmask+1: residual energy (if any)
    Hs : ndarray (nmask+2,)
        Significant wave height for each partition in meters
        Calculated as Hs = 4√(energy)
    
    Notes
    -----
    - Energy integration: e[k] = ∑∑ E[i,j] * Δf[i] * Δθ  where M[i,j] = k
    - Debug output prints energy conservation check
    - Hs relationship: Hs ≈ 4√m0 for narrow-banded spectra
    """
    e = np.zeros(nmask + 2)

    for i in range(NF):
        for j in range(ND):
            mask_idx = M[i, j]
            e[mask_idx] += E[i, j] * delf[i] * ddir

    # Debug: sum of energies of partitions
    print(f"[DEBUG] Sum of partition energies: {np.sum(e):.6f}")
    print(f"[DEBUG] Expected total: {np.sum(E * np.tile(delf[:, np.newaxis], (1, ND)) * ddir):.6f}")
    
    Hs = 4 * np.sqrt(e)  # Significant wave height per partition
    return e, Hs

def renumber_partitions_by_energy(mask, Hs, e=None):
    """
    Renumber partitions by energy content (highest energy becomes partition #1).
    
    Sorts all identified wave systems by their total energy content and reassigns
    partition labels such that partition 1 has the most energy, partition 2 has
    the second most, etc. This provides a consistent ordering convention across
    different spectra.
    
    Parameters
    ----------
    mask : ndarray (NF, ND)
        Current partition mask with arbitrary numbering
    Hs : ndarray
        Significant wave height array for each partition (used as energy proxy)
    e : ndarray, optional
        Actual energy values for each partition
        If provided, will also be reordered to match new numbering
    
    Returns
    -------
    new_mask : ndarray (NF, ND)
        Renumbered partition mask with energy-based ordering
    new_Hs : ndarray
        Reordered significant wave height array
    new_e : ndarray, optional
        Reordered energy array (only if e was provided)
    
    Notes
    -----
    - Partition 0 (unclassified) is preserved
    - Higher-numbered partitions beyond nmask are mapped to nmask+1
    - Uses Hs as energy proxy (since Hs ∝ √energy)
    - Convention: larger Hs → more important wave system → lower partition number
    """
    unique_partitions = sorted([p for p in np.unique(mask) if p > 0 and p < len(Hs)])
    if len(unique_partitions) == 0:
        if e is not None:
            return mask.copy(), Hs.copy(), e.copy()
        else:
            return mask.copy(), Hs.copy()
    
    # Obter energies for each partition
    partition_energies = [(p, Hs[p]) for p in unique_partitions]
    sorted_partitions = sorted(partition_energies, key=lambda x: x[1], reverse=True)
    
    partition_mapping = {}
    new_Hs = np.zeros_like(Hs)
    new_Hs[0] = Hs[0]
    
    if e is not None:
        new_e = np.zeros_like(e)
        new_e[0] = e[0]
    
    for new_idx, (old_idx, _) in enumerate(sorted_partitions, start=1):
        partition_mapping[old_idx] = new_idx
        new_Hs[new_idx] = Hs[old_idx]
        if e is not None:
            new_e[new_idx] = e[old_idx]
    
    new_mask = np.zeros_like(mask)
    for old_idx, new_idx in partition_mapping.items():
        new_mask[mask == old_idx] = new_idx
    
    new_mask[mask == 0] = 0
    new_mask[mask >= len(Hs)] = len(sorted_partitions) + 1
    
    if e is not None:
        return new_mask, new_Hs, new_e
    else:
        return new_mask, new_Hs

def calculate_peak_parameters(E, mask, frequencies, directions_rad, NF, ND, nmask, delf, ddir):
    """
    Calculate peak period (Tp) and peak direction (Dp) for each partition.
    
    For each identified wave system, computes the characteristic period and direction
    using the same methodology as for the total spectrum: finding the frequency of
    maximum energy and calculating the energy-weighted mean direction at that frequency.
    
    Parameters
    ----------
    E : ndarray (NF, ND)
        2D spectral energy density matrix
    mask : ndarray (NF, ND)
        Partition mask (1 to nmask for partitions)
    frequencies : ndarray (NF,)
        Frequency array in Hz
    directions_rad : ndarray (ND,)
        Direction array in radians
    NF : int
        Number of frequency bins
    ND : int
        Number of directional bins
    nmask : int
        Number of partitions
    delf : ndarray (NF,)
        Frequency bin widths
    ddir : float
        Directional bin width in radians
    
    Returns
    -------
    Tp : ndarray (nmask+2,)
        Peak period for each partition in seconds
        NaN for partitions with in the energy
    Dp : ndarray (nmask+2,)
        Peak direction for each partition in degrees (0-360°)
        NaN for partitions with in the energy
    
    Algorithm
    ---------
    For each partition:
        1. Create partition-specific spectrum (zero elsewhere)
        2. Integrate over direction to get 1D frequency spectrum
        3. Find frequency of maximum: fp
        4. Calculate peak period: Tp = 1/fp
        5. At frequency fp, compute energy-weighted mean direction: Dp
    
    Notes
    -----
    - Uses same method as calculate_wave_parameters for consistency
    - Direction is oceanographic convention (direction waves travel TO)
    - Returns NaN for empty partitions
    """
    Tp = np.full(nmask + 2, np.nan)
    Dp = np.full(nmask + 2, np.nan)

    for idx in range(1, nmask + 1):
        # Mask for this partition
        partition_mask = (mask == idx)
        if not np.any(partition_mask):
            continue
        
        # 1D spectrum for this partition
        E_part = np.zeros_like(E)
        E_part[partition_mask] = E[partition_mask]
        spec1d, _ = spectrum1d_from_2d(E_part, directions_rad)
        
        # Find peak in frequency spectrum
        i_peak = np.argmax(spec1d) if np.max(spec1d) > 0 else 0
        if frequencies[i_peak] > 0:
            Tp[idx] = 1.0 / frequencies[i_peak]
        
        # Find direction at peak
        if np.any(E_part[i_peak, :] > 0):
            weighted_dir = np.sum(E_part[i_peak, :] * directions_rad) / np.sum(E_part[i_peak, :])
            Dp[idx] = np.degrees(weighted_dir) % 360
    
    return Tp, Dp

def calculate_spectral_moments(E, mask, freq, dirs_rad, delf, ddir, partition_idx=None):
    """
    Calculate spectral moments (m0, m1, m2) for a partition or entire spectrum.
    
    Computes the first three spectral moments which characterize the energy distribution
    in frequency space. These moments are fundamental to wave statistics and provide
    information about total energy, mean frequency, and frequency spread.
    
    Parameters
    ----------
    E : ndarray (NF, ND)
        2D spectral energy density matrix
    mask : ndarray (NF, ND) or None
        Partition mask (only used if partition_idx is specified)
    freq : ndarray (NF,)
        Frequency array in Hz
    dirs_rad : ndarray (ND,)
        Direction array in radians
    delf : ndarray (NF,)
        Frequency bin widths in Hz
    ddir : float
        Directional bin width in radians
    partition_idx : int, optional
        Partition number to calculate moments for
        If None, calculates for entire spectrum
    
    Returns
    -------
    m0 : float
        Zeroth moment (total energy) in m²
        m0 = ∫∫ E(f,θ) df dθ
    m1 : float
        First moment in m²·rad/s
        m1 = ∫∫ ω·E(f,θ) df dθ  where ω = 2πf
    m2 : float
        Second moment in m²·rad²/s²
        m2 = ∫∫ ω²·E(f,θ) df dθ
    
    Notes
    -----
    - Uses angular frequency ω = 2πf for moment calculations
    - If partition_idx provided, only integrates over that partition's domain
    - Moments relate to physical parameters:
        * m0: total wave energy
        * m1/m0: mean angular frequency
        * √(m2/m0): RMS angular frequency
        * (m0·m2/m1² - 1): spectral width parameter
    """
    # If partition_idx is None, calculate for entire spectrum
    if partition_idx is not None:
        # Create a mask for this partition
        E_mask = np.zeros_like(E)
        E_mask[mask == partition_idx] = E[mask == partition_idx]
        E_calc = E_mask
    else:
        E_calc = E
    
    # Get 1D spectrum
    spec1d, _ = spectrum1d_from_2d(E_calc, dirs_rad)
    
    # Calculate spectral moments
    m0 = 0.0
    m1 = 0.0
    m2 = 0.0
    
    for i in range(len(freq)):
        if freq[i] > 0:  # Avoid division by zero
            omega = 2 * np.pi * freq[i]  # Convert to angular frequency (rad/s)
            m0 += spec1d[i] * delf[i]
            m1 += omega * spec1d[i] * delf[i]
            m2 += (omega**2) * spec1d[i] * delf[i]
    
    return m0, m1, m2




def partition_spectrum(E, frequencies, directions_rad, energy_threshold, max_partitions):
    """
    Execute complete spectrum partitioning process using Hanson & Phillips algorithm.
    
    This is the main entry point for spectral partitioning. It implements the full
    Hanson & Phillips (2001) algorithm to identify and separate multiple wave systems
    (swell and wind sea) present in a 2D directional wave spectrum. The algorithm
    consists of several steps:
    
    1. Peak identification: Find local maxima in the spectrum
    2. Watershed segmentation: Assign each spectrum point to nearest peak
    3. Distance calculation: Compute separation between peaks
    4. Spreading calculation: Measure spatial extent of each partition
    5. Merging: Combine overlapping/nearby systems
    6. Parameter calculation: Compute Hs, Tp, Dp for each partition
    7. Reordering: Sort partitions by energy (largest first)
    
    Parameters
    ----------
    E : ndarray (NF, ND)
        2D directional spectral energy density in m²·s·rad⁻¹
    frequencies : ndarray (NF,)
        Frequency array in Hz
    directions_rad : ndarray (ND,)
        Direction array in radians (oceanographic convention)
    energy_threshold : float
        Minimum energy threshold for peak identification
    max_partitions : int
        Maximum number of partitions/peaks to identify
    
    Returns
    -------
    results : dict or None
        Dictionary containing partitioning results, or None if in the peaks found.
        Keys:
            'mask' : ndarray (NF, ND)
                Partition mask with labels 1 to nmask
            'energy' : ndarray (nmask+2,)
                Energy (m0) for each partition in m²
            'Hs' : ndarray (nmask+2,)
                Significant wave height for each partition in m
            'Tp' : ndarray (nmask+2,)
                Peak period for each partition in s
            'Dp' : ndarray (nmask+2,)
                Peak direction for each partition in degrees
            'total_m0' : float
                Total spectrum energy in m²
            'total_Hs' : float
                Total significant wave height in m
            'total_Tp' : float
                Total peak period in s
            'total_Dp' : float
                Total peak direction in degrees
            'nmask' : int
                Number of partitions identified
            'peaks' : ndarray (nmask, 2)
                Peak locations [freq_idx, dir_idx]
            'moments' : dict
                Spectral moments (m0, m1, m2) for total and each partition
    
    References
    ----------
    Hanson, J. L., & Phillips, O. M. (2001). Automated analysis of ocean surface
    directional wave spectra. Journal of Atmospheric and Oceanic Technology, 18(2),
    277-293.
    
    Notes
    -----
    - Energy conservation is checked and reported
    - Partitions are numbered by energy: 1 = most energetic system
    - Returns None if in the spectral peaks are identified
    
    Examples
    --------
    >>> # Define parameters explicitly
    >>> energy_threshold = 0.05
    >>> max_partitions = 5
    >>> results = partition_spectrum(E2d, freq, dirs_rad, energy_threshold, max_partitions)
    >>> print(f"Found {results['nmask']} wave systems")
    >>> print(f"Primary system: Hs={results['Hs'][1]:.2f}m, Tp={results['Tp'][1]:.1f}s")
    """
    NF, ND = E.shape
    ICOD, MASK, peaks, nmask = identify_spectral_peaks(
        E, NF, ND, energy_threshold, max_partitions
    )
    if nmask == 0:
        print("No spectral peaks identified.")
        return None
    MASK = generate_mask(ICOD, MASK, NF, ND)
    # Ensure alignment before proceeding
    if E.shape != MASK.shape:
        print(f"[DEBUG] Correcting shape: E{E.shape} vs MASK{MASK.shape}")
        if E.shape == MASK.T.shape:
            MASK = MASK.T
        else:
            raise ValueError(f"Incompatible shape: E{E.shape} vs MASK{MASK.shape}")
    distances = calculate_peak_distances(peaks, frequencies, directions_rad, nmask)
    hs, tp, dp, m0, delf, ddir, _, _ = calculate_wave_parameters(E, frequencies, directions_rad)
    Eip = calculate_peak_spreading(E, MASK, frequencies, directions_rad, NF, ND, nmask, m0, delf, ddir)
    MASK = merge_overlapping_systems(MASK, distances, Eip, peaks, nmask)
    
    # Use corrected energy calculation function
    e, Hs = calculate_partitioned_energy(E, MASK, delf, ddir, NF, ND, nmask)
    
    # Verify energy sum
    e_total = np.sum(e)
    print(f"Total spectrum energy: {m0:.6f}")
    print(f"Sum of partitioned energies: {e_total:.6f}")
    if abs(e_total - m0) > 1e-4:
        print(f"WARNING: Discrepancy in total energy: {abs(e_total - m0):.6f}")
    
    # Renumber partitions by energy - MUST BE DONE BEFORE calculating spectral moments
    M_renumbered, Hs_renumbered, e_renumbered = renumber_partitions_by_energy(MASK, Hs, e)
    
    # Use corrected function for Tp and Dp of partitions:
    Tp, Dp = calculate_peak_parameters(E, M_renumbered, frequencies, directions_rad, NF, ND, nmask, delf, ddir)
    
    # Calculate spectral moments for total spectrum - NOW after renumbering
    m0_total, m1_total, m2_total = calculate_spectral_moments(E, None, frequencies, directions_rad, delf, ddir)
    
    # Calculate spectral moments for each partition - USING M_renumbered
    m0_parts = np.zeros(nmask + 2)
    m1_parts = np.zeros(nmask + 2)
    m2_parts = np.zeros(nmask + 2)
    for idx in range(nmask + 2):
        if idx <= nmask or idx == 0:  # Calculate for each partition and unclassified (0)
            m0_parts[idx], m1_parts[idx], m2_parts[idx] = calculate_spectral_moments(
                E, M_renumbered, frequencies, directions_rad, delf, ddir, idx
            )
    
    # Create results dictionary
    results = {
        "mask": M_renumbered,
        "energy": e_renumbered,
        "Hs": Hs_renumbered,
        "Tp": Tp,
        "Dp": Dp,
        "total_m0": m0,
        "total_Hs": 4*np.sqrt(m0),
        "total_Tp": tp,
        "total_Dp": dp,
        "nmask": nmask,
        "peaks": peaks,
        # Add spectral moments
        "moments": {
            "total": (m0_total, m1_total, m2_total),
            "m0": m0_parts,
            "m1": m1_parts,
            "m2": m2_parts
        }
    }
    
    return results


def plot_directional_spectrum(E2d, freq, dirs, selected_time, hs, tp, dp):
    """
    Plot 2D polar directional spectrum.
    
    Parameters:
    -----------
    E2d : array-like
        2D directional spectrum energy
    freq : array-like
        Frequency array
    dirs : array-like
        Direction array (degrees)
    selected_time : datetime
        Timestamp for the data
    hs : float
        Significant wave height (m)
    tp : float
        Peak period (s)
    dp : float
        Peak direction (degrees)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.cm import ScalarMappable
    
    Eplot = np.nan_to_num(E2d, nan=0.0, neginf=0.0, posinf=0.0)

    # Ensure 1D arrays
    freq_plot = np.asarray(freq).flatten()
    dirs_plot = np.asarray(dirs).flatten()

    # Convert directions to radians & sort
    dirs_rad_plot = np.radians(dirs_plot)
    sort_idx = np.argsort(dirs_rad_plot)
    dirs_sorted = dirs_rad_plot[sort_idx]
    Eplot_sorted = Eplot[:, sort_idx]

    # Guarantee periodic wrap (append 2pi)
    if not np.isclose(dirs_sorted[0], 0.0):
        dirs_sorted = np.insert(dirs_sorted, 0, 0.0)
        Eplot_sorted = np.insert(Eplot_sorted, 0, Eplot_sorted[:, 0], axis=1)
    if not np.isclose(dirs_sorted[-1], 2*np.pi):
        dirs_sorted = np.append(dirs_sorted, 2*np.pi)
        Eplot_sorted = np.concatenate([Eplot_sorted, Eplot_sorted[:, 0:1]], axis=1)

    # Radial = period (s)
    with np.errstate(divide='ignore', invalid='ignore'):
        period = np.where(freq_plot > 0, 1.0 / freq_plot, 0)

    theta, r = np.meshgrid(dirs_sorted, period)

    # Default color scales
    vmin = 2.5
    vmax = 25
    step = max((vmax - vmin)/50.0, 0.5)
    levels = np.arange(vmin + step, vmax + step*0.51, step)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')

    cs = ax.contour(theta, r, Eplot_sorted, levels, cmap='jet', vmin=vmin, vmax=vmax)

    # Axes style
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rticks([5, 10, 15, 20])
    ax.set_yticklabels(['5s', '10s', '15s', '20s'], color='gray', fontsize=7.5)
    ax.set_rlim(0, 25)
    ax.set_rlabel_position(30)
    ax.tick_params(axis='y', colors='gray', labelsize=16)
    ticks = ['N','NE','E','SE','S','SW','W','NW']
    tick_angles = np.deg2rad(np.linspace(0, 315, 8))
    ax.set_xticks(tick_angles)
    ax.set_xticklabels(ticks)
    ax.tick_params(axis='x', colors='k', labelsize=16)
    title = 'Directional Spectrum'
    ax.set_title(title, fontsize=16, color='k', pad=30)

    # Stats box
    stats_ax = fig.add_axes([0.75, 0.7, 0.2, 0.15], facecolor='white')
    stats_ax.patch.set_alpha(0.8)
    stats_ax.patch.set_edgecolor('black')
    stats_ax.patch.set_linewidth(1.5)
    stats_ax.axis('off')

    stats_ax.text(0.7, 1.9, 'Statistics', fontsize=14, color='k', ha='center', va='center', weight='bold')
    date_str = selected_time.strftime('%Y-%m-%d %H:%M:%S')
    stats_ax.text(0.7, 1.7, f'Date: {date_str}', fontsize=12, color='k', ha='center', va='center')
    y_offset = 1.55
    stats_ax.text(0.7, y_offset, f'Hs: {hs:.2f} m', fontsize=12, color='k', ha='center', va='center')
    stats_ax.text(0.7, y_offset-0.15, f'Tp: {tp:.1f} s', fontsize=12, color='k', ha='center', va='center')
    stats_ax.text(0.7, y_offset-0.3, f'Dp: {dp:.1f}°', fontsize=12, color='k', ha='center', va='center')

    colorbar_label = 'm²·s·rad⁻¹'

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.025, pad=0.1, ax=ax, extend='both')
    cbar.set_label(colorbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    tick_interval = (vmax - vmin) / 5
    cbar.set_ticks(np.arange(vmin, vmax + 0.5 * tick_interval, tick_interval))

    # Manual adjustment instead of tight_layout
    fig.subplots_adjust(left=0.06, right=0.86, top=0.9, bottom=0.05)

    plt.show()
    
    return fig, ax

