"""
STEP 4: Validatete SAR vs WW3

This script compares SAR and WW3 partitioning results, matching
similar partitions and generating validatetion metrics and plots.

Workflow:
1. Load SAR and WW3 partitioning CSVs
2. Match partitions based on Tp and Dp
3. Generate paired partition files (partition1.csv, partition2.csv, partition3.csv)
4. Create dispersion plots comparing SAR vs WW3
5. Calculate statistical metrics (bias, RMSE, correlation)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

# ============================================================================
# CONFIGURATION
# ============================================================================

# case = 'surigae'
#case = 'lee'
# case = 'freddy'
case = 'all'

# Directories
WW3_DIR = f'../data/{case}/partition'
SAR_DIR = f'../data/{case}/partition'
OUTPUT_DIR = f'../output/{case}'

# Filtros
QUALITY_FLAG_OPTIONS = [0]  # Apenas data SAR of high quality (0 = best)

# Partition matching criteria
TP_TOLERANCE = 2.0   # Tolerance in peak period [s]
DP_TOLERANCE = 30.0  # Tolerance in peak direction [degrees]
TP_MIN_THRESHOLD = 12.0  # SAR not reliable for Tp < 10s (wind sea)
MAX_TIME_DIFF_HOURS = 1.0  # Maximum acceptable temporal difference [hours]

# Plot limits
PLOT_LIMITS = {
    'Hs': (0, 8),
    'Tp': (4, 20),
    'Dp': (0, 360)
}

# Variables to compare
COMPARISON_VARIABLES = [
    'total_Hs', 'total_Tp', 'total_Dp',
    'P1_Hs', 'P1_Tp', 'P1_Dp',
    'P2_Hs', 'P2_Tp', 'P2_Dp',
    'P3_Hs', 'P3_Tp', 'P3_Dp'
]

# ============================================================================
# I/O AND FILTERING FUNCTIONS
# ============================================================================

def load_and_filter_sar(sar_file, quality_flags=None):
    """
    Load data SAR e filtra por quality_flag
    
    Parameters:
    -----------
    sar_file : Path
        Path of file SAR
    quality_flags : list, optional
        Lista of quality flags aceitos (standard: [0])
    
    Returns:
    --------
    pd.DataFrame: Dados SAR filtrados
    """
    if quality_flags is None:
        quality_flags = QUALITY_FLAG_OPTIONS
    
    df = pd.read_csv(sar_file)
    
    if 'quality_flag' in df.columns:
        df_filtered = df[df['quality_flag'].isin(quality_flags)]
    else:
        df_filtered = df
    
    return df_filtered


def load_ww3_files_dict(ww3_dir):
    """
    Load todos os files WW3 e cria dictionary indexado por reference_id
    
    IMPORTANT: Each reference_id can have multiple simulated timestamps.
    Armazenamos uma lista of DataFrames for fazer matching temporal posterior.
    
    Parameters:
    -----------
    ww3_dir : str or Path
        Directory contendo files WW3
    
    Returns:
    --------
    dict: {reference_id: list of DataFrames}
    """
    ww3_files = list(Path(ww3_dir).glob('ww3_*.csv'))
    ww3_dict = {}
    
    for ww3_file in ww3_files:
        df_ww3 = pd.read_csv(ww3_file)
        if 'reference_id' in df_ww3.columns and len(df_ww3) > 0:
            ref_id = df_ww3['reference_id'].iloc[0]
            
            # Store as list to support multiple timestamps
            if ref_id not in ww3_dict:
                ww3_dict[ref_id] = []
            ww3_dict[ref_id].append(df_ww3)
    
    return ww3_dict


# ============================================================================
# PARTITION MATCHING FUNCTIONS
# ============================================================================

def validatete_temporal_match(sar_time, ww3_time, max_diff_hours=MAX_TIME_DIFF_HOURS):
    """
    Validate se o matching temporal between SAR e WW3 é acceptable
    
    Parameters:
    -----------
    sar_time : str or pd.Timestamp
        Timestamp of the observation SAR
    ww3_time : str or pd.Timestamp
        Timestamp of the simulation WW3
    max_diff_hours : float
        Difference temporal maximum acceptable [hours]
    
    Returns:
    --------
    tuple: (is_valid, time_diff_hours)
        is_valid: bool - True se difference temporal é acceptable
        time_diff_hours: float - Difference temporal in hours
    """
    sar_dt = pd.to_datetime(sar_time)
    ww3_dt = pd.to_datetime(ww3_time)
    
    time_diff_hours = abs((sar_dt - ww3_dt).total_seconds() / 3600.0)
    is_valid = time_diff_hours <= max_diff_hours
    
    return is_valid, time_diff_hours


def compute_angular_difference(angle1, angle2):
    """
    Calculates difference angular minimum considering circularity (0-360°)
    
    Parameters:
    -----------
    angle1, angle2 : float
        Angles in degrees
    
    Returns:
    --------
    float: Difference angular minimum [0, 180]
    """
    return abs((angle1 - angle2 + 180) % 360 - 180)


def extract_partitions_from_row(row, prefix=''):
    """
    Extracts data of partitions (Hs, Tp, Dp) of uma line of DataFrame
    
    Parameters:
    -----------
    row : pd.Series
        Row of DataFrame
    prefix : str
        Prefixo of colunas (vazio por standard)
    
    Returns:
    --------
    list: Lista of dicts with data of partitions
    """
    partitions = []
    
    for p in [1, 2, 3]:
        hs = row.get(f'{prefix}P{p}_Hs', np.nan)
        tp = row.get(f'{prefix}P{p}_Tp', np.nan)
        dp = row.get(f'{prefix}P{p}_Dp', np.nan)
        
        # Add only if Hs is not NaN (partition exists)
        if not np.isnan(hs):
            partitions.append({
                'partition': p,
                'hs': hs,
                'tp': tp,
                'dp': dp
            })
    
    return partitions


def find_best_match(sar_partitions, ww3_partitions, sar_pnum, 
                    tp_tol=TP_TOLERANCE, dp_tol=DP_TOLERANCE, 
                    tp_min=TP_MIN_THRESHOLD):
    """
    Find a best partition WW3 correspondente a uma partition SAR
    baseado in proximidade of Tp e Dp.
    
    Note: Partitions with Tp < 10s are rejected since SAR not é reliable
    for detection of wind sea of high frequency.
    
    Parameters:
    -----------
    sar_partitions : list
        Lista of partitions SAR
    ww3_partitions : list
        Lista of partitions WW3
    sar_pnum : int
        Number of the partition SAR (1, 2, ou 3)
    tp_tol : float
        Tolerance in Tp [s]
    dp_tol : float
        Tolerance in Dp [degrees]
    tp_min : float
        Tp minimum for considerar [s]
    
    Returns:
    --------
    tuple: (sar_data, ww3_data) ou (sar_data, None) se not houver match
    """
    # Findr partition SAR
    sar_data = next((p for p in sar_partitions if p['partition'] == sar_pnum), None)
    if not sar_data:
        return None, None
    
    # Reject SAR partitions with Tp < tp_min (SAR unreliable for wind sea)
    if not np.isnan(sar_data['tp']) and sar_data['tp'] < tp_min:
        return sar_data, None
    
    # Buscar best match WW3
    best_ww3 = None
    min_score = None
    
    for ww3 in ww3_partitions:
        # Skip if values are NaN
        if (np.isnan(ww3['tp']) or np.isnan(sar_data['tp']) or
            np.isnan(ww3['dp']) or np.isnan(sar_data['dp'])):
            continue
        
        # Reject WW3 partitions with Tp < tp_min (cannot be validateted with SAR)
        if ww3['tp'] < tp_min:
            continue
        
        # Calculate differences
        tp_diff = abs(ww3['tp'] - sar_data['tp'])
        dp_diff = compute_angular_difference(ww3['dp'], sar_data['dp'])
        
        # Accept only if both within tolerance
        if tp_diff <= tp_tol and dp_diff <= dp_tol:
            # Score ponderado for desempate
            score = tp_diff + dp_diff / 40.0
            
            if min_score is None or score < min_score:
                min_score = score
                best_ww3 = ww3
    
    return sar_data, best_ww3


def create_match_record(ref_id, sar_row, ww3_row, sar_match, ww3_match, time_diff_hours):
    """
    Create registro of partitions paired
    
    Returns:
    --------
    dict: Registro with data SAR, WW3 e differences
    """
    tp_diff = abs(sar_match['tp'] - ww3_match['tp'])
    dp_diff = compute_angular_difference(sar_match['dp'], ww3_match['dp'])
    
    return {
        'reference_id': ref_id,
        'sar_obs_time': sar_row.get('obs_time', ''),
        'ww3_obs_time': ww3_row.get('obs_time', ''),
        'time_diff_hours': time_diff_hours,
        'longitude': sar_row.get('longitude', np.nan),
        'latitude': sar_row.get('latitude', np.nan),
        'quality_flag': sar_row.get('quality_flag', np.nan),
        'sar_partition': sar_match['partition'],
        'ww3_partition': ww3_match['partition'],
        'sar_Hs': sar_match['hs'],
        'sar_Tp': sar_match['tp'],
        'sar_Dp': sar_match['dp'],
        'ww3_Hs': ww3_match['hs'],
        'ww3_Tp': ww3_match['tp'],
        'ww3_Dp': ww3_match['dp'],
        'tp_diff': tp_diff,
        'dp_diff': dp_diff
    }


# ============================================================================
# CREATING PAIRED PARTITION FILES
# ============================================================================

def create_partition_matches(tp_tol=TP_TOLERANCE, dp_tol=DP_TOLERANCE, 
                             quality_flags=None):
    """
    Create files of partitions paired (partition1.csv, partition2.csv, partition3.csv)
    
    Returns:
    --------
    dict: {partition_num: list of matches}
    """
    if quality_flags is None:
        quality_flags = QUALITY_FLAG_OPTIONS
    
    sar_files = list(Path(SAR_DIR).glob('sar_*.csv'))
    ww3_dict = load_ww3_files_dict(WW3_DIR)
    
    print(f"Found {len(sar_files)} SAR files and {len(ww3_dict)} WW3 files")
    
    # Storage for paired partitions and statistics
    partition_matches = {1: [], 2: [], 3: []}
    total_sar_files = 0
    temporal_match_valid = 0
    temporal_match_rejected = 0
    spatial_match_not_found = 0
    
    # Process each file SAR
    for sar_file in sar_files:
        df_sar = load_and_filter_sar(sar_file, quality_flags=quality_flags)
        
        if len(df_sar) == 0:
            continue
        
        total_sar_files += 1
        
        # Obter reference_id of file SAR
        if 'reference_id' not in df_sar.columns:
            continue
        
        ref_id = df_sar['reference_id'].iloc[0]
        
        # Findr data WW3 correspondentes (matching espacial)
        if ref_id not in ww3_dict:
            spatial_match_not_found += 1
            continue
        
        ww3_list = ww3_dict[ref_id]  # Lista of DataFrames WW3 for este ref_id
        
        # Extract time SAR
        if len(df_sar) == 0:
            continue
            
        sar_row = df_sar.iloc[0]
        sar_time = sar_row.get('obs_time', '')
        sar_time_dt = pd.to_datetime(sar_time)
        
        # Find closest WW3 temporally
        best_ww3 = None
        best_time_diff = float('inf')
        
        for df_ww3 in ww3_list:
            if len(df_ww3) == 0:
                continue
            ww3_row = df_ww3.iloc[0]
            ww3_time = ww3_row.get('obs_time', '')
            ww3_time_dt = pd.to_datetime(ww3_time)
            
            time_diff = abs((sar_time_dt - ww3_time_dt).total_seconds() / 3600.0)
            
            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_ww3 = (df_ww3, ww3_row, time_diff)
        
        # Validatete if best temporal match is acceptable
        if best_ww3 is None:
            spatial_match_not_found += 1
            continue
            
        df_ww3, ww3_row, time_diff_hours = best_ww3
        
        if time_diff_hours > MAX_TIME_DIFF_HOURS:
            temporal_match_rejected += 1
            continue
        
        temporal_match_valid += 1
        
        # Extract partitions
        sar_partitions = extract_partitions_from_row(sar_row)
        ww3_partitions = extract_partitions_from_row(ww3_row)
        
        # Findr melhores matches for each partition SAR
        for sar_pnum in [1, 2, 3]:
            sar_match, ww3_match = find_best_match(
                sar_partitions, ww3_partitions, sar_pnum,
                tp_tol=tp_tol, dp_tol=dp_tol
            )
            
            if sar_match and ww3_match:
                match_record = create_match_record(
                    ref_id, sar_row, ww3_row, sar_match, ww3_match, time_diff_hours
                )
                partition_matches[sar_pnum].append(match_record)
    
    # Print matching statistics
    print(f"\n{'='*70}")
    print(f" TEMPORAL MATCHING STATISTICS")
    print(f"{'='*70}")
    print(f"Total SAR files processed: {total_sar_files}")
    print(f"Spatial matches found (same reference_id): {temporal_match_valid + temporal_match_rejected}")
    print(f"Spatial matches NOT found: {spatial_match_not_found}")
    print(f"\nTemporal validatetion (max diff = {MAX_TIME_DIFF_HOURS} hour):")
    print(f"  ✓ Valid temporal matches: {temporal_match_valid}")
    print(f"  ✗ Rejected (time diff > {MAX_TIME_DIFF_HOURS}h): {temporal_match_rejected}")
    
    if temporal_match_valid + temporal_match_rejected > 0:
        valid_pct = 100 * temporal_match_valid / (temporal_match_valid + temporal_match_rejected)
        print(f"  Success rate: {valid_pct:.1f}%")
    print(f"{'='*70}")
    
    # Save partitions paired
    save_partition_matches(partition_matches)
    
    return partition_matches


def save_partition_matches(partition_matches):
    """Save partitions paired in files CSV e print summary"""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pnum in [1, 2, 3]:
        if len(partition_matches[pnum]) > 0:
            df_partition = pd.DataFrame(partition_matches[pnum])
            output_file = output_dir / f'partition{pnum}.csv'
            df_partition.to_csv(output_file, index=False)
            
            print(f"\nPartition {pnum}: {len(df_partition)} matches found")
            print(f"  Saved to: {output_file}")
            print(f"  SAR Hs range: [{df_partition['sar_Hs'].min():.2f}, {df_partition['sar_Hs'].max():.2f}] m")
            print(f"  WW3 Hs range: [{df_partition['ww3_Hs'].min():.2f}, {df_partition['ww3_Hs'].max():.2f}] m")
            print(f"  Mean Tp diff: {df_partition['tp_diff'].mean():.2f} s")
            print(f"  Mean Dp diff: {df_partition['dp_diff'].mean():.2f}°")
        else:
            print(f"\nPartition {pnum}: No matches found")


# ============================================================================
# METRICS CALCULATION FUNCTIONS
# ============================================================================

def compute_metrics(obs, model):
    """
    Calculates metrics of comparison between observations e model
    
    Parameters:
    -----------
    obs : array-like
        Valores observados (SAR)
    model : array-like
        Valores of model (WW3)
    
    Returns:
    --------
    dict: Dictionary with metrics (nbias, nrmse, pearson_r, n_points)
    """
    obs = np.array(obs)
    model = np.array(model)
    
    # Remover valores NaN
    mask = ~(np.isnan(obs) | np.isnan(model))
    obs = obs[mask]
    model = model[mask]
    
    if len(obs) == 0:
        return {'nbias': np.nan, 'nrmse': np.nan, 'pearson_r': np.nan, 'n_points': 0}
    
    # Bias normalizado
    bias = np.mean(model - obs)
    nbias = bias / np.mean(obs) if np.mean(obs) != 0 else np.nan
    
    # RMSE normalizado
    rmse = np.sqrt(np.mean((model - obs)**2))
    nrmse = rmse / np.mean(obs) if np.mean(obs) != 0 else np.nan
    
    # Pearson correlation
    if len(obs) > 1:
        pearson_r, _ = pearsonr(obs, model)
    else:
        pearson_r = np.nan
    
    return {
        'nbias': nbias,
        'nrmse': nrmse,
        'pearson_r': pearson_r,
        'n_points': len(obs)
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def setup_plot_axis(ax, var, axis_limits):
    """Configure limits e appearance dos axes"""
    axis_min, axis_max = axis_limits
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    return axis_min, axis_max


def plot_dispersion_and_lines(ax, sar_clean, ww3_clean, axis_min, axis_max):
    """Plot points of dispersion, line 1:1 e regression"""
    # Scatter plot
    ax.dispersion(sar_clean, ww3_clean, alpha=0.6, s=100, 
              edgecolors='black', linewidth=0.5)
    
    # Row 1:1
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 
           'r--', linewidth=2, label='1:1 line')
    
    # Row of regression
    if len(sar_clean) > 1:
        z = np.polyfit(sar_clean, ww3_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(axis_min, axis_max, 100)
        ax.plot(x_line, p(x_line), 'b-', linewidth=1.5, alpha=0.7,
               label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')


def add_metrics_textbox(ax, sar_clean, ww3_clean):
    """Add text box with metrics to plot"""
    # Calculate metrics
    bias = np.mean(ww3_clean - sar_clean)
    nbias = bias / np.mean(sar_clean) if np.mean(sar_clean) != 0 else np.nan
    rmse = np.sqrt(np.mean((ww3_clean - sar_clean)**2))
    nrmse = rmse / np.mean(sar_clean) if np.mean(sar_clean) != 0 else np.nan
    
    if len(sar_clean) > 1:
        pearson_r, _ = pearsonr(sar_clean, ww3_clean)
    else:
        pearson_r = np.nan
    
    # Create texto with metrics
    metrics_text = (
        f'n = {len(sar_clean)}\n'
        f'Bias = {bias:.3f}\n'
        f'NBias = {nbias:.3f}\n'
        f'RMSE = {rmse:.3f}\n'
        f'NRMSE = {nrmse:.3f}\n'
        f'R = {pearson_r:.3f}'
    )
    
    # Posicionar text box
    y_pos = 0.05 if np.mean(ww3_clean) > np.mean(sar_clean) else 0.95
    va = 'bottom' if y_pos == 0.05 else 'top'
    
    ax.text(0.95, y_pos, metrics_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment=va,
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_single_variable(ax, df, var, var_label):
    """Plot comparison for uma single variable"""
    sar_col = f'sar_{var}'
    ww3_col = f'ww3_{var}'
    
    sar_data = df[sar_col].values
    ww3_data = df[ww3_col].values
    
    # Remover valores NaN
    mask = ~(np.isnan(sar_data) | np.isnan(ww3_data))
    sar_clean = sar_data[mask]
    ww3_clean = ww3_data[mask]
    
    if len(sar_clean) == 0:
        ax.text(0.5, 0.5, 'No valid data',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(var_label)
        return
    
    # Obter limits dos axes
    axis_limits = PLOT_LIMITS.get(var, (0, max(sar_clean.max(), ww3_clean.max())))
    axis_min, axis_max = setup_plot_axis(ax, var, axis_limits)
    
    # Plotar data e linhas
    plot_dispersion_and_lines(ax, sar_clean, ww3_clean, axis_min, axis_max)
    
    # Add labels e legenda
    ax.set_xlabel(f'SAR {var}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'WW3 {var}', fontsize=12, fontweight='bold')
    ax.set_title(var_label, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    # Add metrics
    add_metrics_textbox(ax, sar_clean, ww3_clean)


def plot_partition_comparisons():
    """Create dispersion plots for each partition comparing SAR vs WW3"""
    output_dir = Path(OUTPUT_DIR)
    
    # Variables to plot
    variables = [
        ('Hs', 'Significant Wave Height (m)'),
        ('Tp', 'Peak Period (s)'),
        ('Dp', 'Peak Direction (deg)')
    ]
    
    for pnum in [1, 2, 3]:
        partition_file = output_dir / f'partition{pnum}.csv'
        
        if not partition_file.exists():
            print(f"Partition {pnum} file not found, skipping...")
            continue
        
        df = pd.read_csv(partition_file)
        
        if len(df) == 0:
            print(f"Partition {pnum} has in the data, skipping...")
            continue
        
        # Create figura with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Partition {pnum} - SAR vs WW3 Comparison (n={len(df)})',
                     fontsize=16, fontweight='bold')
        
        # Plotar each variable
        for idx, (var, var_label) in enumerate(variables):
            plot_single_variable(axes[idx], df, var, var_label)
        
        plt.tight_layout()
        
        # Save figura
        output_file = output_dir / f'partition{pnum}_dispersion.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    print("\nAll dispersion plots created successfully!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(create_files=True, create_plots=True):
    """
    Main execution function
    
    Parameters:
    -----------
    create_files : bool
        Se True, cria files of partitions paired (partition*.csv)
    create_plots : bool
        Se True, cria dispersion plots
    """
    if create_files:
        # Create files of partitions paired
        print("="*80)
        print("CREATING MATCHED PARTITION FILES")
        print("="*80)
        partition_matches = create_partition_matches(
            tp_tol=TP_TOLERANCE, 
            dp_tol=DP_TOLERANCE
        )
    
    if create_plots:
        # Create dispersion plots
        print("\n" + "="*80)
        print("CREATING SCATTER PLOTS")
        print("="*80)
        plot_partition_comparisons()
    
    print("\n" + "="*80)
    print("✓ Validatetion complete!")
    print("="*80)


if __name__ == '__main__':
    # ========================================================================
    # EXECUTION OPTIONS
    # ========================================================================
    
    RUN_CREATE_FILES = True   # Create files partition*.csv (matching SAR/WW3)
    RUN_CREATE_PLOTS = True   # Create dispersion plots
    
    # ========================================================================
    
    print("\n" + "="*80)
    print("VALIDATION CONFIGURATION")
    print("="*80)
    print(f"Case: {case}")
    print(f"Create partition files: {RUN_CREATE_FILES}")
    print(f"Create dispersion plots:   {RUN_CREATE_PLOTS}")
    print(f"Quality flags: {QUALITY_FLAG_OPTIONS}")
    print(f"Tp tolerance: {TP_TOLERANCE} s")
    print(f"Dp tolerance: {DP_TOLERANCE}°")
    print(f"Tp min threshold: {TP_MIN_THRESHOLD} s")
    print("="*80 + "\n")
    
    main(create_files=RUN_CREATE_FILES, create_plots=RUN_CREATE_PLOTS)
