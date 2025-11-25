import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend

# Configuration
WW3_DIR = '../data/ww3/partition'
SAR_DIR = '../data/sar/partition'
OUTPUT_DIR = '../output'
QUALITY_FLAG_OPTIONS = [0]

# Plot settings
PLOT_LIMITS = {
    'Hs': (0, 5),
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
# UTILITY FUNCTIONS
# ============================================================================

def load_and_filter_sar(sar_file, quality_flags=None):
    """Load SAR data and filter by quality_flag"""
    if quality_flags is None:
        quality_flags = QUALITY_FLAG_OPTIONS
    df = pd.read_csv(sar_file)
    df_filtered = df[df['quality_flag'].isin(quality_flags)]
    return df_filtered


def load_ww3_files_dict(ww3_dir):
    """Load all WW3 files and create a dictionary indexed by reference_id"""
    ww3_files = list(Path(ww3_dir).glob('partition_ww3_*.csv'))
    ww3_dict = {}
    
    for ww3_file in ww3_files:
        df_ww3 = pd.read_csv(ww3_file)
        if 'reference_id' in df_ww3.columns and len(df_ww3) > 0:
            ref_id = df_ww3['reference_id'].iloc[0]
            ww3_dict[ref_id] = df_ww3
    
    return ww3_dict


def compute_angular_difference(angle1, angle2):
    """Compute minimum angular difference considering circularity (0-360)"""
    return abs((angle1 - angle2 + 180) % 360 - 180)


# ============================================================================
# PARTITION MATCHING
# ============================================================================

def find_best_match(sar_partitions, ww3_partitions, sar_pnum, tp_tol=3.0, dp_tol=40.0):
    """
    Find best matching WW3 partition for a given SAR partition
    based on Tp and Dp proximity
    """
    # Find the SAR partition
    sar_data = next((p for p in sar_partitions if p['partition'] == sar_pnum), None)
    if not sar_data:
        return None, None
    
    # Search for best matching WW3 partition
    best_ww3 = None
    min_score = None
    
    for ww3 in ww3_partitions:
        # Skip if any required value is NaN
        if (np.isnan(ww3['tp']) or np.isnan(sar_data['tp']) or
            np.isnan(ww3['dp']) or np.isnan(sar_data['dp'])):
            continue
        
        # Calculate differences
        tp_diff = abs(ww3['tp'] - sar_data['tp'])
        dp_diff = compute_angular_difference(ww3['dp'], sar_data['dp'])
        
        # Accept only if both within tolerance
        if tp_diff <= tp_tol and dp_diff <= dp_tol:
            score = tp_diff + dp_diff / 40.0  # Weighted score for tie-breaking
            if min_score is None or score < min_score:
                min_score = score
                best_ww3 = ww3
    
    if best_ww3:
        return sar_data, best_ww3
    return sar_data, None


def extract_partitions_from_row(row, prefix=''):
    """Extract partition data (Hs, Tp, Dp) from a dataframe row"""
    partitions = []
    for p in [1, 2, 3]:
        hs = row.get(f'{prefix}P{p}_Hs', np.nan)
        tp = row.get(f'{prefix}P{p}_Tp', np.nan)
        dp = row.get(f'{prefix}P{p}_Dp', np.nan)
        
        if not np.isnan(hs):
            partitions.append({
                'partition': p,
                'hs': hs,
                'tp': tp,
                'dp': dp
            })
    return partitions


def create_match_record(ref_id, sar_row, sar_match, ww3_match):
    """Create a match record dictionary"""
    tp_diff = abs(sar_match['tp'] - ww3_match['tp'])
    dp_diff = compute_angular_difference(sar_match['dp'], ww3_match['dp'])
    
    return {
        'reference_id': ref_id,
        'obs_time': sar_row.get('obs_time', ''),
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
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(obs, model):
    """
    Compute comparison metrics between observations and model
    
    Parameters:
    -----------
    obs : array-like
        Observed values (SAR)
    model : array-like
        Model values (WW3)
    
    Returns:
    --------
    dict with nbias, nrmse, and pearson_r
    """
    obs = np.array(obs)
    model = np.array(model)
    
    # Remove NaN values
    mask = ~(np.isnan(obs) | np.isnan(model))
    obs = obs[mask]
    model = model[mask]
    
    if len(obs) == 0:
        return {'nbias': np.nan, 'nrmse': np.nan, 'pearson_r': np.nan, 'n_points': 0}
    
    # Normalized Bias
    bias = np.mean(model - obs)
    nbias = bias / np.mean(obs) if np.mean(obs) != 0 else np.nan
    
    # Normalized RMSE
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
# PARTITION MATCHING AND FILE GENERATION
# ============================================================================

def create_partition_matches(tp_tol=3.0, dp_tol=40.0, quality_flags=None):
    """
    Create matched partition files using find_best_match
    Filter SAR by quality_flag and match with WW3 partitions
    """
    if quality_flags is None:
        quality_flags = QUALITY_FLAG_OPTIONS
    
    sar_files = list(Path(SAR_DIR).glob('partition_sar_*.csv'))
    ww3_dict = load_ww3_files_dict(WW3_DIR)
    
    print(f"Found {len(sar_files)} SAR files and {len(ww3_dict)} WW3 files")
    
    # Storage for matched partitions
    partition_matches = {1: [], 2: [], 3: []}
    
    # Process each SAR file
    for sar_file in sar_files:
        df_sar = load_and_filter_sar(sar_file, quality_flags=quality_flags)
        
        if len(df_sar) == 0:
            continue
        
        # Get reference_id from SAR file
        if 'reference_id' not in df_sar.columns:
            continue
        
        ref_id = df_sar['reference_id'].iloc[0]
        
        # Find corresponding WW3 data
        if ref_id not in ww3_dict:
            continue
        
        df_ww3 = ww3_dict[ref_id]
        
        # Extract SAR and WW3 partitions
        if len(df_sar) > 0 and len(df_ww3) > 0:
            sar_row = df_sar.iloc[0]
            ww3_row = df_ww3.iloc[0]
            
            sar_partitions = extract_partitions_from_row(sar_row)
            ww3_partitions = extract_partitions_from_row(ww3_row)
            
            # Find best matches for each SAR partition
            for sar_pnum in [1, 2, 3]:
                sar_match, ww3_match = find_best_match(
                    sar_partitions, ww3_partitions, sar_pnum,
                    tp_tol=tp_tol, dp_tol=dp_tol
                )
                
                if sar_match and ww3_match:
                    match_record = create_match_record(ref_id, sar_row, sar_match, ww3_match)
                    partition_matches[sar_pnum].append(match_record)
    
    # Save matched partitions to separate CSV files
    save_partition_matches(partition_matches)
    
    return partition_matches


def save_partition_matches(partition_matches):
    """Save partition matches to CSV files and print summary"""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    for pnum in [1, 2, 3]:
        if len(partition_matches[pnum]) > 0:
            df_partition = pd.DataFrame(partition_matches[pnum])
            output_file = output_dir / f'partition{pnum}.csv'
            df_partition.to_csv(output_file, index=False)
            
            print(f"\nPartition {pnum}: {len(df_partition)} matches found")
            print(f"  Saved to: {output_file}")
            print(f"  SAR Hs range: [{df_partition['sar_Hs'].min():.2f}, {df_partition['sar_Hs'].max():.2f}]")
            print(f"  WW3 Hs range: [{df_partition['ww3_Hs'].min():.2f}, {df_partition['ww3_Hs'].max():.2f}]")
            print(f"  Mean Tp diff: {df_partition['tp_diff'].mean():.2f} s")
            print(f"  Mean Dp diff: {df_partition['dp_diff'].mean():.2f} deg")
        else:
            print(f"\nPartition {pnum}: No matches found")


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def setup_plot_axis(ax, var, axis_limits):
    """Setup axis limits, labels, and appearance"""
    axis_min, axis_max = axis_limits
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    return axis_min, axis_max


def plot_scatter_and_lines(ax, sar_clean, ww3_clean, axis_min, axis_max):
    """Plot scatter points, 1:1 line, and regression line"""
    # Scatter plot
    ax.scatter(sar_clean, ww3_clean, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # 1:1 line
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', linewidth=2, label='1:1 line')
    
    # Regression line
    if len(sar_clean) > 1:
        z = np.polyfit(sar_clean, ww3_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(axis_min, axis_max, 100)
        ax.plot(x_line, p(x_line), 'b-', linewidth=1.5, alpha=0.7,
               label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')


def add_metrics_textbox(ax, sar_clean, ww3_clean):
    """Add metrics text box to plot"""
    # Compute metrics
    bias = np.mean(ww3_clean - sar_clean)
    nbias = bias / np.mean(sar_clean) if np.mean(sar_clean) != 0 else np.nan
    rmse = np.sqrt(np.mean((ww3_clean - sar_clean)**2))
    nrmse = rmse / np.mean(sar_clean) if np.mean(sar_clean) != 0 else np.nan
    
    if len(sar_clean) > 1:
        pearson_r, _ = pearsonr(sar_clean, ww3_clean)
    else:
        pearson_r = np.nan
    
    # Create metrics text
    metrics_text = (
        f'n = {len(sar_clean)}\n'
        f'Bias = {bias:.3f}\n'
        f'NBias = {nbias:.3f}\n'
        f'RMSE = {rmse:.3f}\n'
        f'NRMSE = {nrmse:.3f}\n'
        f'R = {pearson_r:.3f}'
    )
    
    # Position text box
    y_pos = 0.05 if np.mean(ww3_clean) > np.mean(sar_clean) else 0.95
    va = 'bottom' if y_pos == 0.05 else 'top'
    
    ax.text(0.95, y_pos, metrics_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment=va,
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_single_variable(ax, df, var, var_label):
    """Plot comparison for a single variable"""
    sar_col = f'sar_{var}'
    ww3_col = f'ww3_{var}'
    
    sar_data = df[sar_col].values
    ww3_data = df[ww3_col].values
    
    # Remove NaN values
    mask = ~(np.isnan(sar_data) | np.isnan(ww3_data))
    sar_clean = sar_data[mask]
    ww3_clean = ww3_data[mask]
    
    if len(sar_clean) == 0:
        ax.text(0.5, 0.5, 'No valid data',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(var_label)
        return
    
    # Get axis limits
    axis_limits = PLOT_LIMITS.get(var, (0, max(sar_clean.max(), ww3_clean.max())))
    axis_min, axis_max = setup_plot_axis(ax, var, axis_limits)
    
    # Plot data and lines
    plot_scatter_and_lines(ax, sar_clean, ww3_clean, axis_min, axis_max)
    
    # Add labels and legend
    ax.set_xlabel(f'SAR {var}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'WW3 {var}', fontsize=12, fontweight='bold')
    ax.set_title(var_label, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    # Add metrics
    add_metrics_textbox(ax, sar_clean, ww3_clean)


def plot_partition_comparisons():
    """Create scatter plots for each partition comparing SAR vs WW3"""
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
            print(f"Partition {pnum} has no data, skipping...")
            continue
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Partition {pnum} - SAR vs WW3 Comparison (n={len(df)})',
                     fontsize=16, fontweight='bold')
        
        # Plot each variable
        for idx, (var, var_label) in enumerate(variables):
            plot_single_variable(axes[idx], df, var, var_label)
        
        plt.tight_layout()
        
        # Save figure
        output_file = output_dir / f'partition{pnum}_scatter.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    print("\nAll scatter plots created successfully!")


# ============================================================================
# COMPARISON AND METRICS
# ============================================================================

def process_comparison_data(sar_files, ww3_dict, quality_flags):
    """Process SAR and WW3 files to collect comparison data"""
    all_data = {var: {'sar': [], 'ww3': []} for var in COMPARISON_VARIABLES}
    results = []
    
    for sar_file in sar_files:
        df_sar = load_and_filter_sar(sar_file, quality_flags=quality_flags)
        
        if len(df_sar) == 0 or 'reference_id' not in df_sar.columns:
            continue
        
        ref_id = df_sar['reference_id'].iloc[0]
        
        if ref_id not in ww3_dict:
            continue
        
        df_ww3 = ww3_dict[ref_id]
        
        # Process each variable
        for var in COMPARISON_VARIABLES:
            if var in df_sar.columns and var in df_ww3.columns:
                sar_values = df_sar[var].values
                ww3_values = df_ww3[var].values
                
                if len(sar_values) > 0 and len(ww3_values) > 0:
                    # Compute metrics per file
                    metrics = compute_metrics(sar_values, ww3_values)
                    
                    results.append({
                        'reference_id': ref_id,
                        'variable': var,
                        'nbias': metrics['nbias'],
                        'nrmse': metrics['nrmse'],
                        'pearson_r': metrics['pearson_r'],
                        'n_points': metrics['n_points'],
                        'sar_mean': np.nanmean(sar_values),
                        'ww3_mean': np.nanmean(ww3_values)
                    })
                    
                    # Collect for global metrics
                    all_data[var]['sar'].extend(sar_values.tolist())
                    all_data[var]['ww3'].extend(ww3_values.tolist())
    
    return pd.DataFrame(results), all_data


def compute_global_metrics(all_data):
    """Compute global metrics from all collected data"""
    global_metrics = []
    
    for var in COMPARISON_VARIABLES:
        if len(all_data[var]['sar']) > 0:
            metrics = compute_metrics(all_data[var]['sar'], all_data[var]['ww3'])
            global_metrics.append({
                'variable': var,
                'nbias': metrics['nbias'],
                'nrmse': metrics['nrmse'],
                'pearson_r': metrics['pearson_r'],
                'n_points': metrics['n_points'],
                'sar_mean': np.nanmean(all_data[var]['sar']),
                'ww3_mean': np.nanmean(all_data[var]['ww3'])
            })
    
    return pd.DataFrame(global_metrics)


def save_comparison_results(df_results, df_global):
    """Save comparison results to CSV files"""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    output_file = output_dir / 'comparison_metrics_detailed.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save global metrics
    global_file = output_dir / 'comparison_metrics_global.csv'
    df_global.to_csv(global_file, index=False)
    print(f"Global metrics saved to: {global_file}")
    
    # Compute and save summary
    summary = df_results.groupby('variable').agg({
        'nbias': ['mean', 'std', 'count'],
        'nrmse': ['mean', 'std'],
        'n_points': 'sum'
    }).round(4)
    
    summary_file = output_dir / 'comparison_metrics_summary.csv'
    summary.to_csv(summary_file)
    print(f"Summary statistics saved to: {summary_file}")
    
    return summary


def print_comparison_results(df_global, summary):
    """Print comparison results to console"""
    print("\n" + "="*80)
    print("GLOBAL METRICS (All data points combined)")
    print("="*80)
    print(df_global.to_string(index=False))
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY VARIABLE (Per-file basis)")
    print("="*80)
    print(summary)


def compare_sar_ww3(quality_flags=None):
    """
    Compare SAR and WW3 partition data
    Filter SAR by quality_flag and compute metrics
    """
    if quality_flags is None:
        quality_flags = QUALITY_FLAG_OPTIONS
    
    sar_files = list(Path(SAR_DIR).glob('partition_sar_*.csv'))
    ww3_dict = load_ww3_files_dict(WW3_DIR)
    
    print(f"Found {len(sar_files)} SAR files and {len(ww3_dict)} WW3 files")
    
    # Process data and compute metrics
    df_results, all_data = process_comparison_data(sar_files, ww3_dict, quality_flags)
    
    if len(df_results) > 0:
        # Compute global metrics
        df_global = compute_global_metrics(all_data)
        
        # Save results
        summary = save_comparison_results(df_results, df_global)
        
        # Print results
        print_comparison_results(df_global, summary)
    else:
        print("\nNo matching data found for comparison!")
    
    return df_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(create_files=True, create_plots=True, compute_metrics=True):
    """
    Main execution function
    
    Parameters:
    -----------
    create_files : bool, default=True
        If True, create matched partition CSV files (partition1.csv, partition2.csv, partition3.csv)
    create_plots : bool, default=True
        If True, create scatter plots
    compute_metrics : bool, default=True
        If True, compute comparison metrics
    """
    df_results = None
    
    if create_files:
        # Create matched partition files
        print("="*80)
        print("CREATING MATCHED PARTITION FILES")
        print("="*80)
        partition_matches = create_partition_matches(tp_tol=3.0, dp_tol=40.0)
    
    if create_plots:
        # Create scatter plots
        print("\n" + "="*80)
        print("CREATING SCATTER PLOTS")
        print("="*80)
        plot_partition_comparisons()
    
    if compute_metrics:
        # Compute comparison metrics
        print("\n" + "="*80)
        print("COMPUTING COMPARISON METRICS")
        print("="*80)
        df_results = compare_sar_ww3()
    
    return df_results


if __name__ == '__main__':
    # ========================================================================
    # EXECUTION OPTIONS - Configure what to run
    # ========================================================================
    # Set to True/False to control execution:
    
    RUN_CREATE_FILES = False      # Create partition*.csv files (matching SAR/WW3)
    RUN_CREATE_PLOTS = True       # Create scatter plots
    RUN_COMPUTE_METRICS = False   # Compute and save comparison metrics
    
    # ========================================================================
    
    print("\n" + "="*80)
    print("EXECUTION CONFIGURATION")
    print("="*80)
    print(f"Create partition files: {RUN_CREATE_FILES}")
    print(f"Create scatter plots:   {RUN_CREATE_PLOTS}")
    print(f"Compute metrics:        {RUN_COMPUTE_METRICS}")
    print("="*80 + "\n")
    
    main(create_files=RUN_CREATE_FILES, 
         create_plots=RUN_CREATE_PLOTS, 
         compute_metrics=RUN_COMPUTE_METRICS)

