"""
PASSO 4: Validar SAR vs WW3

Este script compara os resultados de particionamento SAR e WW3, fazendo matching
de partições similares e gerando métricas de validação e gráficos.

Workflow:
1. Carrega CSVs de particionamento SAR e WW3
2. Faz matching de partições baseado em Tp e Dp
3. Gera arquivos de partições pareadas (partition1.csv, partition2.csv, partition3.csv)
4. Cria scatter plots comparando SAR vs WW3
5. Calcula métricas estatísticas (bias, RMSE, correlação)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para salvar figuras

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# case = 'surigae'
#case = 'lee'
# case = 'freddy'
case = 'all'

# Diretórios
WW3_DIR = f'../data/{case}/partition'
SAR_DIR = f'../data/{case}/partition'
OUTPUT_DIR = f'../output/{case}'

# Filtros
QUALITY_FLAG_OPTIONS = [0]  # Apenas dados SAR de alta qualidade (0 = melhor)

# Critérios de matching de partições
TP_TOLERANCE = 2.0   # Tolerância em período de pico [s]
DP_TOLERANCE = 30.0  # Tolerância em direção de pico [graus]
TP_MIN_THRESHOLD = 12.0  # SAR não é confiável para Tp < 10s (wind sea)
MAX_TIME_DIFF_HOURS = 1.0  # Diferença temporal máxima aceitável [horas]

# Limites dos gráficos
PLOT_LIMITS = {
    'Hs': (0, 8),
    'Tp': (4, 20),
    'Dp': (0, 360)
}

# Variáveis para comparar
COMPARISON_VARIABLES = [
    'total_Hs', 'total_Tp', 'total_Dp',
    'P1_Hs', 'P1_Tp', 'P1_Dp',
    'P2_Hs', 'P2_Tp', 'P2_Dp',
    'P3_Hs', 'P3_Tp', 'P3_Dp'
]

# ============================================================================
# FUNÇÕES DE I/O E FILTRAGEM
# ============================================================================

def load_and_filter_sar(sar_file, quality_flags=None):
    """
    Carrega dados SAR e filtra por quality_flag
    
    Parameters:
    -----------
    sar_file : Path
        Caminho do arquivo SAR
    quality_flags : list, optional
        Lista de quality flags aceitos (padrão: [0])
    
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
    Carrega todos os arquivos WW3 e cria dicionário indexado por reference_id
    
    IMPORTANTE: Cada reference_id pode ter múltiplos timestamps simulados.
    Armazenamos uma lista de DataFrames para fazer matching temporal posterior.
    
    Parameters:
    -----------
    ww3_dir : str or Path
        Diretório contendo arquivos WW3
    
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
            
            # Armazenar como lista para suportar múltiplos timestamps
            if ref_id not in ww3_dict:
                ww3_dict[ref_id] = []
            ww3_dict[ref_id].append(df_ww3)
    
    return ww3_dict


# ============================================================================
# FUNÇÕES DE MATCHING DE PARTIÇÕES
# ============================================================================

def validate_temporal_match(sar_time, ww3_time, max_diff_hours=MAX_TIME_DIFF_HOURS):
    """
    Valida se o matching temporal entre SAR e WW3 é aceitável
    
    Parameters:
    -----------
    sar_time : str or pd.Timestamp
        Timestamp da observação SAR
    ww3_time : str or pd.Timestamp
        Timestamp da simulação WW3
    max_diff_hours : float
        Diferença temporal máxima aceitável [horas]
    
    Returns:
    --------
    tuple: (is_valid, time_diff_hours)
        is_valid: bool - True se diferença temporal é aceitável
        time_diff_hours: float - Diferença temporal em horas
    """
    sar_dt = pd.to_datetime(sar_time)
    ww3_dt = pd.to_datetime(ww3_time)
    
    time_diff_hours = abs((sar_dt - ww3_dt).total_seconds() / 3600.0)
    is_valid = time_diff_hours <= max_diff_hours
    
    return is_valid, time_diff_hours


def compute_angular_difference(angle1, angle2):
    """
    Calcula diferença angular mínima considerando circularidade (0-360°)
    
    Parameters:
    -----------
    angle1, angle2 : float
        Ângulos em graus
    
    Returns:
    --------
    float: Diferença angular mínima [0, 180]
    """
    return abs((angle1 - angle2 + 180) % 360 - 180)


def extract_partitions_from_row(row, prefix=''):
    """
    Extrai dados de partições (Hs, Tp, Dp) de uma linha do DataFrame
    
    Parameters:
    -----------
    row : pd.Series
        Linha do DataFrame
    prefix : str
        Prefixo das colunas (vazio por padrão)
    
    Returns:
    --------
    list: Lista de dicts com dados das partições
    """
    partitions = []
    
    for p in [1, 2, 3]:
        hs = row.get(f'{prefix}P{p}_Hs', np.nan)
        tp = row.get(f'{prefix}P{p}_Tp', np.nan)
        dp = row.get(f'{prefix}P{p}_Dp', np.nan)
        
        # Adicionar apenas se Hs não for NaN (partição existe)
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
    Encontra a melhor partição WW3 correspondente a uma partição SAR
    baseado em proximidade de Tp e Dp.
    
    Nota: Partições com Tp < 10s são rejeitadas pois SAR não é confiável
    para detecção de wind sea de alta frequência.
    
    Parameters:
    -----------
    sar_partitions : list
        Lista de partições SAR
    ww3_partitions : list
        Lista de partições WW3
    sar_pnum : int
        Número da partição SAR (1, 2, ou 3)
    tp_tol : float
        Tolerância em Tp [s]
    dp_tol : float
        Tolerância em Dp [graus]
    tp_min : float
        Tp mínimo para considerar [s]
    
    Returns:
    --------
    tuple: (sar_data, ww3_data) ou (sar_data, None) se não houver match
    """
    # Encontrar partição SAR
    sar_data = next((p for p in sar_partitions if p['partition'] == sar_pnum), None)
    if not sar_data:
        return None, None
    
    # Rejeitar partições SAR com Tp < tp_min (SAR não confiável para wind sea)
    if not np.isnan(sar_data['tp']) and sar_data['tp'] < tp_min:
        return sar_data, None
    
    # Buscar melhor match WW3
    best_ww3 = None
    min_score = None
    
    for ww3 in ww3_partitions:
        # Pular se valores são NaN
        if (np.isnan(ww3['tp']) or np.isnan(sar_data['tp']) or
            np.isnan(ww3['dp']) or np.isnan(sar_data['dp'])):
            continue
        
        # Rejeitar partições WW3 com Tp < tp_min (não podem ser validadas com SAR)
        if ww3['tp'] < tp_min:
            continue
        
        # Calcular diferenças
        tp_diff = abs(ww3['tp'] - sar_data['tp'])
        dp_diff = compute_angular_difference(ww3['dp'], sar_data['dp'])
        
        # Aceitar apenas se ambas dentro da tolerância
        if tp_diff <= tp_tol and dp_diff <= dp_tol:
            # Score ponderado para desempate
            score = tp_diff + dp_diff / 40.0
            
            if min_score is None or score < min_score:
                min_score = score
                best_ww3 = ww3
    
    return sar_data, best_ww3


def create_match_record(ref_id, sar_row, ww3_row, sar_match, ww3_match, time_diff_hours):
    """
    Cria registro de partições pareadas
    
    Returns:
    --------
    dict: Registro com dados SAR, WW3 e diferenças
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
# CRIAÇÃO DE ARQUIVOS DE PARTIÇÕES PAREADAS
# ============================================================================

def create_partition_matches(tp_tol=TP_TOLERANCE, dp_tol=DP_TOLERANCE, 
                             quality_flags=None):
    """
    Cria arquivos de partições pareadas (partition1.csv, partition2.csv, partition3.csv)
    
    Returns:
    --------
    dict: {partition_num: list of matches}
    """
    if quality_flags is None:
        quality_flags = QUALITY_FLAG_OPTIONS
    
    sar_files = list(Path(SAR_DIR).glob('sar_*.csv'))
    ww3_dict = load_ww3_files_dict(WW3_DIR)
    
    print(f"Found {len(sar_files)} SAR files and {len(ww3_dict)} WW3 files")
    
    # Armazenamento para partições pareadas e estatísticas
    partition_matches = {1: [], 2: [], 3: []}
    total_sar_files = 0
    temporal_match_valid = 0
    temporal_match_rejected = 0
    spatial_match_not_found = 0
    
    # Processar cada arquivo SAR
    for sar_file in sar_files:
        df_sar = load_and_filter_sar(sar_file, quality_flags=quality_flags)
        
        if len(df_sar) == 0:
            continue
        
        total_sar_files += 1
        
        # Obter reference_id do arquivo SAR
        if 'reference_id' not in df_sar.columns:
            continue
        
        ref_id = df_sar['reference_id'].iloc[0]
        
        # Encontrar dados WW3 correspondentes (matching espacial)
        if ref_id not in ww3_dict:
            spatial_match_not_found += 1
            continue
        
        ww3_list = ww3_dict[ref_id]  # Lista de DataFrames WW3 para este ref_id
        
        # Extrair tempo SAR
        if len(df_sar) == 0:
            continue
            
        sar_row = df_sar.iloc[0]
        sar_time = sar_row.get('obs_time', '')
        sar_time_dt = pd.to_datetime(sar_time)
        
        # Encontrar o WW3 temporalmente mais próximo
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
        
        # Validar se o melhor match temporal é aceitável
        if best_ww3 is None:
            spatial_match_not_found += 1
            continue
            
        df_ww3, ww3_row, time_diff_hours = best_ww3
        
        if time_diff_hours > MAX_TIME_DIFF_HOURS:
            temporal_match_rejected += 1
            continue
        
        temporal_match_valid += 1
        
        # Extrair partições
        sar_partitions = extract_partitions_from_row(sar_row)
        ww3_partitions = extract_partitions_from_row(ww3_row)
        
        # Encontrar melhores matches para cada partição SAR
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
    
    # Imprimir estatísticas de matching
    print(f"\n{'='*70}")
    print(f" TEMPORAL MATCHING STATISTICS")
    print(f"{'='*70}")
    print(f"Total SAR files processed: {total_sar_files}")
    print(f"Spatial matches found (same reference_id): {temporal_match_valid + temporal_match_rejected}")
    print(f"Spatial matches NOT found: {spatial_match_not_found}")
    print(f"\nTemporal validation (max diff = {MAX_TIME_DIFF_HOURS} hour):")
    print(f"  ✓ Valid temporal matches: {temporal_match_valid}")
    print(f"  ✗ Rejected (time diff > {MAX_TIME_DIFF_HOURS}h): {temporal_match_rejected}")
    
    if temporal_match_valid + temporal_match_rejected > 0:
        valid_pct = 100 * temporal_match_valid / (temporal_match_valid + temporal_match_rejected)
        print(f"  Success rate: {valid_pct:.1f}%")
    print(f"{'='*70}")
    
    # Salvar partições pareadas
    save_partition_matches(partition_matches)
    
    return partition_matches


def save_partition_matches(partition_matches):
    """Salva partições pareadas em arquivos CSV e imprime resumo"""
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
# FUNÇÕES DE CÁLCULO DE MÉTRICAS
# ============================================================================

def compute_metrics(obs, model):
    """
    Calcula métricas de comparação entre observações e modelo
    
    Parameters:
    -----------
    obs : array-like
        Valores observados (SAR)
    model : array-like
        Valores do modelo (WW3)
    
    Returns:
    --------
    dict: Dicionário com métricas (nbias, nrmse, pearson_r, n_points)
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
    
    # Correlação de Pearson
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
# FUNÇÕES DE PLOTAGEM
# ============================================================================

def setup_plot_axis(ax, var, axis_limits):
    """Configura limites e aparência dos eixos"""
    axis_min, axis_max = axis_limits
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    return axis_min, axis_max


def plot_scatter_and_lines(ax, sar_clean, ww3_clean, axis_min, axis_max):
    """Plota pontos de dispersão, linha 1:1 e regressão"""
    # Scatter plot
    ax.scatter(sar_clean, ww3_clean, alpha=0.6, s=100, 
              edgecolors='black', linewidth=0.5)
    
    # Linha 1:1
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 
           'r--', linewidth=2, label='1:1 line')
    
    # Linha de regressão
    if len(sar_clean) > 1:
        z = np.polyfit(sar_clean, ww3_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(axis_min, axis_max, 100)
        ax.plot(x_line, p(x_line), 'b-', linewidth=1.5, alpha=0.7,
               label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')


def add_metrics_textbox(ax, sar_clean, ww3_clean):
    """Adiciona caixa de texto com métricas ao gráfico"""
    # Calcular métricas
    bias = np.mean(ww3_clean - sar_clean)
    nbias = bias / np.mean(sar_clean) if np.mean(sar_clean) != 0 else np.nan
    rmse = np.sqrt(np.mean((ww3_clean - sar_clean)**2))
    nrmse = rmse / np.mean(sar_clean) if np.mean(sar_clean) != 0 else np.nan
    
    if len(sar_clean) > 1:
        pearson_r, _ = pearsonr(sar_clean, ww3_clean)
    else:
        pearson_r = np.nan
    
    # Criar texto com métricas
    metrics_text = (
        f'n = {len(sar_clean)}\n'
        f'Bias = {bias:.3f}\n'
        f'NBias = {nbias:.3f}\n'
        f'RMSE = {rmse:.3f}\n'
        f'NRMSE = {nrmse:.3f}\n'
        f'R = {pearson_r:.3f}'
    )
    
    # Posicionar caixa de texto
    y_pos = 0.05 if np.mean(ww3_clean) > np.mean(sar_clean) else 0.95
    va = 'bottom' if y_pos == 0.05 else 'top'
    
    ax.text(0.95, y_pos, metrics_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment=va,
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_single_variable(ax, df, var, var_label):
    """Plota comparação para uma única variável"""
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
    
    # Obter limites dos eixos
    axis_limits = PLOT_LIMITS.get(var, (0, max(sar_clean.max(), ww3_clean.max())))
    axis_min, axis_max = setup_plot_axis(ax, var, axis_limits)
    
    # Plotar dados e linhas
    plot_scatter_and_lines(ax, sar_clean, ww3_clean, axis_min, axis_max)
    
    # Adicionar labels e legenda
    ax.set_xlabel(f'SAR {var}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'WW3 {var}', fontsize=12, fontweight='bold')
    ax.set_title(var_label, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    # Adicionar métricas
    add_metrics_textbox(ax, sar_clean, ww3_clean)


def plot_partition_comparisons():
    """Cria scatter plots para cada partição comparando SAR vs WW3"""
    output_dir = Path(OUTPUT_DIR)
    
    # Variáveis para plotar
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
        
        # Criar figura com 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Partition {pnum} - SAR vs WW3 Comparison (n={len(df)})',
                     fontsize=16, fontweight='bold')
        
        # Plotar cada variável
        for idx, (var, var_label) in enumerate(variables):
            plot_single_variable(axes[idx], df, var, var_label)
        
        plt.tight_layout()
        
        # Salvar figura
        output_file = output_dir / f'partition{pnum}_scatter.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    print("\nAll scatter plots created successfully!")


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

def main(create_files=True, create_plots=True):
    """
    Função principal de execução
    
    Parameters:
    -----------
    create_files : bool
        Se True, cria arquivos de partições pareadas (partition*.csv)
    create_plots : bool
        Se True, cria scatter plots
    """
    if create_files:
        # Criar arquivos de partições pareadas
        print("="*80)
        print("CREATING MATCHED PARTITION FILES")
        print("="*80)
        partition_matches = create_partition_matches(
            tp_tol=TP_TOLERANCE, 
            dp_tol=DP_TOLERANCE
        )
    
    if create_plots:
        # Criar scatter plots
        print("\n" + "="*80)
        print("CREATING SCATTER PLOTS")
        print("="*80)
        plot_partition_comparisons()
    
    print("\n" + "="*80)
    print("✓ Validation complete!")
    print("="*80)


if __name__ == '__main__':
    # ========================================================================
    # OPÇÕES DE EXECUÇÃO
    # ========================================================================
    
    RUN_CREATE_FILES = True   # Criar arquivos partition*.csv (matching SAR/WW3)
    RUN_CREATE_PLOTS = True   # Criar scatter plots
    
    # ========================================================================
    
    print("\n" + "="*80)
    print("VALIDATION CONFIGURATION")
    print("="*80)
    print(f"Case: {case}")
    print(f"Create partition files: {RUN_CREATE_FILES}")
    print(f"Create scatter plots:   {RUN_CREATE_PLOTS}")
    print(f"Quality flags: {QUALITY_FLAG_OPTIONS}")
    print(f"Tp tolerance: {TP_TOLERANCE} s")
    print(f"Dp tolerance: {DP_TOLERANCE}°")
    print(f"Tp min threshold: {TP_MIN_THRESHOLD} s")
    print("="*80 + "\n")
    
    main(create_files=RUN_CREATE_FILES, create_plots=RUN_CREATE_PLOTS)
