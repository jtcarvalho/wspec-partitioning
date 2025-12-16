"""
PASSO 3: Particionar espectros WW3

Este script processa espectros 2D do modelo WW3, aplica o algoritmo de particionamento
e salva os resultados em CSV.

Workflow:
1. Lê coordenadas/timestamps do arquivo auxiliar
2. Para cada ponto, carrega espectro WW3 no tempo mais próximo
3. Aplica algoritmo watershed de particionamento
4. Salva Hs, Tp, Dp para cada partição identificada
"""

import os
import sys
import xarray as xr
import pandas as pd
import numpy as np

# Adicionar lib/ ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from io_ww3 import find_closest_time, load_ww3_spectrum
from wave_params import calculate_wave_parameters
from partition import partition_spectrum

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

#case = 'lee'
# case = 'freddy'
# case = 'surigae'
case = 'all'

# Diretórios
OUTPUT_DIR = f'../data/{case}/partition'
CSV_PATH = f'../auxdata/sar_matches_{case}_track.csv'
WW3_DATA_PATH = f'/Users/jtakeo/data/ww3/{case}'

# Parâmetros do particionamento
MIN_ENERGY_THRESHOLD_FRACTION = 0.01  # 1% da energia total
PEAK_DETECTION_SENSITIVITY = 0.5

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def count_significant_partitions(results, min_energy_threshold):
    """Conta partições com energia acima do threshold"""
    count = 0
    for i in range(1, len(results['Hs'])):
        if results['energy'][i] > min_energy_threshold:
            count += 1
    return count


# ============================================================================
# FUNÇÕES DE IMPRESSÃO
# ============================================================================

def print_case_header(idx, total_cases, ref, target_time_dt):
    """Imprime cabeçalho do caso sendo processado"""
    print(f"\n{'='*60}")
    print(f"Processing case {idx + 1}/{total_cases}")
    print(f"{'='*60}")
    print(f"Reference ID: {ref}")
    print(f"Target SAR time: {target_time_dt}")


def print_time_match_info(closest_time, itime, time_diff_hours):
    """Imprime informações de matching temporal"""
    print(f"Closest WW3 time: {closest_time}")
    print(f"Time index (itime): {itime}")
    print(f"Time difference: {time_diff_hours:.2f} hours")


def print_partitioning_summary(n_peaks_initial, n_partitions_final):
    """Imprime resumo do processo de particionamento"""
    print("\n" + "="*70)
    print(" SPECTRAL PARTITIONING - PROCESS SUMMARY")
    print("="*70)
    print(f"🔍 Spectral peaks initially identified: {n_peaks_initial}")
    print(f"🔗 After merging nearby systems: {n_partitions_final} partition(s)")
    print("="*70)


def print_partitioning_results(results, min_energy_threshold):
    """Imprime resultados detalhados do particionamento"""
    n_partitions = count_significant_partitions(results, min_energy_threshold)
    
    print("\n" + "="*70)
    print(" PARTITIONING RESULTS")
    print("="*70)
    print(f"Number of partitions found: {n_partitions}")
    print("─"*70)
    
    # Mostrar cada partição
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
    
    # Mostrar total integrado
    print("\n" + "─"*70)
    print(f"Integrated total:")
    print(f"  Hs = {results['total_Hs']:.2f} m")
    print(f"  Tp = {results['total_Tp']:.2f} s")
    print(f"  Dp = {results['total_Dp']:.0f}°")
    print("="*70)


def print_save_confirmation(output_path, df_results):
    """Imprime confirmação de salvamento"""
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
    Cria dicionário com resultados do particionamento
    
    Returns:
    --------
    dict: Dicionário pronto para conversão em DataFrame
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
        
        # Espectro total
        'total_energy': results['total_m0'],
        'total_Hs': results['total_Hs'],
        'total_Tp': results['total_Tp'],
        'total_Dp': results['total_Dp'],
        'total_m0': m0_total,
        'total_m1': m1_total,
        'total_m2': m2_total,
    }
    
    # Adicionar dados das partições (até 3 partições)
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
            # Preencher com zeros se partição não existe
            data[f'P{p}_energy'] = 0.0
            data[f'P{p}_Hs'] = 0.0
            data[f'P{p}_Tp'] = 0.0
            data[f'P{p}_Dp'] = 0.0
            data[f'P{p}_m0'] = 0.0
            data[f'P{p}_m1'] = 0.0
            data[f'P{p}_m2'] = 0.0
    
    return data


def save_partition_results(ref, selected_time, data, output_dir):
    """
    Salva resultados do particionamento em arquivo CSV
    
    Returns:
    --------
    tuple: (output_path, df_results)
    """
    # Criar nome do arquivo
    date_time_formatted = selected_time.strftime('%Y%m%d-%H%M%S')
    output_filename = f'ww3_{ref:03d}_{date_time_formatted}.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    # Criar DataFrame e salvar
    df_results = pd.DataFrame([data])
    df_results.to_csv(output_path, index=False, float_format='%.6f')
    
    return output_path, df_results


def process_single_case(row, idx, total_cases, output_dir):
    """
    Processa um único caso WW3
    
    Parameters:
    -----------
    row : pandas.Series
        Linha do CSV com informações do caso
    idx : int
        Índice do caso atual
    total_cases : int
        Número total de casos a processar
    output_dir : str
        Diretório de saída para resultados
    """
    # Extrair informações do caso
    ref = int(row['ref'])
    target_time_str = row['time']
    target_time_dt = pd.to_datetime(target_time_str)
    
    # Construir caminho do arquivo
    file_path = f'{WW3_DATA_PATH}/ww3_sar{ref:03d}_2020_spec.nc'
    
    # Imprimir cabeçalho
    print_case_header(idx, total_cases, ref, target_time_dt)
    
    # Encontrar tempo mais próximo
    itime, closest_time, time_diff_hours = find_closest_time(file_path, target_time_dt)
    print_time_match_info(closest_time, itime, time_diff_hours)
    
    # Carregar espectro
    E2d, freq, dirs, dirs_rad, lon, lat = load_ww3_spectrum(file_path, itime)
    
    # Aplicar particionamento
    results = partition_spectrum(E2d, freq, dirs_rad, PEAK_DETECTION_SENSITIVITY, 5)
    
    if results is None:
        print("⚠ No spectral peaks identified!")
        return
    
    # Calcular threshold e contar partições
    min_energy_threshold = MIN_ENERGY_THRESHOLD_FRACTION * results['total_m0']
    n_peaks_initial = len(results['peaks'])
    n_partitions_final = count_significant_partitions(results, min_energy_threshold)
    
    # Imprimir resultados
    print_partitioning_summary(n_peaks_initial, n_partitions_final)
    print_partitioning_results(results, min_energy_threshold)
    
    # Criar e salvar resultados
    data = create_partition_data_dict(ref, closest_time, lon, lat, file_path,
                                      results, min_energy_threshold)
    output_path, df_results = save_partition_results(ref, closest_time, data, output_dir)
    
    # Imprimir confirmação
    print_save_confirmation(output_path, df_results)


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal de execução"""
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carregar casos
    df = pd.read_csv(CSV_PATH)
    total_cases = len(df)
    
    print(f"{'='*60}")
    print(f"WW3 SPECTRAL PARTITIONING PROCESSOR")
    print(f"{'='*60}")
    print(f"Total cases to process: {total_cases}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Processar cada caso
    for idx, row in df.iterrows():
        try:
            process_single_case(row, idx, total_cases, OUTPUT_DIR)
        except Exception as e:
            print(f"\n❌ Error processing case {idx + 1}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
