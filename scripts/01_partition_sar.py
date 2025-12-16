"""
PASSO 1: Particionar espectros SAR

Este script processa espectros 2D do SAR (Sentinel-1), aplica o algoritmo de 
particionamento e salva os resultados em CSV.

Workflow:
1. Lê arquivo SAR (NetCDF) com espectros 2D
2. Converte de número de onda (k) para frequência (f)
3. Aplica algoritmo watershed de particionamento
4. Salva Hs, Tp, Dp para cada partição identificada
"""

import os
import sys
import pandas as pd
import xarray as xr
import numpy as np

# Adicionar lib/ ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from io_sar import load_sar_spectrum
from wave_params import calculate_wave_parameters
from partition import partition_spectrum

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# case = 'surigae'
# case = 'lee'
# case = 'freddy'
case = 'all'

# Diretórios
OUTPUT_DIR = f'../data/{case}/partition'
SAR_DATA_PATH = f'/Users/jtakeo/data/sentinel1ab/{case}'
CSV_PATH = f'../auxdata/sar_matches_{case}_track.csv'

# Parâmetros do particionamento
MIN_ENERGY_THRESHOLD_FRACTION = 0.01  # 1% da energia total
PEAK_DETECTION_SENSITIVITY = 0.5
MAX_PARTITIONS = 5

# Nome do grupo NetCDF (estrutura CMEMS)
GROUP_NAME = "obs_params"

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def calculate_corrected_total_hs(results, min_energy_threshold):
    """
    Calcula Hs total corrigido a partir da soma das energias das partições
    
    Returns:
    --------
    energy_sum : float
        Soma das energias das partições [m²]
    Hs : float
        Altura significativa total [m]
    """
    energy_sum = sum(results['energy'][i] for i in range(len(results['energy'])) 
                     if i == 0 or results['energy'][i] > min_energy_threshold)
    Hs = 4 * np.sqrt(energy_sum)
    return energy_sum, Hs


def count_significant_partitions(results, min_energy_threshold):
    """Conta partições com energia acima do threshold"""
    return sum(1 for i in range(1, len(results['Hs'])) 
               if results['energy'][i] > min_energy_threshold)


# ============================================================================
# FUNÇÕES DE I/O
# ============================================================================

def load_sar_data(file_path, index):
    """
    Carrega dataset SAR e extrai espectro
    
    Parameters:
    -----------
    file_path : str
        Caminho do arquivo NetCDF SAR
    index : int
        Índice da observação a carregar
    
    Returns:
    --------
    ds : xarray.Dataset
        Dataset SAR aberto
    E2d : ndarray
        Espectro 2D [m²·s·rad⁻¹]
    freq : ndarray
        Frequências [Hz]
    dirs : ndarray
        Direções [graus]
    dirs_rad : ndarray
        Direções [radianos]
    selected_time : pd.Timestamp
        Timestamp da observação
    """
    ds = xr.open_dataset(file_path, group=GROUP_NAME)
    E2d, freq, dirs, dirs_rad, selected_time = load_sar_spectrum(
        ds, date_time=None, index=index
    )
    return ds, E2d, freq, dirs, dirs_rad, selected_time


def extract_sar_metadata(ds, index):
    """
    Extrai metadados do dataset SAR
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset SAR
    index : int
        Índice da observação
    
    Returns:
    --------
    quality_flag : int
        Flag de qualidade (0 = melhor)
    actual_time : pd.Timestamp
        Timestamp da observação
    lon : float
        Longitude
    lat : float
        Latitude
    """
    quality_flag = ds.L2_partition_quality_flag[index].values
    actual_time = pd.to_datetime(ds.time[index].values)
    lon = ds.longitude[index].values if 'longitude' in ds else None
    lat = ds.latitude[index].values if 'latitude' in ds else None
    return quality_flag, actual_time, lon, lat


# ============================================================================
# PROCESSAMENTO DE DADOS
# ============================================================================

def create_partition_data_dict(ref, index, quality_flag, date_time, lon, lat, 
                                file_path, energy_sum, Hs, results, min_energy_threshold):
    """
    Cria dicionário com resultados do particionamento
    
    Returns:
    --------
    dict: Dicionário pronto para conversão em DataFrame
    """
    moments = results['moments']
    
    data = {
        'reference_id': ref,
        'obs_index': index,
        'quality_flag': quality_flag,
        'obs_time': date_time,
        'longitude': float(lon),
        'latitude': float(lat),
        'source_file': os.path.basename(file_path),
        
        # Espectro total
        'total_energy': energy_sum,
        'total_Hs': Hs,
        'total_Tp': results['total_Tp'],
        'total_Dp': results['total_Dp'],
        'total_m0': moments['total'][0],
        'total_m1': moments['total'][1],
        'total_m2': moments['total'][2],
    }
    
    # Adicionar dados das partições (até 3 partições)
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
            # Preencher com zeros se partição não existe
            data.update({
                f'P{p}_energy': 0.0, 
                f'P{p}_Hs': 0.0, 
                f'P{p}_Tp': 0.0,
                f'P{p}_Dp': 0.0, 
                f'P{p}_m0': 0.0, 
                f'P{p}_m1': 0.0, 
                f'P{p}_m2': 0.0,
            })
    
    return data


def save_partition_results(ref, index, date_time, data, output_dir):
    """
    Salva resultados do particionamento em arquivo CSV
    
    Returns:
    --------
    tuple: (output_path, df_results)
    """
    dt = pd.to_datetime(date_time)
    date_time_formatted = dt.strftime('%Y%m%d-%H%M%S')
    output_filename = f'sar_{ref:03d}_{index}_{date_time_formatted}.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    df_results = pd.DataFrame([data])
    df_results.to_csv(output_path, index=False, float_format='%.6f')
    
    return output_path, df_results


# ============================================================================
# FUNÇÕES DE IMPRESSÃO
# ============================================================================

def print_case_header(idx, total_cases, ref, index, file_name):
    """Imprime cabeçalho do caso sendo processado"""
    print(f"\n{'='*60}")
    print(f"Processing case {idx + 1}/{total_cases}")
    print(f"{'='*60}")
    print(f"File: {file_name}")
    print(f"Reference ID: {ref}, Observation index: {index}")


def print_location_info(lon, lat, quality_flag, hs, tp, dp):
    """Imprime informações de localização e parâmetros integrados"""
    print(f"Location: ({lon:.2f}°E, {lat:.2f}°N)")
    print(f"Quality flag: {quality_flag}")
    print(f"Integrated spectrum: Hs={hs:.2f}m, Tp={tp:.2f}s, Dp={dp:.0f}°")


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


def print_save_confirmation(output_path, energy_sum, Hs):
    """Imprime confirmação de salvamento"""
    print(f"\n✓ Results saved to: {output_path}")
    print(f"✓ Corrected total: energy={energy_sum:.6f}m², Hs={Hs:.3f}m")


# ============================================================================
# PROCESSAMENTO PRINCIPAL
# ============================================================================

def process_single_case(row, idx, total_cases, output_dir):
    """
    Processa um único caso SAR
    
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
    # Extrair parâmetros
    file_path = os.path.join(SAR_DATA_PATH, row['filename'])
    index = int(row['obs_index'])
    date_time = row['time']
    ref = int(row['ref'])
    
    # Imprimir cabeçalho
    print_case_header(idx, total_cases, ref, index, os.path.basename(file_path))
    
    # Carregar dados
    ds, E2d, freq, dirs, dirs_rad, selected_time = load_sar_data(file_path, index)
    
    # Calcular parâmetros integrados
    hs, tp, dp, m0, _, _, _, _ = calculate_wave_parameters(E2d, freq, dirs_rad)
    quality_flag, actual_time, lon, lat = extract_sar_metadata(ds, index)
    
    # Imprimir informações
    print_location_info(lon, lat, quality_flag, hs, tp, dp)
    
    # Aplicar particionamento
    results = partition_spectrum(
        E2d, freq, dirs_rad, PEAK_DETECTION_SENSITIVITY, MAX_PARTITIONS
    )
    
    if results is None:
        print("⚠ No spectral peaks identified!")
        ds.close()
        return
    
    # Calcular threshold e contar partições
    min_energy_threshold = MIN_ENERGY_THRESHOLD_FRACTION * results['total_m0']
    n_peaks_initial = len(results['peaks'])
    n_partitions_final = count_significant_partitions(results, min_energy_threshold)
    
    # Imprimir resultados
    print_partitioning_summary(n_peaks_initial, n_partitions_final)
    print_partitioning_results(results, min_energy_threshold)
    
    # Calcular Hs corrigido
    energy_sum, Hs = calculate_corrected_total_hs(results, min_energy_threshold)
    
    # Criar e salvar resultados
    data = create_partition_data_dict(
        ref, index, quality_flag, date_time, lon, lat,
        file_path, energy_sum, Hs, results, min_energy_threshold
    )
    output_path, _ = save_partition_results(ref, index, date_time, data, output_dir)
    
    # Imprimir confirmação
    print_save_confirmation(output_path, energy_sum, Hs)
    
    # Fechar dataset
    ds.close()


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
    print(f"SAR SPECTRAL PARTITIONING PROCESSOR")
    print(f"{'='*60}")
    print(f"Case: {case}")
    print(f"Total cases to process: {total_cases}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Processar cada caso
    for idx, row in df.iterrows():
        try:
            process_single_case(row, idx, total_cases, OUTPUT_DIR)
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
