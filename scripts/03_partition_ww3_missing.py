"""
PASSO 3B: Particionar espectros WW3 - Processar casos faltantes

Este script processa apenas os reference_ids que não foram processados
no script 03_partition_ww3.py original, com melhor tratamento de erros.
"""

import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# Adicionar lib/ ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from io_ww3 import find_closest_time, load_ww3_spectrum
from wave_params import calculate_wave_parameters
from partition import partition_spectrum

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

case = 'all'

# Diretórios
OUTPUT_DIR = f'../data/{case}/partition'
CSV_PATH = f'../auxdata/sar_matches_{case}_track.csv'
WW3_DATA_PATH = f'/Users/jtakeo/data/ww3/{case}'

# Parâmetros do particionamento
MIN_ENERGY_THRESHOLD_FRACTION = 0.01
PEAK_DETECTION_SENSITIVITY = 0.5

# ============================================================================
# FUNÇÕES
# ============================================================================

def get_processed_ref_ids(output_dir):
    """Retorna set de reference_ids já processados"""
    output_path = Path(output_dir)
    ww3_files = list(output_path.glob('ww3_*.csv'))
    
    processed = set()
    for f in ww3_files:
        ref_id = int(f.stem.split('_')[1])
        processed.add(ref_id)
    
    return processed


def get_missing_ref_ids(csv_path, processed_ref_ids):
    """Retorna lista de reference_ids que precisam ser processados"""
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # Filtrar por período WW3
    df_period = df[(df['time'] >= '2020-06-15') & (df['time'] <= '2020-12-31')]
    
    all_ref_ids = set(df_period['ref'].values)
    missing = sorted(list(all_ref_ids - processed_ref_ids))
    
    return missing, df_period


def process_single_ref_id(ref_id, df_sar, output_dir):
    """
    Processa um único reference_id
    
    Returns:
    --------
    tuple: (success, message)
    """
    # Encontrar dados SAR para este ref_id
    sar_rows = df_sar[df_sar['ref'] == ref_id]
    if len(sar_rows) == 0:
        return False, "No SAR data found"
    
    sar_row = sar_rows.iloc[0]
    target_time = pd.to_datetime(sar_row['time'])
    
    # Verificar se arquivo WW3 existe
    file_path = f'{WW3_DATA_PATH}/ww3_sar{ref_id:04d}_2020_spec.nc'
    if not os.path.exists(file_path):
        return False, "WW3 NetCDF file not found"
    
    try:
        # Encontrar tempo mais próximo
        itime, closest_time, time_diff_hours = find_closest_time(file_path, target_time)
        
        # Carregar espectro
        E2d, freq, dirs, dirs_rad, lon, lat = load_ww3_spectrum(file_path, itime)
        
        # Aplicar particionamento
        results = partition_spectrum(E2d, freq, dirs_rad, PEAK_DETECTION_SENSITIVITY, 5)
        
        if results is None:
            return False, "No spectral peaks identified"
        
        # Calcular threshold
        min_energy_threshold = MIN_ENERGY_THRESHOLD_FRACTION * results['total_m0']
        moments = results['moments']
        
        # Criar dicionário de dados
        data = {
            'reference_id': ref_id,
            'obs_time': closest_time,
            'longitude': float(lon),
            'latitude': float(lat),
            'source_file': os.path.basename(file_path),
            'total_energy': results['total_m0'],
            'total_Hs': results['total_Hs'],
            'total_Tp': results['total_Tp'],
            'total_Dp': results['total_Dp'],
            'total_m0': moments['total'][0],
            'total_m1': moments['total'][1],
            'total_m2': moments['total'][2],
        }
        
        # Adicionar partições
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
                data[f'P{p}_energy'] = 0.0
                data[f'P{p}_Hs'] = 0.0
                data[f'P{p}_Tp'] = 0.0
                data[f'P{p}_Dp'] = 0.0
                data[f'P{p}_m0'] = 0.0
                data[f'P{p}_m1'] = 0.0
                data[f'P{p}_m2'] = 0.0
        
        # Salvar
        date_time_formatted = closest_time.strftime('%Y%m%d-%H%M%S')
        output_filename = f'ww3_{ref_id:03d}_{date_time_formatted}.csv'
        output_path = os.path.join(output_dir, output_filename)
        
        df_results = pd.DataFrame([data])
        df_results.to_csv(output_path, index=False, float_format='%.6f')
        
        return True, f"Saved to {output_filename}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal de execução"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"WW3 SPECTRAL PARTITIONING - PROCESSING MISSING CASES")
    print(f"{'='*70}")
    
    # Identificar casos faltantes
    processed = get_processed_ref_ids(OUTPUT_DIR)
    missing_ids, df_sar = get_missing_ref_ids(CSV_PATH, processed)
    
    print(f"\nProcessed reference_ids: {len(processed)}")
    print(f"Missing reference_ids: {len(missing_ids)}")
    
    if len(missing_ids) == 0:
        print("\n✓ All reference_ids already processed!")
        return
    
    # Processar casos faltantes
    print(f"\nProcessing {len(missing_ids)} missing cases...")
    print(f"{'='*70}\n")
    
    success_count = 0
    error_count = 0
    errors_log = []
    
    for idx, ref_id in enumerate(missing_ids):
        success, message = process_single_ref_id(ref_id, df_sar, OUTPUT_DIR)
        
        if success:
            success_count += 1
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(missing_ids)} (✓ {success_count}, ✗ {error_count})")
        else:
            error_count += 1
            errors_log.append((ref_id, message))
            if error_count <= 20:  # Mostrar apenas os primeiros 20 erros
                print(f"  ✗ ref_id {ref_id}: {message}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    
    if error_count > 0:
        print(f"\nMost common errors:")
        error_types = {}
        for ref_id, msg in errors_log:
            error_type = msg.split(':')[0]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count} cases")
    
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
