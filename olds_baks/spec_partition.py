#!/usr/bin/env python3
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import glob
import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from directional_plot import plot_directional_spectrum

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES GLOBAIS
# -----------------------------------------------------------------------------

CONFIG = {
    # Configurações de entrada
    "source_type": "sar",         # "ww3", "ndbc", ou "sar"
    "data_directory": "/Users/jtakeo/googleDrive/myProjects/finalPartSpec/surigae/sarspec/sar-data",  # Diretório com arquivos NetCDF SAR
    "data_ini": "2021-01-01 00:00",  # Data/hora inicial (ajustada para os dados disponíveis)
    "data_fim": "2021-12-31 23:59",  # Data/hora final (ajustada para os dados disponíveis)
    "time_step_hours": 1,         # Intervalo de horas entre processamentos (1, 3, 6, etc)
    
    # Configurações para processamento batch
    "batch_mode": True,          # Se True, processa múltiplos pontos do CSV
    "csv_points_file": "/Users/jtakeo/googleDrive/myProjects/sar-spec-partitioning/auxdata/sar_matches_with_track_3day.csv",  # CSV com ref, obs_index, time e filename
    
    # Configurações de processamento
    "station_idx": 0,            # Índice da estação para WW3
    "spectrum_unit": "m2_s_rad", # Unidade para espectros (usar m2_Hz_rad como padrão)
    "sar_scaling_factor": 1.0,   # Fator de escala para SAR
    
    # Configurações de particionamento
    "do_partition": True,        # Executar particionamento
    "energy_threshold": 5e-2,    # Limiar ainda mais restritivo para SAR
    "max_partitions": 3,         # Número máximo de partições (muito reduzido)
    "relative_threshold": 0.25,  # Threshold relativo ao pico máximo (25% - muito restritivo)
    
    # Configurações de saída
    "save_spectrum": True,
    "save_plot": True,
    "show_plots": False,
    "use_enhanced_plot": True,
    "output_dir": "/Users/jtakeo/googleDrive/myProjects/sar-spec-partitioning/output",
    # Padronização com ww3
    "point_name": None,
    "max_density_1d": 25.0,
    "max_density_2d": 25.0
}

# -----------------------------------------------------------------------------
# HELPER PARA NOMES DE ARQUIVOS DE SAÍDA BASEADOS NO ARQUIVO DE ENTRADA
# -----------------------------------------------------------------------------
def sanitize_file_tag(file_path):
    """Gera uma tag segura a partir do nome do arquivo de entrada.

    Regras:
      - Usa apenas o nome base (sem diretório)
      - Remove extensão .nc ou outras extensões conhecidas
      - Substitui espaços por '_' 
      - Mantém somente caracteres [A-Za-z0-9._-], demais viram '_'
      - Condensa múltiplos '_' consecutivos

    Args:
        file_path (str): Caminho completo ou nome do arquivo
    Returns:
        str: Tag sanitizada para inserir nos nomes de saída
    """
    base = os.path.basename(str(file_path))
    # remover múltiplas extensões típicas
    base = re.sub(r"(\.nc|\.txt|\.csv|\.gz|\.zip)$", "", base, flags=re.IGNORECASE)
    base = base.replace(" ", "_")
    # substituir caracteres inválidos por '_'
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    # condensar underscores
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "input"
    return base

def infer_point_name(file_path, explicit_name=None):
    """Inferir nome do ponto a partir do nome do arquivo.
    Padrões buscados: ponto<digitos>, pt<digitos>. Caso contrário usa base.
    """
    if explicit_name:
        return str(explicit_name).strip().lower()
    base = os.path.basename(str(file_path))
    base_noext = re.sub(r"\.[A-Za-z0-9]+$", "", base)
    m = re.search(r"(ponto\d+|pt\d+)", base_noext, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    simple = re.sub(r"[^A-Za-z0-9]+", "_", base_noext).strip("_").lower()
    return simple or "ponto"

# -----------------------------------------------------------------------------
# FUNÇÕES DE GERENCIAMENTO DE ARQUIVOS SAR
# -----------------------------------------------------------------------------

def find_sar_files_in_date_range(data_directory, start_date, end_date):
    """
    Encontra arquivos SAR no diretório que estão dentro do intervalo de datas especificado.
    
    Args:
        data_directory (str): Diretório contendo os arquivos SAR ou caminho para arquivo específico
        start_date (str or datetime): Data/hora inicial
        end_date (str or datetime): Data/hora final
    
    Returns:
        list: Lista de tuplas (arquivo, datetime_inicio, datetime_fim) ordenadas por data
    """
    # Converter datas para datetime se necessário
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    print(f"Procurando arquivos SAR entre {start_date} e {end_date}")
    
    # Função auxiliar para extrair período de tempo abrindo o NetCDF
    def get_time_range_from_file(nc_path):
        try:
            # Tentar grupo comum primeiro
            for grp in ["obs_params", None]:
                try:
                    ds = xr.open_dataset(nc_path, group=grp) if grp else xr.open_dataset(nc_path)
                except Exception:
                    continue
                try:
                    # Nomes candidatos de tempo (coords/vars)
                    time_candidates = [
                        "time", "TIME", "obs_time", "acquisition_time",
                        "time_center", "valid_time", "t"
                    ]
                    time_vals = None
                    for tname in time_candidates:
                        if tname in ds:
                            time_vals = ds[tname].values
                            break
                    # Se não existir vetor de tempo, tentar attrs CF
                    if time_vals is None:
                        ats = ds.attrs or {}
                        if "time_coverage_start" in ats and "time_coverage_end" in ats:
                            t0 = pd.to_datetime(ats["time_coverage_start"]) 
                            t1 = pd.to_datetime(ats["time_coverage_end"]) 
                            ds.close()
                            return t0, t1
                    if time_vals is not None and np.size(time_vals) > 0:
                        times = pd.to_datetime(time_vals)
                        t0 = pd.to_datetime(np.min(times))
                        t1 = pd.to_datetime(np.max(times))
                        ds.close()
                        return t0, t1
                finally:
                    try:
                        ds.close()
                    except Exception:
                        pass
        except Exception as e:
            print(f"  ✗ Falha ao abrir {os.path.basename(nc_path)} para obter tempo: {e}")
        return None, None

    # Verificar se data_directory é um arquivo específico
    if os.path.isfile(data_directory) and data_directory.endswith('.nc'):
        print(f"Caminho aponta para arquivo específico: {data_directory}")
        
        # Verificar se o arquivo contém dados no intervalo especificado
        file_start, file_end = get_time_range_from_file(data_directory)
        if file_start is not None and file_end is not None:
            print(f"  Arquivo contém dados de {file_start} a {file_end}")
            if (file_start <= end_date) and (file_end >= start_date):
                return [(data_directory, file_start, file_end)]
            else:
                print(f"  ✗ Arquivo fora do intervalo especificado")
                return []
        else:
            print(f"  ✗ Não foi possível determinar o período do arquivo")
            return []
    
    # Caso seja um diretório, procurar por todos NetCDF e tentar deduzir tempo
    # Primeiro usar padrão conhecido CMEMS; se não houver, cair para *.nc
    pattern = os.path.join(data_directory, "dataset-wav-sar-l3-spc-nrt-global-s1*_*T*Z_*T*Z_*.nc")
    files = glob.glob(pattern)
    if not files:
        pattern_fallback = os.path.join(data_directory, "*.nc")
        files = glob.glob(pattern_fallback)
    
    matched_files = []
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Extrair timestamps do nome do arquivo com múltiplos padrões
        # Ex.: dataset-wav-sar-l3-spc-nrt-global-s1a_20200130T090000Z_20200130T120000Z_...
        patterns = [
            r'_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)_',   # com Z
            r'_(\d{8}T\d{6})_(\d{8}T\d{6})',       # sem Z
            r'(\d{8}T\d{6}Z).*?(\d{8}T\d{6}Z)',    # qualquer lugar com Z
            r'(\d{8}T\d{6}).*?(\d{8}T\d{6})'       # qualquer lugar sem Z
        ]
        match = None
        for pat in patterns:
            match = re.search(pat, filename)
            if match:
                break

        if match:
            start_str, end_str = match.group(1), match.group(2)
            # Detectar se possui Z
            has_z = start_str.endswith('Z') and end_str.endswith('Z')
            fmt = '%Y%m%dT%H%M%SZ' if has_z else '%Y%m%dT%H%M'
            try:
                file_start = pd.to_datetime(start_str, format=fmt)
                file_end = pd.to_datetime(end_str, format=fmt)
            except Exception:
                file_start, file_end = None, None
        else:
            file_start, file_end = None, None

        # Caso não consiga pelo nome, abrir o arquivo e obter período
        if file_start is None or file_end is None:
            fs, fe = get_time_range_from_file(file_path)
            if fs is not None and fe is not None:
                file_start, file_end = fs, fe
                print(f"  (meta) {filename}: {file_start} - {file_end}")
            else:
                print(f"  ✗ Não foi possível obter período de {filename}")
                continue

        # Verificar overlap
        if (file_start <= end_date) and (file_end >= start_date):
            matched_files.append((file_path, file_start, file_end))
            print(f"  ✓ {filename}: {file_start} - {file_end}")
        else:
            print(f"  ✗ {filename}: {file_start} - {file_end} (fora do intervalo)")
    
    # Ordenar por data de início
    matched_files.sort(key=lambda x: x[1])
    
    print(f"Encontrados {len(matched_files)} arquivos SAR no intervalo especificado")
    return matched_files

def extract_datetime_from_sar_filename(file_path_or_name):
    """
    Tenta extrair datetime central do arquivo SAR a partir do nome; se falhar,
    abre o NetCDF e calcula a partir do vetor de tempo/metadados.
    
    Args:
        file_path_or_name (str): Caminho completo ou apenas o nome do arquivo SAR
        
    Returns:
        datetime or None: Data/hora central do arquivo
    """
    filename = os.path.basename(file_path_or_name)
    # Padrões com e sem 'Z'
    patterns = [
        r'_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)_',
        r'_(\d{8}T\d{6})_(\d{8}T\d{6})',
        r'(\d{8}T\d{6}Z).*?(\d{8}T\d{6}Z)',
        r'(\d{8}T\d{6}).*?(\d{8}T\d{6})'
    ]
    match = None
    for pat in patterns:
        match = re.search(pat, filename)
        if match:
            break
    if match:
        start_str, end_str = match.group(1), match.group(2)
        has_z = start_str.endswith('Z') and end_str.endswith('Z')
        fmt = '%Y%m%dT%H%M%SZ' if has_z else '%Y%m%dT%H%M'
        try:
            file_start = pd.to_datetime(start_str, format=fmt)
            file_end = pd.to_datetime(end_str, format=fmt)
            return file_start + (file_end - file_start) / 2
        except Exception:
            pass
    # Fallback: abrir arquivo e inferir tempo
    if os.path.isfile(file_path_or_name):
        try:
            for grp in ["obs_params", None]:
                try:
                    ds = xr.open_dataset(file_path_or_name, group=grp) if grp else xr.open_dataset(file_path_or_name)
                except Exception:
                    continue
                try:
                    time_candidates = [
                        "time", "TIME", "obs_time", "acquisition_time",
                        "time_center", "valid_time", "t"
                    ]
                    time_vals = None
                    for tname in time_candidates:
                        if tname in ds:
                            time_vals = ds[tname].values
                            break
                    if time_vals is not None and np.size(time_vals) > 0:
                        times = pd.to_datetime(time_vals)
                        t0 = pd.to_datetime(np.min(times))
                        t1 = pd.to_datetime(np.max(times))
                        return t0 + (t1 - t0) / 2
                    ats = ds.attrs or {}
                    if "time_coverage_start" in ats and "time_coverage_end" in ats:
                        t0 = pd.to_datetime(ats["time_coverage_start"]) 
                        t1 = pd.to_datetime(ats["time_coverage_end"]) 
                        return t0 + (t1 - t0) / 2
                finally:
                    try:
                        ds.close()
                    except Exception:
                        pass
        except Exception:
            pass
    return None

# -----------------------------------------------------------------------------
# FUNÇÕES DE CARREGAMENTO DE ESPECTROS
# -----------------------------------------------------------------------------

def open_dataset(file_path, source_type="ww3"):
    """Abre arquivo de dados e retorna o dataset."""
    if source_type == "sar":
        # Tentar primeiro com grupo obs_params
        try:
            ds = xr.open_dataset(file_path, group="obs_params")
            if 'time' in ds:
                print(f"Período disponível: {ds.time.values[0]} a {ds.time.values[-1]}")
                print(f"Total de registros: {len(ds.time)}")
            else:
                print("Arquivo SAR com registro único")
        except (OSError, KeyError):
            # Se falhar, tentar abrir sem grupo
            print("Grupo 'obs_params' não encontrado, abrindo arquivo diretamente...")
            ds = xr.open_dataset(file_path)
            if 'time' in ds:
                print(f"Período disponível: {ds.time.values[0]} a {ds.time.values[-1]}")
                print(f"Total de registros: {len(ds.time)}")
            else:
                print("Arquivo SAR com registro único")
    else:
        ds = xr.open_dataset(file_path)
        print(f"Período disponível: {ds.time.values[0]} a {ds.time.values[-1]}")
        print(f"Total de registros: {len(ds.time)}")
    return ds

def load_sar_spectrum(ds, date_time=None, index=0, output_unit="m2_Hz_rad", scaling_factor=1.0):
    """Carrega espectro SAR para data/hora específica, compatível com arquivos preprocessados Sentinel-1A/B (CMEMS)."""
    print("Variáveis disponíveis no arquivo SAR:", list(ds.variables.keys()))
    # Busca variável por múltiplos nomes possíveis
    def get_var(ds, varnames):
        for var in varnames:
            if var in ds.variables:
                return ds[var].values
        raise ValueError(f"Nenhuma das variáveis {varnames} encontrada no arquivo SAR.")

    # Nomes possíveis para cada variável
    wave_spec_names = ['wave_spec', 'obs_params/wave_spec', 'wave_spectrum', 'obs_params/wave_spectrum']
    k_names = ['wavenumber_spec', 'obs_params/wavenumber_spec']
    phi_names = ['direction_spec', 'obs_params/direction_spec']
    time_names = ['time', 'obs_params/time', 'TIME', 'obs_time', 'acquisition_time', 'time_center', 'valid_time', 't']

    try:
        E_sar = get_var(ds, wave_spec_names)  # (NF, ND, Nobs)
        k = get_var(ds, k_names)  # (NF,)
        phi = get_var(ds, phi_names)  # (ND,)
        # Tempo
        times = None
        for tname in time_names:
            if tname in ds.variables:
                times = ds[tname].values
                break
        nobs = E_sar.shape[2] if E_sar.ndim == 3 else 1
        if nobs > 1:
            if date_time is not None and times is not None:
                if isinstance(date_time, str):
                    date_time = pd.to_datetime(date_time)
                idx = abs(times - np.datetime64(date_time)).argmin()
                actual_time = pd.to_datetime(times[idx])
            else:
                idx = index
                actual_time = pd.to_datetime(times[idx]) if times is not None else None
            E_sar = np.squeeze(E_sar[:, :, idx])
        else:
            E_sar = np.squeeze(E_sar)
            actual_time = pd.to_datetime(times[0]) if times is not None else None
        print(f"Usando arquivo preprocessado (CMEMS), shape E_sar: {E_sar.shape}")
    except Exception as e:
        # Tenta formato antigo
        if 'oswPolSpec' in ds.variables:
            if 'time' in ds and len(ds.time) > 1:
                if date_time is not None:
                    if isinstance(date_time, str):
                        date_time = pd.to_datetime(date_time)
                    idx = abs(ds.time.values - np.datetime64(date_time)).argmin()
                    actual_time = pd.to_datetime(ds.time.values[idx])
                else:
                    idx = index
                    actual_time = pd.to_datetime(ds.time.values[idx])
                E_sar = np.squeeze(ds.oswPolSpec.values[idx])
            else:
                E_sar = np.squeeze(ds.oswPolSpec.values)
                if 'time' in ds:
                    timestamp = ds.time.values[0]
                    actual_time = pd.to_datetime(timestamp)
                else:
                    actual_time = None
            k = ds.oswK.values
            phi = ds.oswPhi.values
            print(f"Usando arquivo SAR antigo, shape E_sar: {E_sar.shape}")
        else:
            raise ValueError("Arquivo SAR não possui variáveis reconhecidas (nem wave_spec nem oswPolSpec)")

    print(f"Shape k: {k.shape}")
    print(f"Shape phi: {phi.shape}")
    # Converter SAR para unidades de energia convencionais (já em m²/Hz/rad)
    E2d, freq, dirs, dirs_rad = convert_sar_energy_units(E_sar, k, phi, scaling_factor)
    # Converter para unidade desejada se diferente
    if output_unit != "m2_Hz_rad":
        E2d = convert_spectrum_units(E2d, freq, dirs, "m2_Hz_rad", output_unit)
    return E2d, freq, dirs, dirs_rad, actual_time

def load_ww3_spectrum(ds, date_time=None, index=0, station_idx=0, output_unit="m2_Hz_rad"):
    """Carrega espectro do WW3 para data/hora específica (cópia leve do ww3_part)."""
    if date_time is not None:
        if isinstance(date_time, str):
            date_time = pd.to_datetime(date_time)
        idx = abs(ds.time.values - np.datetime64(date_time)).argmin()
        actual_time = pd.to_datetime(ds.time.values[idx])
    else:
        idx = index
        actual_time = pd.to_datetime(ds.time.values[idx])

    E2d = ds.efth[idx, station_idx, :, :].values  # m²·s·rad⁻¹
    E2d = np.where(np.abs(E2d) > 1e30, 0, E2d)
    E2d = np.maximum(E2d, 0)
    freq = ds.frequency.values
    dirs = ds.direction.values
    dirs_rad = np.radians(dirs)
    return E2d, freq, dirs, dirs_rad, actual_time

def load_ndbc_spectrum(ds, date_time=None, index=0):
    """Carrega espectro do NDBC para data/hora específica (cópia leve do ww3_part)."""
    if date_time is not None:
        if isinstance(date_time, str):
            date_time = pd.to_datetime(date_time)
        idx = abs(ds.time.values - np.datetime64(date_time)).argmin()
        actual_time = pd.to_datetime(ds.time.values[idx])
    else:
        idx = index
        actual_time = pd.to_datetime(ds.time.values[idx])
    spec_1d = ds['spectral_wave_density'].isel(time=idx).values
    r1 = ds['wave_spectrum_r1'].isel(time=idx).values
    r2 = ds['wave_spectrum_r2'].isel(time=idx).values
    alpha1 = ds['mean_wave_dir'].isel(time=idx).values
    alpha2 = ds['principal_wave_dir'].isel(time=idx).values
    freq = ds['frequency'].values
    theta_deg = np.arange(0, 360, 15)
    theta_rad = np.radians(theta_deg)
    E2d = np.zeros((len(freq), len(theta_deg)))
    for i, f in enumerate(freq):
        a1 = r1[i] * np.cos(np.radians(alpha1[i]))
        b1 = r1[i] * np.sin(np.radians(alpha1[i]))
        a2 = r2[i] * np.cos(np.radians(alpha2[i]))
        b2 = r2[i] * np.sin(np.radians(alpha2[i]))
        spread = (1/(2*np.pi)) * (
            1 + 2*a1*np.cos(theta_rad) + 2*b1*np.sin(theta_rad) +
            2*a2*np.cos(2*theta_rad) + 2*b2*np.sin(2*theta_rad)
        )
        E2d[i, :] = spec_1d[i] * spread
    return E2d, freq, theta_deg, theta_rad, actual_time

def load_spectrum_from_txt(filepath, source_type="ww3"):
    """Carrega espectro 2D a partir de arquivo texto."""
    E = np.loadtxt(filepath, delimiter=',')
    NF, ND = E.shape

    # Verificar e corrigir shape se necessário
    if source_type == "ww3" and E.shape != (NF, ND):
        print(f"Aviso: E tem shape {E.shape}, esperado ({NF}, {ND})")
        if E.shape == (ND, NF):
            E = E.T
            print("Transpondo E para corresponder ao shape esperado.")
        else:
            raise ValueError("Formato de dados não corresponde ao esperado.")

    print(f"Shape de E: {E.shape}")
    return E, NF, ND

def convert_spectrum_units(E2d, freq, dirs, from_unit, to_unit):
    """Converte espectro entre diferentes unidades de energia."""
    # Se unidades forem iguais, retornar cópia do original
    if from_unit == to_unit:
        return E2d.copy()
    
    result = E2d.copy()
    
    # Conversões entre unidades
    if from_unit == "m2_s_rad" and to_unit == "m2_Hz_rad":
        result = result / (2 * np.pi)
    elif from_unit == "m2_Hz_rad" and to_unit == "m2_s_rad":
        result = result * (2 * np.pi)
    elif from_unit == "m2_s_rad" and to_unit == "m2_Hz_deg":
        result = result / (2 * np.pi) * (180 / np.pi)
    elif from_unit == "m2_Hz_rad" and to_unit == "m2_Hz_deg":
        result = result * (180 / np.pi)
    
    return result

def get_unit_label(unit):
    """Retorna rótulo formatado para a unidade do espectro."""
    unit_labels = {
        "m2_s_rad": "m²·s·rad⁻¹",
        "m2_Hz_rad": "m²/Hz/rad",
        "m2_s_deg": "m²·s·deg⁻¹",
        "m2_Hz_deg": "m²/Hz/deg",
        "m2_Hz": "m²/Hz",
        "m2_s": "m²·s"
    }
    return unit_labels.get(unit, unit)

# -----------------------------------------------------------------------------
# FUNÇÕES DE CÁLCULO DE PARÂMETROS DE ONDAS
# -----------------------------------------------------------------------------

def spectrum1d_from_2d(E2d, dirs_rad):
    """Integra espectro 2D para obter espectro 1D E(f)."""
    E2d = np.where(np.isfinite(E2d) & (E2d >= 0), E2d, 0)
    ddir = 2 * np.pi / len(dirs_rad)
    spec1d = np.sum(E2d, axis=1) * ddir
    return spec1d, ddir

def calculate_wave_parameters(E2d, freq, dirs_rad):
    """Calcula Hs, Tp, Dp e outros parâmetros do espectro."""
    # Integrar espectro 2D para obter 1D
    spec1d, ddir = spectrum1d_from_2d(E2d, dirs_rad)
    
    # Calcular incrementos de frequência
    delf = np.zeros_like(freq)
    for i in range(len(freq)-1):
        delf[i] = freq[i+1] - freq[i]
    delf[-1] = delf[-2]
    
    # Calcular momento espectral m0
    m0 = np.sum(spec1d * delf)
    
    # Calcular altura significativa
    hs = 4 * np.sqrt(m0) if m0 > 0 else 0.0
    
    # Encontrar pico no espectro de frequência
    i_peak = np.argmax(spec1d) if np.max(spec1d) > 0 else 0
    tp = 1.0 / freq[i_peak] if i_peak < len(freq) and freq[i_peak] > 0 else np.nan
    
    # Encontrar direção no pico
    j_peak = np.argmax(E2d[i_peak, :]) if i_peak < len(freq) else 0
    
    # Calcular direção média ponderada no pico
    if np.any(E2d[i_peak, :] > 0):
        weighted_dir = np.sum(E2d[i_peak, :] * dirs_rad) / np.sum(E2d[i_peak, :])
        dp = np.degrees(weighted_dir) % 360
    else:
        dp = np.nan
    
    return hs, tp, dp, m0, delf, ddir, i_peak, j_peak

# -----------------------------------------------------------------------------
# FUNÇÕES DE PARTICIONAMENTO DE ESPECTRO
# -----------------------------------------------------------------------------

def identify_spectral_peaks(E, NF, ND, energy_threshold=0.05, max_partitions=5, source_type="ww3"):
    """Identifica picos espectrais na matriz de energia E."""
    print(f"Identificando picos espectrais com threshold: {energy_threshold:.2e}")
    print(f"Valores do espectro: min={np.min(E):.2e}, max={np.max(E):.2e}, mean={np.mean(E):.2e}")

    ICOD = np.zeros((NF, ND), dtype=int)
    peaks_list = []
    
    # Calcular threshold relativo para SAR - utiliza um percentual do valor máximo
    if source_type == "sar":
        e_max = np.max(E)
        relative_threshold = CONFIG.get("relative_threshold", 0.05) * e_max
        print(f"Valor máximo de energia: {e_max:.2e}")
        print(f"Usando threshold relativo: {relative_threshold:.2e} ({CONFIG.get('relative_threshold', 0.05)*100}% do máximo)")
        # Usar o maior dos dois thresholds (absoluto ou relativo)
        if relative_threshold > energy_threshold:
            energy_threshold = relative_threshold
            print(f"Threshold relativo adotado: {energy_threshold:.2e}")
    
    # Para cada ponto no espectro
    for II in range(NF):
        for JJ in range(ND):
            if E[II, JJ] < 1e-15:  # Valor muito pequeno para ignorar
                continue
                
            # Verificar vizinhança 3x3
            RMAX = 0
            IX = 2
            JY = 2
            i_range = [max(0, II-1), II, min(NF-1, II+1)]
            j_range = [(JJ-1) % ND, JJ, (JJ+1) % ND]
            
            # Encontrar o maior vizinho
            for i_idx, I in enumerate(i_range):
                for j_idx, J in enumerate(j_range):
                    if i_idx == 1 and j_idx == 1:  # Pular o próprio ponto
                        continue
                    RT = E[I, J] - E[II, JJ]
                    if RT > RMAX:
                        IX = i_idx + 1
                        JY = j_idx + 1
                        RMAX = RT
            
            # Atribuir código de direção
            ICOD[II, JJ] = JY * 10 + IX
            
            # Considerar como pico apenas se energia >= limiar
            if ICOD[II, JJ] == 22 and E[II, JJ] >= energy_threshold:
                peaks_list.append((II+1, JJ+1, E[II, JJ]))  # Adicionar valor de energia como terceira coluna
    
    # Ordenar picos por energia (maior para menor)
    peaks_list.sort(key=lambda x: x[2], reverse=True)
    
    # Limitar número de picos para max_partitions
    if len(peaks_list) > max_partitions:
        print(f"Limitando número de picos de {len(peaks_list)} para {max_partitions}")
        peaks_list = peaks_list[:max_partitions]
    
    # Remover energia da lista final de picos
    peaks_list = [(p[0], p[1]) for p in peaks_list]
    
    # Criar máscara inicial com os picos
    nmask = len(peaks_list)
    peaks = np.array(peaks_list) if nmask > 0 else np.empty((0, 2))
    MASK = np.zeros((NF, ND), dtype=int)
    
    for im in range(nmask):
        ii = int(peaks[im, 0]) - 1
        jj = int(peaks[im, 1]) - 1
        MASK[ii, jj] = im + 1
    
    print(f"Identificados {nmask} picos espectrais")
    
    return ICOD, MASK, peaks, nmask

def generate_mask(ICOD, MASK, NF, ND):
    """Gera máscara a partir dos códigos ICOD e picos identificados."""
    mask_copy = MASK.copy()

    print("Gerando máscara a partir do ICOD...")
    # Primeira passagem - propagar valores usando direções ICOD
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
    
    # Segunda passagem - tratar zeros restantes
    while 0 in mask_copy:
        zero_changed = False
        for i in range(NF):
            for j in range(ND):
                if mask_copy[i, j] == 0:
                    # Verificar 8 células vizinhas
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni = i + di
                            nj = (j + dj) % ND  # Tratar direção circular
                            if 0 <= ni < NF:
                                if mask_copy[ni, nj] > 0:
                                    neighbors.append(mask_copy[ni, nj])
                    
                    if neighbors:
                        # Atribuir valor vizinho mais comum
                        most_common = Counter(neighbors).most_common(1)[0][0]
                        mask_copy[i, j] = most_common
                        zero_changed = True
        
        if not zero_changed:
            # Se nenhum zero foi alterado, atribuir valor padrão
            for i in range(NF):
                for j in range(ND):
                    if mask_copy[i, j] == 0:
                        mask_copy[i, j] = 1  # Valor padrão
            break
            
    return mask_copy

def calculate_peak_distances(peaks, frequencies, directions_rad, nmask):
    """Calcula distâncias entre picos espectrais."""
    print("Calculando distâncias entre picos...")
    if nmask == 0:
        return np.zeros((0, 0))
    
    # Converter de índices 1-based para 0-based
    i_indices = (peaks[:, 0] - 1).astype(int)
    j_indices = (peaks[:, 1] - 1).astype(int)
    
    # Obter frequências e direções para todos os picos
    freqs = frequencies[i_indices]
    dirs = directions_rad[j_indices]
    
    # Calcular coordenadas x, y para todos os picos
    x_coords = freqs[:, np.newaxis] * np.cos(dirs)[:, np.newaxis]
    y_coords = freqs[:, np.newaxis] * np.sin(dirs)[:, np.newaxis]
    
    # Calcular distâncias ao quadrado
    dx = x_coords - x_coords.T
    dy = y_coords - y_coords.T
    
    return dx**2 + dy**2

def calculate_peak_spreading(E, MASK, frequencies, directions_rad, NF, ND, nmask, Etot, delf, ddir):
    """Calcula o espalhamento de pico individual."""
    print("Calculando espalhamento de pico...")
    if nmask == 0:
        return np.zeros(0)
    
    # Criar meshgrids de frequência e direção
    freq_grid, dir_grid = np.meshgrid(frequencies, directions_rad, indexing='ij')
    
    # Pré-computar termos comuns
    cos_dir = np.cos(dir_grid)
    sin_dir = np.sin(dir_grid)
    freq_cos = freq_grid * cos_dir
    freq_sin = freq_grid * sin_dir
    freq2_cos2 = freq_grid**2 * cos_dir**2
    freq2_sin2 = freq_grid**2 * sin_dir**2
    
    # Criar grade de pesos
    delf_grid = delf[:, np.newaxis] * np.ones(ND)
    weights = E * delf_grid * ddir
    
    # Inicializar arrays
    fx = np.zeros(nmask + 2)
    fy = np.zeros(nmask + 2)
    fxx = np.zeros(nmask + 2)
    fyy = np.zeros(nmask + 2)
    
    # Calcular valores para cada partição
    for idx in range(1, nmask + 1):
        mask = (MASK == idx)
        fx[idx] = np.sum(weights[mask] * freq_cos[mask])
        fy[idx] = np.sum(weights[mask] * freq_sin[mask])
        fxx[idx] = np.sum(weights[mask] * freq2_cos2[mask])
        fyy[idx] = np.sum(weights[mask] * freq2_sin2[mask])
    
    # Calcular espalhamento
    Eip = np.zeros(nmask)
    for i in range(nmask):
        idx = i + 1  # Ajustar para indexação 1-based
        Eip[i] = fxx[idx]/Etot - (fx[idx]/Etot)**2 + fyy[idx]/Etot - (fy[idx]/Etot)**2
    
    return Eip

def merge_overlapping_systems(MASK, dist, Eip, peaks, nmask):
    """Mescla sistemas de ondas sobrepostos."""
    print("Verificando sistemas sobrepostos...")
    print(f"Número de máscaras: {nmask}")

    if nmask <= 1:
        return MASK.copy()
    
    M = MASK.copy()
    
    # Criar matrizes de limiar para comparação
    thresholds_i = Eip * 0.5
    
    # Encontrar pares para mesclar
    for i in range(nmask):
        for j in range(i+1, nmask):
            # Verificar se os sistemas devem ser mesclados
            if dist[i, j] <= thresholds_i[i] and dist[i, j] <= thresholds_i[j]:
                print(f"Distância {dist[i, j]} <= Limiares ({thresholds_i[i]}, {thresholds_i[j]})")
                print("Unindo sistemas!")
                
                # Obter índices para picos i e j
                i_idx = int(peaks[i, 0]) - 1
                j_i_idx = int(peaks[i, 1]) - 1
                j_idx = int(peaks[j, 0]) - 1
                j_j_idx = int(peaks[j, 1]) - 1
                
                # Atualizar máscara para mesclar sistemas
                system_j_mask = (M == j + 1)
                M[system_j_mask] = i + 1
                M[j_idx, j_j_idx] = i + 1
    
    return M

def calculate_partitioned_energy(E, M, delf, ddir, NF, ND, nmask):
    """Calcula energia e altura significativa por partição."""
    e = np.zeros(nmask + 2)

    for i in range(NF):
        for j in range(ND):
            mask_idx = M[i, j]
            e[mask_idx] += E[i, j] * delf[i] * ddir

    # Debug: soma das energias das partições
    print(f"[DEBUG] Soma das energias das partições: {np.sum(e):.6f}")
    print(f"[DEBUG] Total esperado: {np.sum(E * np.tile(delf[:, np.newaxis], (1, ND)) * ddir):.6f}")
    
    Hs = 4 * np.sqrt(e)  # altura significativa por partição
    return e, Hs

def renumber_partitions_by_energy(mask, Hs, e=None):
    """Renumera partições por energia (maior energia = #1)."""
    unique_partitions = sorted([p for p in np.unique(mask) if p > 0 and p < len(Hs)])
    if len(unique_partitions) == 0:
        if e is not None:
            return mask.copy(), Hs.copy(), e.copy()
        else:
            return mask.copy(), Hs.copy()
    
    # Obter energias para cada partição
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
    """Calcula período de pico (Tp) e direção (Dp) para cada partição usando a mesma metodologia do total."""
    Tp = np.full(nmask + 2, np.nan)
    Dp = np.full(nmask + 2, np.nan)

    for idx in range(1, nmask + 1):
        # Máscara para esta partição
        partition_mask = (mask == idx)
        if not np.any(partition_mask):
            continue
        
        # Espectro 1D para esta partição
        E_part = np.zeros_like(E)
        E_part[partition_mask] = E[partition_mask]
        spec1d, _ = spectrum1d_from_2d(E_part, directions_rad)
        
        # Encontrar pico no espectro de frequência
        i_peak = np.argmax(spec1d) if np.max(spec1d) > 0 else 0
        if frequencies[i_peak] > 0:
            Tp[idx] = 1.0 / frequencies[i_peak]
        
        # Encontrar direção no pico
        if np.any(E_part[i_peak, :] > 0):
            weighted_dir = np.sum(E_part[i_peak, :] * directions_rad) / np.sum(E_part[i_peak, :])
            Dp[idx] = np.degrees(weighted_dir) % 360
    
    return Tp, Dp

def calculate_spectral_moments(E, mask, freq, dirs_rad, delf, ddir, partition_idx=None):
    """Calcula momentos espectrais (m0, m1, m2) para uma partição ou todo o espectro."""
    # Se partition_idx for None, calcula para todo o espectro
    if partition_idx is not None:
        # Criar uma máscara para esta partição
        E_mask = np.zeros_like(E)
        E_mask[mask == partition_idx] = E[mask == partition_idx]
        E_calc = E_mask
    else:
        E_calc = E
    
    # Obter espectro 1D
    spec1d, _ = spectrum1d_from_2d(E_calc, dirs_rad)
    
    # Calcular momentos espectrais
    m0 = 0.0
    m1 = 0.0
    m2 = 0.0
    
    for i in range(len(freq)):
        if freq[i] > 0:  # Evitar divisão por zero
            omega = 2 * np.pi * freq[i]  # Converter para frequência angular (rad/s)
            m0 += spec1d[i] * delf[i]
            m1 += omega * spec1d[i] * delf[i]
            m2 += (omega**2) * spec1d[i] * delf[i]
    
    return m0, m1, m2

def partition_spectrum(E, frequencies, directions_rad, energy_threshold=0.05, source_type="ww3"):
    """Executa o processo completo de particionamento de espectros."""
    NF, ND = E.shape
    max_partitions = CONFIG.get("max_partitions", 10) if source_type == "sar" else 20
    ICOD, MASK, peaks, nmask = identify_spectral_peaks(
        E, NF, ND, energy_threshold, max_partitions, source_type
    )
    if nmask == 0:
        print("Nenhum pico espectral identificado.")
        return None
    MASK = generate_mask(ICOD, MASK, NF, ND)
    # Garante alinhamento antes de seguir
    if E.shape != MASK.shape:
        print(f"[DEBUG] Corrigindo shape: E{E.shape} vs MASK{MASK.shape}")
        if E.shape == MASK.T.shape:
            MASK = MASK.T
        else:
            raise ValueError(f"Shape incompatível: E{E.shape} vs MASK{MASK.shape}")
    distances = calculate_peak_distances(peaks, frequencies, directions_rad, nmask)
    hs, tp, dp, m0, delf, ddir, _, _ = calculate_wave_parameters(E, frequencies, directions_rad)
    Eip = calculate_peak_spreading(E, MASK, frequencies, directions_rad, NF, ND, nmask, m0, delf, ddir)
    MASK = merge_overlapping_systems(MASK, distances, Eip, peaks, nmask)
    
    # Usar a função de cálculo de energia corrigida
    e, Hs = calculate_partitioned_energy(E, MASK, delf, ddir, NF, ND, nmask)
    
    # Verificar soma de energias
    e_total = np.sum(e)
    print(f"Energia total do espectro: {m0:.6f}")
    print(f"Soma das energias particionadas: {e_total:.6f}")
    if abs(e_total - m0) > 1e-4:
        print(f"AVISO: Discrepância na energia total: {abs(e_total - m0):.6f}")
    
    # Renumerar partições por energia - DEVE SER FEITO ANTES de calcular momentos espectrais
    M_renumbered, Hs_renumbered, e_renumbered = renumber_partitions_by_energy(MASK, Hs, e)
    
    # Use a função corrigida para Tp e Dp das partições:
    Tp, Dp = calculate_peak_parameters(E, M_renumbered, frequencies, directions_rad, NF, ND, nmask, delf, ddir)
    
    # Calcular momentos espectrais para o espectro total - AGORA após renumerar
    m0_total, m1_total, m2_total = calculate_spectral_moments(E, None, frequencies, directions_rad, delf, ddir)
    
    # Calcular momentos espectrais para cada partição - USANDO M_renumbered
    m0_parts = np.zeros(nmask + 2)
    m1_parts = np.zeros(nmask + 2)
    m2_parts = np.zeros(nmask + 2)
    for idx in range(nmask + 2):
        if idx <= nmask or idx == 0:  # Calcular para cada partição e para os não classificados (0)
            m0_parts[idx], m1_parts[idx], m2_parts[idx] = calculate_spectral_moments(
                E, M_renumbered, frequencies, directions_rad, delf, ddir, idx
            )
    
    # Criar dicionário de resultados
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
        # Adicionar momentos espectrais
        "moments": {
            "total": (m0_total, m1_total, m2_total),
            "m0": m0_parts,
            "m1": m1_parts,
            "m2": m2_parts
        }
    }
    
    return results

# -----------------------------------------------------------------------------
# FUNÇÕES DE PLOTAGEM
# -----------------------------------------------------------------------------

def plot_spectrum_1d(spec1d, freq, selected_time, hs, tp, source_type, 
                     save_dir=None, show_plot=True, spectrum_unit="m2_s", file_tag=None,
                     max_density=25.0, point_name=None, lon=None, lat=None, index=None):
    """Plota espectro 1D padronizado (freq <=0.35Hz, y 0..max_density, unidade m²·s)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq, spec1d, 'b-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(f'Spectral Density ({get_unit_label("m2_s")})')
    ax.grid(True)
    
    # Adicionar marcador de frequência de pico
    if not np.isnan(tp) and tp > 0:
        peak_freq = 1.0 / tp
        peak_density = np.interp(peak_freq, freq, spec1d)
        ax.plot(peak_freq, peak_density, 'ro', markersize=8,
               label=f'Peak (f={peak_freq:.3f} Hz, Tp={tp:.1f} s)')
        ax.legend()
    
    # Configurar título com coordenadas se disponíveis
    title_parts = [f'{source_type.upper()} 1D Spectrum - {selected_time.strftime("%d-%m-%Y %H:%M")}']
    if lon is not None and lat is not None:
        title_parts.append(f'Lon: {lon:.3f}° | Lat: {lat:.3f}°')
    title_parts.append(f'Hs: {hs:.2f} m | Tp: {tp:.1f} s')
    ax.set_title('\n'.join(title_parts))
    
    # Configurar eixo x para mostrar tanto frequência quanto período
    fmax = 0.35
    ax.set_xlim(min(freq), fmax)
    ax.set_xticks(np.arange(0, fmax+1e-6, 0.05))
    ax.set_xticklabels([f"{v:.2f}" for v in np.arange(0, fmax+1e-6, 0.05)])
    
    # Adicionar segundo eixo x com valores de período
    ax2 = ax.twiny()
    period_ticks = [2, 4, 6, 8, 10, 15, 20]
    freq_ticks = [1/p for p in period_ticks]
    ax2.set_xticks(freq_ticks)
    ax2.set_xticklabels([f"{p}" for p in period_ticks])
    ax2.set_xlabel('Period (s)')
    ax2.set_xlim(ax.get_xlim())
    
    # Limitar eixo Y e margens
    ax.set_ylim(0, max_density)
    ax.margins(x=0)
    ax.set_ybound(lower=0)

    # Salvar e mostrar
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        date_str = selected_time.strftime("%Y%m%d_%H%M")
        # Usar point_name diretamente (já estará no formato sar{ref} se ref foi fornecido)
        base_filename = f"{point_name}_{date_str}"
        fig_path = os.path.join(save_dir, f"{base_filename}_spectrum_1D.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Figura salva em: {fig_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_spectrum_2d_enhanced(
    E2d, freq, dirs, selected_time, hs, tp, dp, source_type, 
    save_dir=None, show_plot=True, spectrum_unit="m2_Hz_rad",
    plot_type="contour", vmin=None, vmax=None, normalize_by_hs=False, file_tag=None,
    lon=None, lat=None, index=None, point_name=None
):
    """Plota espectro 2D com visual aprimorado."""
    # Converter direções para radianos
    dirs_rad = np.radians(dirs)
    
    # Criar meshgrid para plot polar - usar período ao invés de frequência
    r_rad = np.zeros_like(freq)
    for i, f in enumerate(freq):
        if f > 0:
            r_rad[i] = 1.0 / f  # Converter para período
    
    # Corrigir criação do meshgrid - usar indexing='ij' consistente
    theta_mesh, r_mesh = np.meshgrid(dirs_rad, r_rad, indexing='ij')
    
    # Verificar e corrigir dimensões se necessário
    print(f"Shape do espectro E2d: {E2d.shape}")
    print(f"Shape do theta_mesh: {theta_mesh.shape}")
    print(f"Shape do r_mesh: {r_mesh.shape}")
    
    # Garantir que as dimensões estejam corretas
    if E2d.shape != theta_mesh.shape:
        if E2d.shape == theta_mesh.T.shape:
            E2d = E2d.T
            print(f"Transpondo E2d para shape: {E2d.shape}")
        else:
            # Recriar meshgrid com dimensões corretas
            if E2d.shape[0] == len(dirs_rad) and E2d.shape[1] == len(r_rad):
                theta_mesh, r_mesh = np.meshgrid(dirs_rad, r_rad, indexing='xy')
                print(f"Recriando meshgrid com indexing='xy': {theta_mesh.shape}")
            else:
                theta_mesh, r_mesh = np.meshgrid(dirs_rad, r_rad, indexing='ij')
                E2d = E2d.T
                print(f"Recriando meshgrid com indexing='ij' e transpondo E2d: {E2d.shape}")
    
    # Configurar figura com espaço extra para estatísticas
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, projection='polar')
    
    # Ajustar a posição do gráfico principal para dar espaço às estatísticas
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.75, pos.height])
    
    # Normalização opcional
    if normalize_by_hs and hs > 0:
        E2d = E2d / (hs**2)
    
    # Determinar limites de cor caso não definidos
    finite_vals = E2d[np.isfinite(E2d)]
    if vmin is None:
        vmin = np.nanpercentile(finite_vals, 30) if finite_vals.size else 0.0
    if vmax is None:
        vmax = np.nanpercentile(finite_vals, 99) if finite_vals.size else 1.0
    if vmax <= vmin:
        vmax = vmin * 1.1 if vmin > 0 else 1.0

    # Máscara valores abaixo do mínimo para não “sumir” tudo
    E2d_plot = E2d.copy()
    E2d_plot = np.where(E2d_plot >= vmin, E2d_plot, np.nan)
    
    # Colormap
    cmap = plt.get_cmap('twilight_shifted')
    
    # Níveis fixos
    e_min, e_max = vmin, vmax
    # Se a faixa for muito estreita, gere ao menos alguns níveis
    levels = np.linspace(e_min, e_max, 40)
    
    # Plot contornos de acordo com o tipo selecionado
    if plot_type in ["contour", "both"]:
        cs = ax.contour(theta_mesh, r_mesh, E2d_plot, levels=levels, cmap=cmap,
                      vmin=e_min, vmax=e_max, linewidths=0.8)
    
    if plot_type in ["filled", "both"]:
        cs_filled = ax.contourf(theta_mesh, r_mesh, E2d_plot, levels=levels, cmap=cmap,
                               vmin=e_min, vmax=e_max, alpha=0.7)
    
    # Configurar eixos
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # sentido horário
    
    # Configurar eixo radial para mostrar períodos
    periods = [5, 10, 15, 20]
    ax.set_rticks(periods)
    ax.set_yticklabels([f'{p}s' for p in periods], color='gray', fontsize=7.5)
    ax.set_rlim(0, 25)
    ax.set_rlabel_position(30)
    ax.tick_params(axis='y', colors='gray', labelsize=16)
    
    # Configurar eixo angular para direções cardeais
    ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.tick_params(axis='x', colors='k', labelsize=16)
    ax.set_xticklabels(ticks)
    
    # Configurar título com posição mais baixa
    ax.set_title('Directional Spectrum', fontsize=14, color='k', pad=20)
    
    # Criar caixa de estatísticas ajustada para não sobrepor o título
    stats_ax = fig.add_axes([0.78, 0.6, 0.18, 0.25], facecolor='white')
    stats_ax.patch.set_alpha(0.8)
    stats_ax.patch.set_edgecolor('black')
    stats_ax.patch.set_linewidth(1.5)
    stats_ax.axis('off')  # Ocultar eixos
    
    # Adicionar título e estatísticas na caixa com melhor espaçamento
    stats_ax.text(0.5, 0.95, 'Statistics',
                fontsize=12, color='k',
                ha='center', va='top', weight='bold',
                transform=stats_ax.transAxes)
    
    # Formatar timestamp
    if isinstance(selected_time, np.datetime64):
        selected_time = pd.to_datetime(selected_time)
    date_str = selected_time.strftime('%Y-%m-%d %H:%M')
    stats_ax.text(0.5, 0.85, f"Date: {date_str}",
               fontsize=10, color='k',
               ha='center', va='top',
               transform=stats_ax.transAxes)
    
    # Adicionar coordenadas se disponíveis
    if lon is not None and lat is not None:
        stats_ax.text(0.5, 0.7, f'Lon: {lon:.3f}°',
                   fontsize=10, color='k',
                   ha='center', va='top',
                   transform=stats_ax.transAxes)
        stats_ax.text(0.5, 0.6, f'Lat: {lat:.3f}°',
                   fontsize=10, color='k',
                   ha='center', va='top',
                   transform=stats_ax.transAxes)
        y_pos = 0.5
    else:
        y_pos = 0.7
    
    # Adicionar parâmetros de onda
    stats_ax.text(0.5, y_pos, f'Hs: {hs:.2f} m',
               fontsize=10, color='k',
               ha='center', va='top',
               transform=stats_ax.transAxes)
    stats_ax.text(0.5, y_pos-0.1, f'Tp: {tp:.1f} s',
               fontsize=10, color='k',
               ha='center', va='top',
               transform=stats_ax.transAxes)
    stats_ax.text(0.5, y_pos-0.2, f'Dp: {dp:.1f}°',
               fontsize=10, color='k',
               ha='center', va='top',
               transform=stats_ax.transAxes)
    
    # Adicionar colorbar
    norm = plt.Normalize(vmin=e_min, vmax=e_max)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, fraction=0.025, pad=0.05, ax=ax, extend='both')
    cbar.set_label(f'Energy ({get_unit_label(spectrum_unit)})', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(levels[::4])
    
    # Usar subplots_adjust ao invés de tight_layout para melhor controle
    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
    
    # Salvar figura
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        time_str = selected_time.strftime("%Y%m%d_%H%M")
        tag_part = f"{file_tag}_" if file_tag else ""
        indices_str = None
        try:
            import sys
            args = None
            for a in sys.argv:
                if '--obs-indices' in a:
                    indices_str = a.split('--obs-indices')[1].strip()
                    break
            if indices_str is None:
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument('--obs-indices', nargs='+', type=int)
                args, _ = parser.parse_known_args()
                if args.obs_indices:
                    indices_str = '_'.join(str(i) for i in args.obs_indices)
        except Exception:
            indices_str = None
        
        # Usar point_name se fornecido, caso contrário usar file_tag
        date_str = selected_time.strftime("%Y%m%d_%H%M")
        if point_name:
            base_filename = f"{point_name}_{date_str}"
        elif file_tag:
            base_filename = f"{file_tag}_{date_str}"
        else:
            base_filename = f"{source_type}_{date_str}"
        
        fig_path = os.path.join(save_dir, f"{base_filename}_spectrum_2D.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figura salva em: {fig_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return plt.gcf()

def plot_partition_mask(mask, E2d, freq, dirs, selected_time, source_type, 
                       save_dir=None, show_plot=True, file_tag=None, index=None):
    """Plota a máscara de partição com cores para cada partição."""
    # Converter para convenção oceanográfica (para onde vai)
    dirs_rad = np.radians(dirs)
    
    # Criar meshgrid para plot polar - usar período ao invés de frequência
    r_rad = np.zeros_like(freq)
    for i, f in enumerate(freq):
        if f > 0:
            r_rad[i] = 1.0 / f  # Converter para período
    
    # Criar meshgrid com dimensões corretas
    theta_mesh, r_mesh = np.meshgrid(dirs_rad, r_rad, indexing='ij')
    
    # Verificar e corrigir dimensões da máscara
    if mask.shape != theta_mesh.shape:
        if mask.shape == theta_mesh.T.shape:
            mask = mask.T
            print(f"Transpondo mask para shape: {mask.shape}")
        else:
            # Recriar meshgrid se necessário
            if mask.shape[0] == len(dirs_rad) and mask.shape[1] == len(r_rad):
                theta_mesh, r_mesh = np.meshgrid(dirs_rad, r_rad, indexing='xy')
            else:
                theta_mesh, r_mesh = np.meshgrid(dirs_rad, r_rad, indexing='ij')
                mask = mask.T
    
    # Verificar e corrigir dimensões do E2d
    if E2d.shape != theta_mesh.shape:
        if E2d.shape == theta_mesh.T.shape:
            E2d = E2d.T
        else:
            if E2d.shape[0] == len(dirs_rad) and E2d.shape[1] == len(r_rad):
                pass  # meshgrid já foi ajustado acima
            else:
                E2d = E2d.T
    
    # Configurar figura
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, projection='polar')
    
    # Criar colormap discreto com cores para cada partição
    num_partitions = int(np.max(mask))
    colors = plt.cm.jet(np.linspace(0, 1, num_partitions + 1))
    
    # Plot da máscara
    cmap = ListedColormap(colors)
    cs = ax.pcolormesh(theta_mesh, r_mesh, mask, cmap=cmap, alpha=0.7)
    
    # Adicionar contorno do espectro original
    levels = np.linspace(0, np.max(E2d), 10)
    cs2 = ax.contour(theta_mesh, r_mesh, E2d, levels=levels, colors='k', linewidths=0.5)
    
    # Configurar eixos
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # sentido horário
    
    # Configurar eixo radial para mostrar períodos
    periods = [5, 10, 15, 20]
    ax.set_rticks(periods)
    ax.set_yticklabels([f'{p}s' for p in periods], color='gray', fontsize=7.5)
    ax.set_rlim(0, 25)
    ax.set_rlabel_position(30)
    ax.tick_params(axis='y', colors='gray', labelsize=16)
    
    # Configurar eixo angular para direções cardeais
    ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.tick_params(axis='x', colors='k', labelsize=16)
    ax.set_xticklabels(ticks)
    
    # Configurar título
    ax.set_title('Spectrum Partitioning', fontsize=8, color='k')
    
    # Adicionar colorbar
    cbar = plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label('Partition Number', fontsize=12)
    cbar.set_ticks(np.arange(0.5, num_partitions + 0.5))
    cbar.set_ticklabels(range(1, num_partitions + 1))
    
    plt.tight_layout()
    
    # Salvar figura
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        time_str = selected_time.strftime("%Y%m%d_%H%M")
        tag_part = f"{file_tag}_" if file_tag else ""
        indices_str = None
        try:
            import sys
            args = None
            for a in sys.argv:
                if '--obs-indices' in a:
                    indices_str = a.split('--obs-indices')[1].strip()
                    break
            if indices_str is None:
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument('--obs-indices', nargs='+', type=int)
                args, _ = parser.parse_known_args()
                if args.obs_indices:
                    indices_str = '_'.join(str(i) for i in args.obs_indices)
        except Exception:
            indices_str = None
        if indices_str:
            fig_path = os.path.join(save_dir, f"{file_tag}_{indices_str}_{index}_partition.png")
        else:
            fig_path = os.path.join(save_dir, f"{file_tag}_{index}_partition.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Máscara de partição salva em: {fig_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return plt.gcf()

def print_partition_summary(partition_results):
    """Imprime tabela de resumo dos resultados da partição."""
    m0 = partition_results["total_m0"]
    Hs_total = partition_results["total_Hs"]
    Tp_total = partition_results["total_Tp"]
    Dp_total = partition_results["total_Dp"]
    e = partition_results["energy"]
    Hs = partition_results["Hs"]
    Tp = partition_results["Tp"]
    Dp = partition_results["Dp"]
    
    # Obter momentos espectrais
    m0_parts = partition_results["moments"]["m0"]
    m1_parts = partition_results["moments"]["m1"]
    m2_parts = partition_results["moments"]["m2"]
    m0_total, m1_total, m2_total = partition_results["moments"]["total"]
    
    print("\nTabela de resumo:")
    print(f"{'Partição':>10} {'Energia':>12} {'Hs':>10} {'Tp':>10} {'Dp':>10} {'m0':>10} {'m1':>12} {'m2':>12}")
    print(f"{'Total':>10} {m0:12.4f} {Hs_total:10.3f} {Tp_total:10.3f} {Dp_total:10.1f} {m0_total:10.4f} {m1_total:12.6f} {m2_total:12.6f}")
    
    for idx in range(1, len(Hs)):
        if np.isnan(Tp[idx]) and np.isnan(Dp[idx]):
            continue
        print(f"{idx:10d} {e[idx]:12.4f} {Hs[idx]:10.3f} {Tp[idx]:10.3f} {Dp[idx]:10.1f} {m0_parts[idx]:10.4f} {m1_parts[idx]:12.6f} {m2_parts[idx]:12.6f}")

def save_partition_results(partition_results, selected_time, source_type_point, lon=None, lat=None, output_dir=None):
    """Salvar resultados de partição SAR em CSV único por ponto, padrão ww3.

    Diretório: <output>/<source>_partitions
    Arquivo:   <source>_<point>_partition_summary.csv (append)
    """
    if output_dir is None:
        output_dir = "./output"
    base_source = source_type_point.split('_')[0]
    part_dir = os.path.join(output_dir, f"{base_source}_partitions")
    os.makedirs(part_dir, exist_ok=True)
    summary_path = os.path.join(part_dir, f"{source_type_point}_partition_summary.csv")
    
    # Salvar picos
    date_str = selected_time.strftime("%Y-%m-%d %H:%M")
    
    # Obter momentos espectrais totais e das partições para uso no CSV
    m0_total, m1_total, m2_total = partition_results["moments"]["total"]
    m0_parts = partition_results["moments"]["m0"]
    m1_parts = partition_results["moments"]["m1"]
    m2_parts = partition_results["moments"]["m2"]

    # Cabeçalho dinâmico
    header = ["DateTime", "Total_Energy", "Total_Hs", "Total_Tp", "Total_Dp", "Total_m0", "Total_m1", "Total_m2"]
    n_parts = len(partition_results["Hs"]) - 1
    for idx in range(1, n_parts + 1):
        header += [
            f"P{idx}_Energy", f"P{idx}_Hs", f"P{idx}_Tp", f"P{idx}_Dp", 
            f"P{idx}_m0", f"P{idx}_m1", f"P{idx}_m2"
        ]
    
    row = [
        date_str,
        f"{partition_results['total_m0']:.4f}",
        f"{partition_results['total_Hs']:.3f}",
        f"{partition_results['total_Tp']:.3f}",
        f"{partition_results['total_Dp']:.1f}",
        f"{m0_total:.6f}",
        f"{m1_total:.6f}",
        f"{m2_total:.6f}"
    ]
    
    for idx in range(1, n_parts + 1):
        row += [
            f"{partition_results['energy'][idx]:.4f}",
            f"{partition_results['Hs'][idx]:.3f}",
            f"{partition_results['Tp'][idx]:.3f}",
            f"{partition_results['Dp'][idx]:.1f}",
            f"{m0_parts[idx]:.6f}",
            f"{m1_parts[idx]:.6f}",
            f"{m2_parts[idx]:.6f}"
        ]
    
    write_header = not os.path.exists(summary_path)
    with open(summary_path, 'a') as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(row) + "\n")
    print(f"Resultados da partição salvos em {part_dir}")
    return part_dir

# -----------------------------------------------------------------------------
# FUNÇÕES UTILITÁRIAS PARA GERENCIAMENTO DE ARQUIVOS
# -----------------------------------------------------------------------------

def list_partition_files_by_coordinates(output_dir, source_type="sar"):
    """
    Lista todos os arquivos CSV de partição organizados por coordenadas.
    
    Args:
        output_dir (str): Diretório de saída
        source_type (str): Tipo de fonte ("sar", "ww3", "ndbc")
    
    Returns:
        dict: Dicionário com coordenadas como chave e info do arquivo como valor
    """
    part_dir = os.path.join(output_dir, f"{source_type}_partitions")
    
    if not os.path.exists(part_dir):
        print(f"Diretório {part_dir} não existe.")
        return {}
    
    # Padrão para arquivos com coordenadas
    # Aceitar arquivos com ou sem prefixo de tag (.*_)?
    pattern = f"*{source_type}_partition_summary_lon*_lat*.csv"
    csv_files = glob.glob(os.path.join(part_dir, pattern))
    
    files_info = {}
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        # Extrair coordenadas do nome do arquivo
        match = re.search(r'lon([np\d]+)_lat([np\d]+)\.csv', filename)
        if match:
            lon_str = match.group(1).replace('n', '-').replace('p', '.')
            lat_str = match.group(2).replace('n', '-').replace('p', '.')
            
            try:
                lon = float(lon_str)
                lat = float(lat_str)
                
                # Contar registros no arquivo
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    num_records = len(df)
                    date_range = f"{df['DateTime'].iloc[0]} to {df['DateTime'].iloc[-1]}" if num_records > 0 else "Vazio"
                except Exception:
                    num_records = "Erro ao ler"
                    date_range = "N/A"
                
                files_info[(lon, lat)] = {
                    'filename': filename,
                    'path': file_path,
                    'records': num_records,
                    'date_range': date_range
                }
            except ValueError:
                print(f"Erro ao extrair coordenadas de {filename}")
    
    return files_info

def print_partition_files_summary(output_dir, source_type="sar"):
    """
    Imprime um resumo dos arquivos CSV de partição por coordenadas.
    """
    files_info = list_partition_files_by_coordinates(output_dir, source_type)
    
    if not files_info:
        print(f"Nenhum arquivo de partição encontrado para {source_type}")
        return
    
    print(f"\nResumo dos arquivos de partição {source_type.upper()}:")
    print("=" * 80)
    print(f"{'Longitude':>12} {'Latitude':>12} {'Registros':>10} {'Período':>30}")
    print("-" * 80)
    
    for (lon, lat), info in sorted(files_info.items()):
        print(f"{lon:12.4f} {lat:12.4f} {info['records']:>10} {info['date_range']:>30}")
    
    print(f"\nTotal: {len(files_info)} localidades diferentes")

# -----------------------------------------------------------------------------
# FUNÇÕES PRINCIPAIS DE PROCESSAMENTO
# -----------------------------------------------------------------------------

def process_spectrum(
    source_type, data_file, date_time=None, index=0, station_idx=0,
    do_partition=True, energy_threshold=0.05,
    save_spectrum=True, save_plot=True, show_plots=False,
    output_dir="./output", use_enhanced_plot=True, spectrum_unit="m2_Hz_rad",
    sar_scaling_factor=1.0, point_name=None, ref=None
):
    """Processa um único espectro, com opção de particionamento."""
    # Criar diretórios de saída
    if save_spectrum or save_plot:
        os.makedirs(output_dir, exist_ok=True)
        if save_spectrum:
            os.makedirs(os.path.join(output_dir, f"{source_type}_spectra"), exist_ok=True)
        if save_plot:
            os.makedirs(os.path.join(output_dir, f"{source_type}_figs"), exist_ok=True)
    
    # Gerar tag do arquivo de entrada
    file_tag = sanitize_file_tag(data_file) if data_file else None
    
    # Se ref foi fornecido, usar formato sar{ref}, caso contrário usar point_name
    if ref is not None:
        point_name = f"sar{ref}"
        print(f"[INFO] Usando ref={ref} -> point_name='{point_name}' para arquivos de saída SAR")
    else:
        point_name = infer_point_name(data_file, point_name or CONFIG.get("point_name"))
        print(f"[INFO] Usando point_name='{point_name}' para arquivos de saída SAR")

    # Abrir arquivo de dados
    ds = open_dataset(data_file, source_type)
    
    # Extrair coordenadas se disponíveis (para SAR)
    lat = None
    lon = None
    if source_type == "sar":
        try:
            # Para SAR, as coordenadas estão no grupo obs_params
            ds_obs = xr.open_dataset(data_file, group='obs_params')
            if 'latitude' in ds_obs.variables and 'longitude' in ds_obs.variables:
                lat = float(ds_obs.variables['latitude'][index])
                lon = float(ds_obs.variables['longitude'][index])
                # Coordenadas extraídas corretamente para cada índice
            ds_obs.close()
        except Exception:
            pass
    
    # Carregar espectro com base no tipo de fonte
    if source_type == "ww3":
        E2d, freq, dirs, dirs_rad, selected_time = load_ww3_spectrum(
            ds, date_time, index, station_idx, output_unit="m2_s_rad"
        )
    elif source_type == "sar":
        # Carregar SAR em m²·s·rad⁻¹ para cálculo, igual ao WW3
        E2d, freq, dirs, dirs_rad = None, None, None, None
        E2d, freq, dirs, dirs_rad, selected_time = load_sar_spectrum(
            ds, date_time, index, output_unit="m2_s_rad", scaling_factor=sar_scaling_factor
        )
    else:
        E2d, freq, dirs, dirs_rad, selected_time = load_ndbc_spectrum(
            ds, date_time, index
        )
    
    # Para cálculos de parâmetros de onda, usar sempre m²·s·rad⁻¹
    if spectrum_unit == "m2_Hz_rad":
        E2d_calc = convert_spectrum_units(E2d, freq, dirs, "m2_Hz_rad", "m2_s_rad")
    else:
        E2d_calc = E2d.copy()
    
    # Calcular parâmetros básicos (sempre em m²·s·rad⁻¹)
    hs, tp, dp, m0, delf, ddir, _, _ = calculate_wave_parameters(
        E2d, freq, dirs_rad
    )
    spec1d, _ = spectrum1d_from_2d(E2d, dirs_rad)

    # Para visualização, converter para unidade desejada
    if spectrum_unit != "m2_s_rad":
        E2d_plot = convert_spectrum_units(E2d, freq, dirs, "m2_s_rad", spectrum_unit)
        spec1d_unit = "m2_Hz" if "Hz" in spectrum_unit else "m2_s"
    else:
        E2d_plot = E2d.copy()
        spec1d_unit = "m2_s"

    # Imprimir parâmetros básicos
    print(f"\nResumo estatístico {source_type.upper()} para {selected_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Hs: {hs:.3f} m | Tp: {tp:.2f} s | Dp: {dp:.1f}°")
    print(f"Momento espectral m0: {m0:.6f}")
    
    # Salvar espectros
    if save_spectrum:
        spectrum_dir = os.path.join(output_dir, f"{source_type}_spectra")
        # Formato: sar{ref}_{date} ou source_type_{point_name}_{date}
        date_str = selected_time.strftime('%Y%m%d_%H%M')
        base_filename = f"{point_name}_{date_str}"
        
        with open(os.path.join(spectrum_dir, f"{base_filename}_spectrum_2D.txt"), 'w') as f:
            # Salvar dados do espectro
            for i in range(E2d.shape[0]):
                row = [f"{E2d[i, j]:.6g}" for j in range(E2d.shape[1])]
                f.write(",".join(row) + "\n")
        with open(os.path.join(spectrum_dir, f"{base_filename}_spectrum_1D.txt"), 'w') as f:
            f.write(f"# {source_type.upper()} 1D Wave Spectrum - {selected_time}\n")
            f.write(f"# Frequency (Hz), Spectral Density ({get_unit_label(spec1d_unit)})\n")
            for i in range(len(freq)):
                f.write(f"{freq[i]:.5f},{spec1d[i]:.6e}\n")
    
    # Plotar espectros
    if save_plot or show_plots:
        plot_dir = os.path.join(output_dir, f"{source_type}_figs")
        # Plotar espectro 2D
        if use_enhanced_plot:
            # Ajustar limites de visualização para SAR
            if source_type == "sar":
                # Se a unidade de exibição for m2_Hz_rad, usa limites pequenos; caso contrário, usa dinâmica
                if spectrum_unit == "m2_Hz_rad":
                    vmin, vmax = 1e-6, 1e-3
                else:
                    vmin, vmax = None, None
            else:
                vmin, vmax = 0.025, 2.75
            
            plot_directional_spectrum(
                E2d_plot, freq, dirs, hs, tp, dp, selected_time, point_name,
                save_dir=plot_dir if save_plot else None,
                show_plot=show_plots,
                spectrum_unit="m2_s_rad",
                max_density_2d=CONFIG.get("max_density_2d", 25.0),
                lon=lon,
                lat=lat,
                file_tag=file_tag,
                index=index
            )
        # Plotar espectro 1D
        plot_spectrum_1d(
            spec1d, freq, selected_time, hs, tp, source_type,
            save_dir=plot_dir if save_plot else None,
            show_plot=show_plots,
            spectrum_unit="m2_s",
            max_density=CONFIG.get("max_density_1d", 25.0),
            point_name=point_name,
            file_tag=file_tag,
            lon=lon,
            lat=lat,
            index=index
        )
    
    # Executar particionamento se solicitado
    if do_partition:
        print("\nExecutando particionamento do espectro...")
        # Usar espectro em unidades de cálculo para particionamento
        partition_results = partition_spectrum(
            E2d, freq, dirs_rad, energy_threshold, source_type=source_type
        )
        
        if partition_results is not None:
            # Imprimir resumo
            print_partition_summary(partition_results)
            
            # Salvar resultados
            if save_spectrum:
                # Buscar latitude e longitude do dataset SAR
                lat = None
                lon = None
                try:
                    if hasattr(ds, 'variables'):
                        if 'latitude' in ds.variables:
                            lat = float(ds.variables['latitude'][0])
                        if 'longitude' in ds.variables:
                            lon = float(ds.variables['longitude'][0])
                except Exception:
                    pass
                save_partition_results(
                    partition_results, selected_time, f"{source_type}_{point_name}",
                    lon=lon, lat=lat, output_dir=output_dir
                )
            # (Removido) Não gerar máscara de partição
            
            return {
                "hs": hs,
                "tp": tp,
                "dp": dp,
                "partition_results": partition_results
            }
    
    # Fechar o dataset
    ds.close()
    
    return {
        "hs": hs,
        "tp": tp,
        "dp": dp
    }

def process_time_range(config):
    """Processa espectros para um intervalo de datas usando múltiplos arquivos SAR."""
    # Extrair configurações
    source_type = config["source_type"]
    
    # Usar data_directory ou data_file (compatibilidade retroativa)
    if "data_directory" in config:
        data_directory = config["data_directory"]
        data_file = None
    else:
        data_file = config.get("data_file")
        data_directory = None
        
    start_date = config.get("data_ini", config.get("start_date"))  # Compatibilidade retroativa
    end_date = config.get("data_fim", config.get("end_date"))      # Compatibilidade retroativa
    time_step_hours = config.get("time_step_hours", 1)
    station_idx = config.get("station_idx", 0)
    do_partition = config.get("do_partition", True)
    energy_threshold = config.get("energy_threshold", 0.05)
    save_spectrum = config.get("save_spectrum", True)
    save_plot = config.get("save_plot", True)
    show_plots = config.get("show_plots", False)
    output_dir = config.get("output_dir", "./output")
    use_enhanced_plot = config.get("use_enhanced_plot", True)
    spectrum_unit = config.get("spectrum_unit", "m2_Hz_rad")
    sar_scaling_factor = config.get("sar_scaling_factor", 1.0)
    
    # Converter datas para datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    results = []
    
    if source_type == "sar" and data_directory:
        # Processar múltiplos arquivos SAR
        print(f"Processando arquivos SAR de {start_dt} a {end_dt}")
        
        # Encontrar arquivos SAR no intervalo
        sar_files = find_sar_files_in_date_range(data_directory, start_dt, end_dt)
        
        if not sar_files:
            print("Nenhum arquivo SAR encontrado no intervalo especificado!")
            return results
        
        # Processar cada arquivo SAR
        for file_path, file_start, file_end in sar_files:
            filename = os.path.basename(file_path)
            file_center = extract_datetime_from_sar_filename(file_path)
            
            print(f"\nProcessando arquivo SAR: {filename}")
            print(f"  Período do arquivo: {file_start} - {file_end}")
            print(f"  Centro do arquivo: {file_center}")
            
            result = process_spectrum(
                source_type=source_type,
                data_file=file_path,
                date_time=file_center,
                station_idx=station_idx,
                do_partition=do_partition,
                energy_threshold=energy_threshold,
                save_spectrum=save_spectrum,
                save_plot=save_plot,
                show_plots=show_plots,
                output_dir=output_dir,
                use_enhanced_plot=use_enhanced_plot,
                spectrum_unit=spectrum_unit,
                sar_scaling_factor=sar_scaling_factor
            )
            results.append((file_center or file_start, result))
    
    else:
        # Modo tradicional - arquivo único ou outros tipos de fonte
        if data_file is None:
            print("Erro: Nem data_directory nem data_file foram especificados!")
            return results
            
        # Criar lista de datas a processar
        if start_dt == end_dt:
            dates_to_process = [start_dt]
            print(f"Processando {source_type.upper()} para: {start_dt}")
        else:
            dates_to_process = pd.date_range(start=start_dt, end=end_dt, freq=f'{time_step_hours}H')
            print(f"Processando {len(dates_to_process)} pontos de tempo de {start_dt} a {end_dt} (a cada {time_step_hours} horas)")
        
        # Processar cada data
        for dt in dates_to_process:
            print(f"\nProcessando {source_type.upper()} para {dt}")
            result = process_spectrum(
                source_type=source_type,
                data_file=data_file,
                date_time=dt,
                station_idx=station_idx,
                do_partition=do_partition,
                energy_threshold=energy_threshold,
                save_spectrum=save_spectrum,
                save_plot=save_plot,
                show_plots=show_plots,
                output_dir=output_dir,
                use_enhanced_plot=use_enhanced_plot,
                spectrum_unit=spectrum_unit,
                sar_scaling_factor=sar_scaling_factor
            )
            results.append((dt, result))
    
    return results

# -----------------------------------------------------------------------------
# FUNÇÕES PARA PROCESSAMENTO BATCH
# -----------------------------------------------------------------------------

def read_points_csv(csv_file):
    """Lê o CSV com dados SAR (ref, obs_index, longitude, latitude, time, filename).
    
    Returns:
        dict: Dicionário com {ref: {'obs_index': int, 'time': datetime, 'filename': str, 'lon': float, 'lat': float}}
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"CSV carregado: {len(df)} registros encontrados")
        
        points_dict = {}
        for _, row in df.iterrows():
            ref = int(row['ref'])
            
            # Converter timestamp para datetime
            time_str = row['time']
            obs_time = pd.to_datetime(time_str)
            
            points_dict[ref] = {
                'obs_index': int(row['obs_index']),
                'time': obs_time,
                'filename': row['filename'],
                'lon': float(row['longitude']),
                'lat': float(row['latitude'])
            }
            
        print(f"Processados {len(points_dict)} pontos válidos do CSV")
        return points_dict
        
    except Exception as e:
        print(f"Erro ao ler CSV {csv_file}: {e}")
        import traceback
        traceback.print_exc()
        return {}

def get_sar_file_for_point(file_template, point_num):
    """Gera o caminho do arquivo SAR para um ponto específico com fallback para arquivos _a."""
    # Tentar arquivo principal primeiro
    file_path = file_template.format(point_num=point_num)
    if os.path.exists(file_path):
        return file_path
    
    # Fallback: tentar arquivo com sufixo '_a'
    file_path_a = file_template.format(point_num=f"{point_num}_a")
    if os.path.exists(file_path_a):
        return file_path_a
    
    # Se nenhum dos dois existe, retorna o caminho principal para erro consistente
    return file_path

def check_sar_file_exists(file_path):
    """Verifica se o arquivo SAR existe e reporta qual versão foi encontrada."""
    exists = os.path.exists(file_path)
    if exists:
        if '_a.nc' in file_path:
            print(f"  📁 Usando arquivo alternativo: {os.path.basename(file_path)}")
        return True
    else:
        print(f"Arquivo SAR não encontrado: {file_path}")
        return False

def get_exact_date_sar(target_date):
    """Retorna a data exata para processamento SAR (sem janela temporal)."""
    return target_date, target_date

def get_flexible_date_sar(target_date, max_hours_diff=6):
    """Retorna janela temporal flexível quando data exata não funciona.
    
    Args:
        target_date: Data alvo
        max_hours_diff: Máxima diferença em horas permitida
    
    Returns:
        tuple: (data_início, data_fim) da janela flexível
    """
    from datetime import timedelta
    target_dt = pd.to_datetime(target_date)
    start_time = target_dt - timedelta(hours=max_hours_diff)
    end_time = target_dt + timedelta(hours=max_hours_diff)
    return start_time, end_time

def process_directory_sar_files(config):
    """Processa todos os arquivos SAR encontrados em um diretório.
    
    Parameters:
        config: Dicionário de configuração contendo:
               - sar_directory: Diretório com arquivos SAR
               - outras configurações de processamento
    
    Returns:
        list: Lista de resultados do processamento
    """
    sar_directory = config.get('sar_directory')
    
    if not os.path.exists(sar_directory):
        print(f"Erro: Diretório não encontrado: {sar_directory}")
        return []
    
    # Buscar todos os arquivos SAR no diretório
    print(f"Procurando arquivos SAR em: {sar_directory}")
    sar_files = []
    for file in os.listdir(sar_directory):
        if file.startswith('surigae_ponto') and file.endswith('.nc'):
            sar_files.append(os.path.join(sar_directory, file))
    
    sar_files.sort()
    print(f"Encontrados {len(sar_files)} arquivos SAR")
    
    if not sar_files:
        print("Nenhum arquivo SAR encontrado no diretório!")
        return []
    
    print(f"\n{'='*70}")
    print(f"PROCESSAMENTO DE DIRETÓRIO SAR - {len(sar_files)} ARQUIVOS")
    print(f"{'='*70}")
    
    results = []
    failed_files = []
    
    for i, file_path in enumerate(sar_files, 1):
        file_name = os.path.basename(file_path)
        
        # Extrair número do ponto do nome do arquivo
        try:
            if '_a.nc' in file_name:
                point_num_str = file_name.replace('surigae_ponto', '').replace('_a.nc', '')
                point_name = f"ponto{point_num_str}_a"
            else:
                point_num_str = file_name.replace('surigae_ponto', '').replace('.nc', '')
                point_name = f"ponto{point_num_str}"
            
            point_num = int(point_num_str)
        except ValueError:
            print(f"❌ Arquivo {file_name}: Não foi possível extrair número do ponto")
            failed_files.append(file_name)
            continue
        
        print(f"\n--- Processando Arquivo SAR {i}/{len(sar_files)} ---")
        print(f"Arquivo: {file_name}")
        print(f"Ponto: {point_name}")
        
        try:
            # Abrir arquivo para descobrir período temporal disponível
            import xarray as xr
            ds = xr.open_dataset(file_path)
            times = pd.to_datetime(ds.time.values)
            
            if len(times) == 0:
                print(f"❌ Arquivo {file_name}: Sem dados temporais")
                failed_files.append(file_name)
                ds.close()
                continue
            
            # Usar todo o período disponível no arquivo
            start_time = times.min()
            end_time = times.max()
            
            print(f"Período disponível: {start_time} a {end_time}")
            print(f"Total de registros: {len(times)}")
            
            ds.close()
            
            # Configurar processamento para este arquivo
            temp_config = config.copy()
            temp_config.update({
                "data_directory": file_path,
                "data_ini": start_time.strftime("%Y-%m-%d %H:%M"),
                "data_fim": end_time.strftime("%Y-%m-%d %H:%M"),
                "point_name": point_name
            })
            
            # Processar arquivo
            file_results = process_time_range(temp_config)
            
            if file_results:
                results.extend([(file_name, dt, result) for dt, result in file_results])
                print(f"✅ Arquivo {file_name}: {len(file_results)} espectros processados")
            else:
                failed_files.append(file_name)
                print(f"❌ Arquivo {file_name}: Nenhum espectro processado")
                
        except Exception as e:
            failed_files.append(file_name)
            print(f"❌ Arquivo {file_name}: Erro no processamento - {e}")
    
    # Resumo final
    print(f"\n{'='*70}")
    print(f"RESUMO DO PROCESSAMENTO DE DIRETÓRIO SAR")
    print(f"{'='*70}")
    print(f"Total de arquivos encontrados: {len(sar_files)}")
    print(f"Arquivos processados com sucesso: {len(sar_files) - len(failed_files)}")
    print(f"Arquivos com falha: {len(failed_files)}")
    if failed_files:
        print(f"Arquivos que falharam: {failed_files[:10]}")  # Mostrar apenas primeiros 10
        if len(failed_files) > 10:
            print(f"... e mais {len(failed_files) - 10} arquivos")
    print(f"Total de espectros gerados: {len(results)}")
    
    return results

def process_batch_sar_points(config):
    """Processa múltiplos pontos SAR baseado no CSV (ref, obs_index, time, filename).
    Equivalente ao script bash que lê linha por linha do CSV.
    
    Parameters:
        config: Dicionário de configuração
    
    Returns:
        list: Lista de resultados para cada ponto processado
    """
    csv_file = config["csv_points_file"]
    data_directory = config["data_directory"]
    
    # Ler pontos do CSV
    points_dict = read_points_csv(csv_file)
    if not points_dict:
        print("Nenhum ponto válido encontrado no CSV")
        return []
    
    # Configurações de processamento
    source_type = config["source_type"]
    do_partition = config.get("do_partition", True)
    energy_threshold = config.get("energy_threshold", 0.05)
    save_spectrum = config.get("save_spectrum", True)
    save_plot = config.get("save_plot", True)
    show_plots = config.get("show_plots", False)
    output_dir = config.get("output_dir", "./output")
    use_enhanced_plot = config.get("use_enhanced_plot", True)
    spectrum_unit = config.get("spectrum_unit", "m2_Hz_rad")
    sar_scaling_factor = config.get("sar_scaling_factor", 1.0)
    
    results = []
    failed_points = []
    
    print(f"\n{'='*70}")
    print(f"PROCESSAMENTO BATCH SAR - {len(points_dict)} PONTOS DO CSV")
    print(f"{'='*70}")
    print(f"CSV: {csv_file}")
    print(f"Diretório de dados: {data_directory}")
    print(f"")
    
    # Processar cada linha do CSV (equivalente ao while IFS=, read do bash)
    for ref in sorted(points_dict.keys()):
        point_data = points_dict[ref]
        obs_index = point_data['obs_index']
        obs_time = point_data['time']
        filename = point_data['filename']
        lon = point_data['lon']
        lat = point_data['lat']
        
        print(f"\n{'─'*70}")
        print(f"📍 Ref={ref} | obs_index={obs_index} | {filename}")
        print(f"   Time: {obs_time} | Lon: {lon:.3f}° | Lat: {lat:.3f}°")
        print(f"{'─'*70}")
        
        # Construir caminho completo do arquivo (equivalente ao $HOME/...)
        file_path = os.path.join(data_directory, filename)
        
        # Verificar se o arquivo existe
        if not os.path.exists(file_path):
            failed_points.append(ref)
            print(f"❌ ERRO: Arquivo não encontrado: {file_path}")
            continue
        
        point_processed = False
        
        try:
            # Processar exatamente como no script bash:
            # python sar_part_surigae.py \
            #   --file "$filename" \
            #   --obs-indices "$obs_index" \
            #   --ref "$ref"
            
            result = process_spectrum(
                source_type=source_type,
                data_file=file_path,
                date_time=obs_time,  # Usar o tempo do CSV
                index=obs_index,     # Usar o obs_index do CSV
                station_idx=0,
                do_partition=do_partition,
                energy_threshold=energy_threshold,
                save_spectrum=save_spectrum,
                save_plot=save_plot,
                show_plots=show_plots,
                output_dir=output_dir,
                use_enhanced_plot=use_enhanced_plot,
                spectrum_unit=spectrum_unit,
                sar_scaling_factor=sar_scaling_factor,
                point_name=None,
                ref=ref  # Nomear como sar1, sar2, sar3, etc
            )
            
            if result:
                results.append((ref, obs_time, result))
                hs_str = f"{result['hs']:.2f}" if 'hs' in result else "N/A"
                tp_str = f"{result['tp']:.1f}" if 'tp' in result else "N/A"
                dp_str = f"{result['dp']:.1f}" if 'dp' in result else "N/A"
                print(f"✅ SUCESSO: Hs={hs_str}m | Tp={tp_str}s | Dp={dp_str}°")
                
                if 'partition_results' in result and result['partition_results']:
                    n_parts = result['partition_results']['nmask']
                    print(f"   → {n_parts} partições identificadas")
                
                point_processed = True
                
        except Exception as e:
            point_processed = False
            print(f"❌ ERRO no processamento: {e}")
            import traceback
            traceback.print_exc()
        
        if not point_processed:
            failed_points.append(ref)
    
    # Resumo final (equivalente ao fim do script bash)
    print(f"\n{'='*70}")
    print(f"📊 RESUMO DO PROCESSAMENTO BATCH")
    print(f"{'='*70}")
    print(f"Total de linhas no CSV: {len(points_dict)}")
    print(f"✅ Processados com sucesso: {len(points_dict) - len(failed_points)}")
    print(f"❌ Falharam: {len(failed_points)}")
    if failed_points:
        print(f"   Refs com falha: {sorted(failed_points)}")
    print(f"📁 Total de espectros gerados: {len(results)}")
    print(f"{'='*70}")
    
    return results

# -----------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# -----------------------------------------------------------------------------

def parse_arguments():
    """Parse argumentos da linha de comando para SAR."""
    parser = argparse.ArgumentParser(
        description='Processamento de espectros SAR - Modo single ou batch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

1. Modo batch - processar múltiplos pontos do CSV:
   python spec_partition.py --batch arquivo.csv
   
   O CSV deve ter: ref, obs_index, longitude, latitude, time, filename

2. Modo single - arquivo único com índice específico:
   python spec_partition.py --file arquivo.nc --obs-indices 34 --ref 1
   
3. Modo single - arquivo único com intervalo de datas:
   python spec_partition.py --file arquivo.nc --start "2021-04-15 00:00" --end "2021-04-16 00:00"
        """
    )
    
    # Argumentos principais
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', 
                      help='Arquivo NetCDF SAR para processamento individual')
    group.add_argument('--batch', '-b', nargs='?', const=True, metavar='CSV_FILE',
                      help='Processar múltiplos pontos do CSV. Uso: --batch arquivo.csv')
    
    # Argumentos para modo single
    parser.add_argument('--start', '-s', 
                       help='Data inicial (formato: "YYYY-MM-DD HH:MM")')
    parser.add_argument('--end', '-e', 
                       help='Data final (formato: "YYYY-MM-DD HH:MM")')
    
    # Argumentos opcionais comuns
    parser.add_argument('--no-partition', action='store_true', 
                       help='Desabilitar particionamento')
    parser.add_argument('--no-save', action='store_true',
                       help='Não salvar arquivos de saída')
    parser.add_argument('--show-plots', action='store_true',
                       help='Mostrar plots na tela')
    parser.add_argument('--output-dir', '-o', 
                       default='/Users/jtakeo/googleDrive/myProjects/finalPartSpec/surigae/sarspec/output',
                       help='Diretório de saída (padrão: /Users/jtakeo/googleDrive/myProjects/finalPartSpec/surigae/sarspec/output)')
    parser.add_argument('--obs-indices', nargs='+', type=int,
                        help='Lista de índices de observações (no grupo obs_params) a processar em vez de intervalo de datas. Ex: --obs-indices 0 1 2 3')
    parser.add_argument('--ref', type=int,
                        help='Número de referência para nomear arquivos (ex: --ref 1 gera sar1_YYYYMMDD_HHMM)')
    
    return parser.parse_args()

def main():
    """Função principal do programa."""
    args = parse_arguments()
    
    # Criar configuração baseada nos argumentos
    config = CONFIG.copy()
    
    # Configurações comuns
    config['do_partition'] = not args.no_partition
    config['save_spectrum'] = not args.no_save
    config['save_plot'] = not args.no_save
    config['show_plots'] = args.show_plots
    config['output_dir'] = args.output_dir
    
    if args.batch:
        # Modo batch
        print("="*70)
        print("MODO BATCH SAR - Processamento múltiplo baseado em CSV")
        print("="*70)
        
        # Determinar arquivo CSV
        if isinstance(args.batch, str):  # --batch arquivo.csv
            csv_file = args.batch
        else:  # --batch sem argumento: usar padrão do CONFIG
            csv_file = config.get('csv_points_file')
        
        if not csv_file:
            print("Erro: É necessário fornecer o arquivo CSV:")
            print("  python spec_partition.py --batch arquivo.csv")
            return
        
        # Atualizar configurações para batch
        config['batch_mode'] = True
        config['csv_points_file'] = csv_file
        
        # Verificar se arquivo CSV existe
        if not os.path.exists(config['csv_points_file']):
            print(f"Erro: Arquivo CSV não encontrado: {config['csv_points_file']}")
            return
        
        print(f"CSV de pontos: {config['csv_points_file']}")
        print(f"Diretório de dados: {config['data_directory']}")
        print(f"Modo: Processar ref, obs_index e time do CSV")
        print(f"Particionamento: {'Sim' if config['do_partition'] else 'Não'}")
        print(f"Salvar resultados: {'Sim' if config['save_spectrum'] else 'Não'}")
        
        # Processar pontos em batch
        results = process_batch_sar_points(config)
        
    else:
        # Modo single
        print("="*60)
        print("MODO SINGLE SAR - Processamento de arquivo único")
        print("="*60)
        
        # Verificar argumentos obrigatórios
        if not args.file:
            print("Erro: Para modo single é necessário fornecer --file")
            return
        
        # Se índices de observação forem fornecidos, ignoramos --start/--end e processamos exatamente aqueles tempos
        process_by_indices = args.obs_indices is not None and len(args.obs_indices) > 0
        if not process_by_indices:
            if not args.start or not args.end:
                print("Erro: Para modo single é necessário fornecer --start e --end (ou usar --obs-indices)")
                return
        
        # Atualizar configurações para single
        config['data_directory'] = args.file
        config['data_ini'] = args.start
        config['data_fim'] = args.end
        
        # Verificar se arquivo existe
        if not os.path.exists(config['data_directory']):
            print(f"Erro: Arquivo não encontrado: {config['data_directory']}")
            return
        
        print(f"Arquivo SAR: {config['data_directory']}")
        if process_by_indices:
            print(f"Processando por índices de observação (grupo obs_params): {args.obs_indices}")
        else:
            print(f"Intervalo de tempo: {config['data_ini']} a {config['data_fim']}")
        print(f"Particionamento: {'Sim' if config['do_partition'] else 'Não'}")
        print(f"Salvar resultados: {'Sim' if config['save_spectrum'] else 'Não'}")
        print(f"Mostrar plots: {'Sim' if config['show_plots'] else 'Não'}")
        
        if process_by_indices:
            # Abrir dataset para obter tempos correspondentes
            try:
                ds_tmp = open_dataset(config['data_directory'], source_type='sar')
            except Exception as e:
                print(f"Erro ao abrir dataset para leitura de tempos: {e}")
                return
            # Descobrir vetor de tempo
            time_var_candidates = ['time','obs_time','TIME','time_center','valid_time','t']
            sar_times = None
            for tv in time_var_candidates:
                if tv in ds_tmp.variables:
                    sar_times = ds_tmp[tv].values
                    break
            if sar_times is None:
                print("Não foi possível localizar variável de tempo no arquivo.")
                return
            # Converter para pandas datetime
            import pandas as pd, numpy as np
            if np.issubdtype(sar_times.dtype, np.datetime64):
                times_dt = pd.to_datetime(sar_times)
            else:
                # assumir dias desde 1950-01-01
                ref = pd.Timestamp('1950-01-01')
                times_dt = pd.to_datetime([ref + pd.to_timedelta(float(d), unit='D') for d in sar_times])
            selected_times = []
            for idx in args.obs_indices:
                if idx < 0 or idx >= len(times_dt):
                    print(f"Aviso: índice {idx} fora do intervalo (0..{len(times_dt)-1}), ignorando.")
                    continue
                selected_times.append(times_dt[idx])
            if not selected_times:
                print("Nenhum índice válido fornecido.")
                return
            print("Tempos correspondentes aos índices:")
            for i, dt in zip(args.obs_indices, selected_times):
                if 0 <= i < len(times_dt):
                    print(f"  idx={i}: {dt}")
            # Processar cada tempo individualmente reutilizando process_spectrum
            results = []
            for idx, dt in zip(args.obs_indices, selected_times):
                print(f"\nProcessando espectro para tempo (idx mapeado): {dt}")
                res = process_spectrum(
                    source_type='sar',
                    data_file=config['data_directory'],
                    date_time=dt,
                    index=idx,  # CORREÇÃO: Passar o índice correto
                    station_idx=config.get('station_idx',0),
                    do_partition=config['do_partition'],
                    energy_threshold=config.get('energy_threshold',0.05),
                    save_spectrum=config['save_spectrum'],
                    save_plot=config['save_plot'],
                    show_plots=config['show_plots'],
                    output_dir=config['output_dir'],
                    use_enhanced_plot=config['use_enhanced_plot'],
                    spectrum_unit=config.get('spectrum_unit','m2_Hz_rad'),
                    sar_scaling_factor=config.get('sar_scaling_factor',1.0),
                    point_name=config.get('point_name'),
                    ref=args.ref if hasattr(args, 'ref') else None
                )
                results.append((dt, res))
        else:
            # Processar intervalo de tempo
            results = process_time_range(config)
        
        print(f"\nProcessamento concluído para {len(results)} pontos de tempo.")
        
        # Resumo final
        print("\nResumo dos resultados:")
        for dt, result in results:
            hs_str = f"{result['hs']:.2f}" if 'hs' in result else "N/A"
            tp_str = f"{result['tp']:.1f}" if 'tp' in result else "N/A"
            dp_str = f"{result['dp']:.1f}" if 'dp' in result else "N/A"
            print(f"{dt.strftime('%Y-%m-%d %H:%M')}: Hs={hs_str}m, Tp={tp_str}s, Dp={dp_str}°")
            
            if 'partition_results' in result:
                part_count = len([h for h in result['partition_results']['Hs'][1:] if h > 0])
                print(f"  - {part_count} partições identificadas")

def convert_meteorological_to_oceanographic(met_dir):
    """Converte direção meteorológica (de onde vem) para oceanográfica (para onde vai).
    
    Args:
        met_dir: Direção meteorológica em graus (de onde o vento/onda vem)
        
    Returns:
        Direção oceanográfica em graus (para onde a onda vai)
    """
    return (met_dir + 180) % 360

def convert_sar_energy_units(E_sar, k, phi, scaling_factor=1.0):
    """Converte espectro SAR de número de onda para frequência em m²·s·rad⁻¹ (igual ao WW3).
    
    IMPORTANTE: SAR usa convenção meteorológica (direção DE onde vem), 
    WW3 usa convenção oceanográfica (direção PARA onde vai).
    Esta função aplica a conversão necessária.
    """
    
    g = 9.81
    omega = np.sqrt(g * k)
    freq = omega / (2 * np.pi)
    
    # Jacobiano teórico: |dk/df| = 8π²f/g (correto)
    dkdf = 8 * np.pi**2 * freq / g
    dkdf_matrix = dkdf.reshape(-1, 1)
    
    # Conversão de graus para radianos: 180/π
    deg_to_rad = 180.0 / np.pi
    
    # CORREÇÃO BASEADA NA ANÁLISE TEÓRICA:
    # Fator de normalização SAR específico para conversão no domínio f
    # Este valor foi determinado pela calibração com dados de referência
    sar_normalization_factor = 3.398953e-05
    
    # E(k,theta) [m4] -> E(f,theta) [m2·s·rad⁻¹]
    E_m2_s_rad = E_sar * sar_normalization_factor * dkdf_matrix * deg_to_rad * scaling_factor
    
    # Ajuste shape para (NF, ND)
    if E_m2_s_rad.shape[0] != len(freq):
        E_m2_s_rad = E_m2_s_rad.T
    
    # CORREÇÃO: Dados SAR já estão em convenção oceanográfica (direção PARA onde vai)
    # NÃO aplicar conversão meteorológica->oceanográfica (estava invertendo as direções)
    phi_oceanographic = phi  # Usar direções SAR originais
    dirs_rad = np.radians(phi_oceanographic)
    
    print(f"Conversão SAR: {E_sar.shape} -> {E_m2_s_rad.shape}")
    print(f"Valores SAR originais: min={np.min(E_sar):.2e}, max={np.max(E_sar):.2e}")
    print(f"Fator de normalização SAR aplicado: {sar_normalization_factor:.6e}")
    print(f"Valores convertidos: min={np.min(E_m2_s_rad):.2e}, max={np.max(E_m2_s_rad):.2e}")
    print(f"Energia total convertida (m2.s.rad-1): {np.sum(E_m2_s_rad):.2e}")
    print(f"Shape freq: {len(freq)}, Shape dirs: {len(phi)}")
    print(f"DIREÇÕES CORRIGIDAS: Usando direções SAR originais (já em convenção oceanográfica)")
    print(f"  Direções SAR (primeiras 5): {phi[:5]}")
    print(f"  Direções utilizadas: {phi_oceanographic[:5]}")
    
    return E_m2_s_rad, freq, phi_oceanographic, dirs_rad

if __name__ == "__main__":
    print("Iniciando processamento de espectros...")
    main()

# %%
