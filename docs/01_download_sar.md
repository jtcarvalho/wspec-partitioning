# 📥 Download de Dados SAR Sentinel-1

## Fonte de Dados

Os dados de espectro de onda SAR do Sentinel-1A/B estão disponíveis no **Copernicus Marine Environment Monitoring Service (CMEMS)**:

🔗 https://marine.copernicus.eu/

---

## Dataset Recomendado

**SENTINEL-1 L2 OCEAN WAVE SPECTRA**

- **Product ID**: `WAVE_GLO_PHY_SWH_L2_NRT_014_016` (Near Real Time)
- **Product ID**: `WAVE_GLO_PHY_SWH_L2_MY_014_006` (Multi-Year Reprocessed)

### Variáveis necessárias:
- `wave_spec` ou `obs_params/wave_spec` - Espectro 2D de energia [m⁴]
- `wavenumber_spec` - Número de onda [rad/m]
- `direction_spec` - Direções [graus]
- `time` - Timestamp da observação
- `longitude`, `latitude` - Coordenadas
- `L2_partition_quality_flag` - Flag de qualidade (0 = melhor)

---

## Passos para Download

### 1. Criar conta no CMEMS
- Acesse: https://marine.copernicus.eu/
- Crie uma conta gratuita (para pesquisa acadêmica)

### 2. Selecionar região e período
Para ciclones tropicais específicos, defina:

**Ciclone Surigae (2021):**
- Período: 15-25 Abril 2021
- Região: 5°N-25°N, 120°E-145°E

**Ciclone Lee (2023):**
- Período: 5-20 Setembro 2023
- Região: 15°N-45°N, 70°W-40°W

**Ciclone Freddy (2023):**
- Período: 6-28 Fevereiro 2023
- Região: 25°S-10°S, 40°E-75°E

### 3. Fazer download via Python (motu-client)

```bash
pip install motuclient
```

```bash
python -m motuclient \
  --motu https://nrt.cmems-du.eu/motu-web/Motu \
  --service-id WAVE_GLO_PHY_SWH_L2_NRT_014_016-TDS \
  --product-id cmems_obs-wave_glo_phy-swh_nrt_s1a-l2-wsp_PT \
  --longitude-min 120 --longitude-max 145 \
  --latitude-min 5 --latitude-max 25 \
  --date-min "2021-04-15 00:00:00" --date-max "2021-04-25 23:59:59" \
  --variable wave_spec \
  --variable wavenumber_spec \
  --variable direction_spec \
  --variable time \
  --variable longitude \
  --variable latitude \
  --variable L2_partition_quality_flag \
  --out-dir ./data/sentinel1ab/surigae/ \
  --out-name surigae_sar.nc \
  --user <SEU_USERNAME> \
  --pwd <SUA_SENHA>
```

---

## Estrutura de Diretórios Esperada

Após o download, organize os arquivos assim:

```
/Users/jtakeo/data/sentinel1ab/
├── all/
│   └── sar_all.nc
├── surigae/
│   └── sar_surigae.nc
├── lee/
│   └── sar_lee.nc
└── freddy/
    └── sar_freddy.nc
```

---

## Verificar Dados Baixados

Use este snippet para verificar se o arquivo está correto:

```python
import xarray as xr

# Carregar arquivo
ds = xr.open_dataset('data/sentinel1ab/surigae/sar_surigae.nc', group='obs_params')

# Verificar variáveis
print("Variáveis disponíveis:", list(ds.variables.keys()))

# Verificar dimensões
print("\nDimensões:")
print(f"  Observações: {len(ds.time)}")
print(f"  Frequências: {len(ds.wavenumber_spec)}")
print(f"  Direções: {len(ds.direction_spec)}")

# Verificar range de datas
print("\nPeríodo:")
print(f"  Início: {ds.time.values[0]}")
print(f"  Fim: {ds.time.values[-1]}")
```

---

## Alternativa: Download Manual via Interface Web

Se preferir interface gráfica:

1. Acesse: https://data.marine.copernicus.eu/
2. Busque por "Sentinel-1 Wave Spectra"
3. Use o mapa interativo para selecionar região
4. Defina período temporal
5. Selecione variáveis necessárias
6. Clique em "Download" e escolha formato NetCDF

---

## Próximo Passo

Após o download, execute:
```bash
cd scripts
python 01_partition_sar.py
```

Veja o README principal para mais detalhes.
