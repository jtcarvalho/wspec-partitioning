## <!-- 🔄 Complete Workflow (WASP + HIVE)

### Step 1: Partition Spectra (WASP)

```bash
cd /path/to/wasp
conda activate wasp

# Partition SAR spectra
python scripts/01_partition_sar.py

# Partition WW3 spectra
python scripts/03_partition_ww3.py
```

**Output:** CSV files with partition parameters in `data/all/partition/`

### Step 2: Analyze and Validate (HIVE)

```bash
# Copy partition results to HIVE
cp data/all/partition/partition*.csv /path/to/hive/data/input/

cd /path/to/hive
conda activate hive

# Run validation and analysis
python scripts/01_validate.py
python scripts/02_analyze_bias.py
python scripts/03_analyze_energy.py
python scripts/04_analyze_characteristics.py
python scripts/05_analyze_overestimation.py
```

**Output:** Comprehensive reports, figures, and statistics in `output/` and `docs/`

## 🎯 WASP Scope

WASP is focused exclusively on **spectral partitioning algorithms**. For a complete analysis workflow including validation, bias analysis, and reporting, use both repositories together.
