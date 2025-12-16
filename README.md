# **WASP** - **WA**ve **S**pectra **P**artitioning

Watershed Algorithm for partitioning the ocean wave spectra from WW3 and SAR (Sentinel)

> **🔗 Companion Repository:** For analysis and validation of partitioned spectra, see [**HIVE** (Hierarchical Integration of Verified wavE partitions)](https://github.com/jtcarvalho/hive)

## 📋 What is WASP?

WASP focuses exclusively on **spectral partitioning** - the process of separating ocean wave spectra into individual wave systems (partitions). Each partition represents a distinct wave system characterized by significant wave height (Hs), peak period (Tp), and direction (Dp).

**WASP handles:**
- ✅ Spectral partitioning using watershed algorithm
- ✅ Processing SAR (Sentinel) and WW3 model spectra
- ✅ Extracting wave parameters (Hs, Tp, Dp) for each partition

**WASP does NOT handle:**
- ❌ Statistical validation and comparison
- ❌ Bias analysis and reporting
- ❌ Visualization of validation metrics

👉 **For analysis and validation**, use the companion repository [**HIVE**](https://github.com/jtcarvalho/hive)

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Git

### Option 1: Using pip (Virtual Environment)

```bash
# Clone the repository
git clone https://github.com/jtcarvalho/wasp.git
cd wasp

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/jtcarvalho/wasp.git
cd wasp

# Create and activate conda environment
conda env create -f environment.yml
conda activate wasp
```

### Verify Installation

```bash
# Check NumPy version (must be >= 2.1.0 for np.trapezoid)
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Check other key packages
python -c "import pandas, xarray, matplotlib, scipy; print('✓ All packages installed successfully')"
```

## 📦 Key Dependencies

- **NumPy >= 2.1.0** (required for `np.trapezoid`)
- pandas >= 2.2.0
- xarray >= 2024.11.0
- matplotlib >= 3.8.0
- scipy >= 1.14.0
- scikit-image >= 0.22.0
- netCDF4 >= 1.5.4

> ⚠️ **Important:** NumPy versions < 2.1.0 will cause errors as `np.trapezoid` is not available (only `np.trapz` which is deprecated).

## 🔄 Complete Workflow (WASP + HIVE)

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
