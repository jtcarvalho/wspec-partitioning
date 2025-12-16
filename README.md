# **WASP** - **WA**ve **S**pectra **P**artitioning

Watershed Algorithm for partitioning the ocean wave spectra from WW3 and SAR (Sentinel

<!--

**🔗 Companion Repository:** For analysis and validation of partitioned spectra, see [**HIVE** (Hierarchical Integration of Verified wavE partitions)](https://github.com/jtcarvalho/hive)

-->

## 📋 What is WASP?

WASP focuses exclusively on **spectral partitioning** - the process of separating ocean wave spectra into individual wave systems (partitions). Each partition represents a distinct wave system characterized by significant wave height (Hs), peak period (Tp), and direction (Dp).

**WASP handles:**

- ✅ Spectral partitioning using watershed algorithm
- ✅ Processing SAR (Sentinel) and WW3 model spectra
- ✅ Extracting wave parameters (Hs, Tp, Dp) for each partition

👉 **For analysis and validation**, use the repository [**HIVE**](https://github.com/jtcarvalho/hive)

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Using pip (Virtual Environment)

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
