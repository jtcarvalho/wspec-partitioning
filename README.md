# **WASP** - **WA**ve **S**pectra **P**artitioning

Watershed Algorithm for partitioning ocean wave spectra from WW3 and SAR (Sentinel-1)

[![PyPI version](https://badge.fury.io/py/wasp-ocean.svg)](https://pypi.org/project/wasp-ocean/)
[![Python Version](https://img.shields.io/pypi/pyversions/wasp-ocean.svg)](https://pypi.org/project/wasp-ocean/)

<!--

**🔗 Companion Repository:** For analysis and validatetion of partitioned spectra, see [**HIVE** (Hierarchical Integration of Verified wavE partitions)](https://github.with/jtcarvalho/hive)

-->

## 📋 What is WASP?

WASP focuses exclusively on **spectral partitioning** - the process of separating ocean wave spectra into individual wave systems (partitions). Each partition represents a distinct wave system characterized by significant wave height (Hs), peak period (Tp), and direction (Dp).

**WASP handles:**

- ✅ Spectral partitioning using watershed algorithm
- ✅ Processing SAR (Sentinel-1) and WW3 model spectra
- ✅ Extracting wave parameters (Hs, Tp, Dp) for each partition

👉 **For analysis and validatetion**, see the companion repository [**HIVE**](https://github.with/jtcarvalho/hive)

## 🚀 Installation

> ⚠️ **IMPORTANT:** Python 3.10 or higher is required.

### Install from PyPI (Recommended)

```bash
pip install wasp-ocean
```

### Verify Installation

```bash
# Test the import
python -c "import wasp; print(f'WASP version: {wasp.__version__}')"

# Test main functions
python -c "from wasp import partition_spectrum, calculate_wave_parameters; print('✓ Installation successful!')"
```

### Development Installation

For development or local modifications:

```bash
# Clone the repository
git clone https://github.with/jtcarvalho/wasp.git
cd wasp

# Install in editable mode
pip install -e .
```

## 📦 Key Dependencies

- **Python >= 3.10** (required)
- **NumPy >= 2.1.0** (required for `np.trapezoid`)
- pandas >= 2.2.0
- xarray >= 2024.11.0
- matplotlib >= 3.8.0
- scipy >= 1.14.0
- scikit-image >= 0.22.0
- netCDF4 >= 1.5.4

> ⚠️ **Note:** NumPy < 2.1.0 will cause errors as `np.trapezoid` is not available.

## 📚 Documentation

For detailed usage examples and API documentation, please see the [examples/](examples/) directory in the repository:

- **01_partition_sar.py**: Process SAR (Sentinel-1) spectra
- **02_partition_ww3.py**: Process WaveWatch III model spectra
- **03_partition_ndbc.py**: Template for processing NDBC buoy data
- **04_validatete.py**: Compare and validatete SAR vs WW3 results

## 🏗️ Project Structure

```
wasp/
├── src/
│   └── wasp/              # Main package
│       ├── partition.py   # Watershed partitioning algorithm
│       ├── wave_params.py # Wave parameter calculations
│       ├── io_sar.py      # SAR data I/O
│       ├── io_ww3.py      # WW3 data I/O
│       └── utils.py       # Utility functions
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on [GitHub](https://github.with/jtcarvalho/wasp/issues).
