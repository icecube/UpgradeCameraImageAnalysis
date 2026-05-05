# ICUcamera Noise Reduced Report Generator

**Comprehensive Image Analysis Tool for IceCube Upgrade Camera Raw Data**

## Overview

**ICUcamera Noise Reduced Report Generator** analyzes raw IceCube Upgrade data from mDOM and DEgg camera images.

### What It Does

This Python script takes a single raw camera image (.raw file) and automatically generates:

1. **32 PNG Images** - 4 stretching methods (linear, log, asinh, gamma) × 8 visualization channels (RGB, Grey, Red, Blue, Green, Pedestal-Sub, B-R Sub, B-G Sub)
2. **one PDF Report** - General info, raw data analysis, RGGB channels, standard visualizations, saturation analysis
3. **Summary Panel** - 2×4 grid showing all 8 channels for quick overview

Each image includes ADU(Analog-to-Digital Unit)-scaled colorbars (actual sensor values, not normalized) and X/Y axis labels for spatial reference.

Raw sensor data requires demosaicing (Bayer → RGB), pedestal subtraction (remove baseline noise), and multiple visualizations to reveal different features. This tool automates the entire process with updated output.

**Authors:**
- Shouvik Mondal (smondal@icecube.wisc.edu)
- Based on original scripts by Seowon Choi (schoi1@icecube.wisc.edu)

---

## Features

### Main Functionalities

-  **Bayer Demosaicing:** RGGB channel extraction from raw 12-bit data
-  **Color Correction:** White balance and saturation correction via ICUCamera library
-  **Noise Reduced Output:** PNG and PDF with colorbars and axis labels
-  **32 Image Combinations:** 4 stretches × 8 channels processed simultaneously
-  **Statistical Analysis:** Channel histograms, saturation maps, metadata extraction
-  **Windows Compatible:** Short filenames (img_001, etc.) avoid 260-char path limits for windows OS

### Visualization Options

- **8 Channels:** RGB (natural color), Grey (grayscale), Red/Blue/Green (individual colors), Pedestal-Sub (dark signal), B-R Sub & B-G Sub (color differences)
- **4 Stretching Methods:**
  - Linear: Standard percentile mapping [0-1]
  - Logarithmic: Faint feature revelation via log compression
  - Asinh: Smooth nonlinear for balanced detail
  - Gamma: Power-law brightness adjustment

---

## Installation

### Prerequisites

```bash
# Install required packages
pip install numpy matplotlib opencv-python

# Verify Python version
python --version  # Should be 3.7+
