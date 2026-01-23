# IceCube Upgrade Image Analyzer (v1.0)

**Written by:** Shouvik Mondal (smondal@icecube.wisc.edu)  
**Based on logic by:** Seowon Choi (schoi1@icecube.wisc.edu)  
**Language:** Python 3.9+  

---

## Overview
The **IceCube Upgrade Image Analyzer** is a specialized GUI tool designed for analyzing raw sensor data from **IceCube Upgrade cameras**. 

It serves as a bridge between raw scientific data and visual inspection, providing real-time debayering, scientific thresholding (12-bit), and channel-specific inspection (RGGB) for `.RAW` and `.tar.gz` datasets.



## Key Features

### 1. Dynamic File Loading
* **Variable window heights:** Automatically detects and loads raw files with variable window heights (e.g., 979px vs 993px) without crashing.
* **Archive Support:** Drag & drop `.tar.gz` files directly; the tool extracts the correct raw image automatically.

### 2. Scientific Inspection
* **8-bit Thresholding:** The threshold slider operates on the full **0–255** sensor range. Pixels exceeding the threshold (e.g., saturation > 255) are highlighted in **Red**.
* **Raw Channel View:** Isolate **Green 1** and **Green 2** raw pixels independently to identify row/column-specific sensor noise.
* **Smart Gray:** Uses the `ICUCamera.py` algorithm to mathematically recover saturated pixels using Red/Green channel correlation.

### 3. Real-Time Dashboard
* **Live Histogram:** Log-scale pixel intensity distribution.
* **FFT Spectrum:** Frequency analysis for noise pattern detection.
* **Pixel Inspection:** Hover over any pixel to see its raw (X, Y) coordinates and exact 12-bit value.

---

## Requirements

* **Python Version:** 3.6+ (Recommended 3.9+)
* **Operating System:** Windows, Linux, or macOS

### Dependencies
Install the required libraries using pip:

```bash
pip install opencv-python numpy Pillow matplotlib tkinterdnd2
