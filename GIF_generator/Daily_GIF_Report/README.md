# 📸 IceCube Upgrade Camera: Automated Timelapse Generator

**Author:** Shouvik Mondal based on Seowon Choi's scripts
**Affiliation:** Department of Physics & Astronomy, University of Utah  
**Experiment:** IceCube Neutrino Observatory (IceCube Upgrade)  
**Version:** v1.0

---

## 📖 Project Overview

This repository hosts the automated data pipeline designed to monitor and visualize the **IceCube Upgrade Camera System**. As new calibration data is acquired from the camera modules from South Pole, this system automatically processes the raw binary data into visualizable timelapse GIFs.

The pipeline runs daily on the cluster, scans for fresh data arrivals, decodes the proprietary sensor formats, applies image enhancement algorithms suitable for low-light environments, and notifies the operator via email with a status report.

### 🎯 Key Objectives
1.  **Automation:** Remove the need for manual file checking and processing.
2.  **Visualization:** Convert raw, dark, high-dynamic-range sensor data into clear, human-readable images.
3.  **Monitoring:** Provide daily verification that cameras are functioning and data is being written correctly.

---
### Features:
### Incremental Processing
* This script checks existing GIFs against the current raw data.
* **If a GIF exists:** It counts the frames. If the raw data has *more* files than the GIF has frames, it appends only the new images.
* **If no GIF exists:** It creates one from scratch.
* **If up-to-date:** It skips the file entirely, saving hours of CPU time.

###  Advanced Image Processing Pipeline
Raw data from the IceCube cameras is not immediately viewable. The script performs the following transformation chain using `OpenCV` and `NumPy`:

1.  **Decompression:** Extracts `.raw` files from daily `.tar.gz` archives.
2.  **Decoding:** Uses the custom `ICUCamera` library to parse the binary sensor stream.
3.  **De-Bayering:** Converts the raw Bayer-pattern sensor data into RGB color space.
4.  **ASINH Stretching:** Applies an **Inverse Hyperbolic Sine (asinh)** stretch.
    * *Why?* The ice is extremely dark, but calibration LEDs are bright. A linear scale would make the ice invisible. ASINH allows us to see faint details in the dark background without saturating the bright light sources.
5.  **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Further enhances local contrast to bring out features in the drill hole ice.

### Geometry & Metadata Overlay
Every frame is stamped with critical metadata for analysis:
* **Timestamp:** Date and time of capture.
* **Geometry:** String Number and DOM/Port Number.
* **Depth:** Automatically maps the Port Number to the specific depth (in meters) using a lookup table (`comm_modules - Sheet1.csv`).

---

## 📂 Directory Structure

The system assumes the following directory hierarchy on the cluster:

```text
/data/user/smondal/
├── automated_GIF_generator_with_email_v1.4.py  # MAIN SCRIPT
├── ICUCamera.py                                # Decoding Library
├── comm_modules - Sheet1.csv                   # Depth mapping database
├── directory_history.txt                       # State file (tracks seen folders)
├── gif_log.txt                                 # Runtime logs
└── gif_env/                                    # Python Virtual Environment
