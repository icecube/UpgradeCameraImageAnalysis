# IceCube Upgrade Image GIF file generator(VISUAL PURPOSE ONLY)

**Version:** v1.1  
**Last Updated:** 03 Feb 2026  
**Author:** Shouvik Mondal (smondal@icecube.wisc.edu)  
**Based on:** Seowon Choi's scripts (schoi1@icecube.wisc.edu)

---

### === Prerequisites ===

* **Libraries:** `os`, `glob`, `cv2` (OpenCV), `numpy`, `PIL` (Pillow)
* **ICUCamera:** Make sure you run this script in the same directory as `ICUCamera.py` (specific to ICU_camera settings).

---

### === INPUT DIRECTORY settings ===

Configure these variables in the script before running:

* **`root_dir`**: All the different run directories are saved in a same directory like the input.
    * *Example:* `"E:\ICUCamera_data"`
    * *Subfolders:* `0102`, `0105`, `0108`, etc.
* **`input_file`**: The specific file identifier to search for (e.g., `"Camera-Run_IIIB_string88_mDOM_port5244_cam2_illum2_gain0_exposure3700ms"`).
* **`output_gif`**: The path for the final output GIF.
* **`frame_duration`**: Speed of the animation (e.g., `500`).

---

### === Transformation Functions ===

#### `asinh_stretch`
* **Description:** Non-linear transformation of raw pixels.
* **Logic:** Directly converts raw data (float or int) to 8-bit using the ICUCamera's logic.
    * Normalizes 0-1.
    * Applies Asinh curve.
    * Returns valid 8-bit integer (0-255).

#### `clahe`
* **Description:** CLAHE stands for "Contrast Limited Adaptive Histogram Equalization".
* **Logic:** Applies CLAHE to 8-bit images.
    * Converts BGR to LAB.
    * Applies CLAHE to the L-channel.
    * Merges and converts back to BGR.

---

### === Main Execution Flow ===

1.  **Search:** Going through `root_dir` using `glob` to help search the subfolders recursively.
2.  **Filter:** Filter 'trial0' files from the list.
3.  **Processing Loop:**
    * **Read RAW:** Uses `ICUCAMera.py` Libraries (`Raw2Npy`).
    * **Stretch:** Converts to 8-bit file. It handles any input type (float/int) and forces it to a standard 0-255 image.
        * *Settings:* Uses `a=40.0` (Strong stretch for dark ice images).
    * **Debayer:** Debayering an image file (8-bit -> 8-bit BGR), opencv supports 8-bit img files.
    * **Enhance:** Apply CLAHE (`clipLimit=3.0`).
    * **Create GIF:** Converts BGR to RGB, converts to PIL Image (Adaptive Palette), and appends to frame list.
4.  **Save:** Saving GIF with `save_all=True` and `optimize=True`.
