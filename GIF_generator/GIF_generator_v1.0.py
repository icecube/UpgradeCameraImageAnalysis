# IceCube Upgrade Image GIF file generator(VISUAL PURPOSE ONLY)
# Version: v1.1 (One Image Per Day + Fixed Timestamp + Fixed Duration)
# Last Updated: 03 Feb 2026
# Written by: Shouvik Mondal(shouvik.mondal@utah.edu) based on Seowon Choi's scripts(schoi1@icecube.wisc.edu)

import os #help search the subfolders
import glob #help search the subfolders
import cv2  #OpenCV library for image processing
import numpy as np
import re
from PIL import Image

# === IMPORT ICUCamera library==== make sure you run this script in the same directory as ICUCamera.py ===
try:
    import ICUCamera as icuc  #specific to ICU_camera settings
except ImportError:
    icuc = None

# ==================== CONFIGURATION ====================
root_dir = r"E:\ICUCamera_data" #all the different run directories are saved in a same directory like the input"E:\ICUCamera_data"
####################"E:\ICUCamera_data\0102";"E:\ICUCamera_data\0105";"E:\ICUCamera_data\0108"...etc.
input_file = "Camera-Run_IIB_string88_mDOM_port5274_cam1_illum1_gain0_exposure3700ms" #change it as your input file name
output_gif = r"E:\ICUCamera_data\Camera-Run_IIB_string88_mDOM_port5274_cam1_illum1_gain0_exposure3700ms.gif"
#change it as your output dir/file name

# Duration of EACH frame in milliseconds
FRAME_DURATION_MS = 500
# =======================================================

def asinh_stretch(x, p_black=0.5, p_white=99.9, a=15.0):
    """Directly converts raw data (float or int) to 8-bit using the ICUCamera's logic."""
    x = x.astype(np.float32)
    black = np.percentile(x, p_black)
    white = np.percentile(x, p_white)
    # Normalize 0-1
    x = (x - black) / max(white - black, 1e-6)
    x = np.clip(x, 0, 1)
    # Apply Asinh curve
    y = np.arcsinh(a * x) / np.arcsinh(a)
    # Returns valid 8-bit integer (0-255)
    return (y * 255).astype(np.uint8)

def clahe(bgr_img, clipLimit=2.0):  #CLAHE stands for "Contrast Limited Adaptive Histogram Equalization"
    """Applies CLAHE to 8-bit images."""
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
# --------------------------------------------------------

def main():
    print(f"Going through {root_dir}...")
    search_pattern = os.path.join(root_dir, "**", f"*{input_file}*.raw")
    found_files = glob.glob(search_pattern, recursive=True)
    
    # === PER DAY image FILTERING ===
    files_by_date = {}
    
    for f in found_files:
        if not os.path.isfile(f): continue
        filename = os.path.basename(f)
        
        # Extract Date (YYYYMMDD) using Regex
        # Matches the first 8 digits in the timestamp pattern
        match = re.search(r"(\d{8})-\d{2}-\d{2}-\d{2}", filename)
        
        if match:
            date_key = match.group(1) # e.g., "20260123"
            
            if date_key not in files_by_date:
                files_by_date[date_key] = []
            files_by_date[date_key].append(f)
        else:
            # If no date found , treat as unique images using filename
            files_by_date[filename] = [f]
    
    unique_files = []
    
    # Sort by date (chronological order)
    for date_key in sorted(files_by_date.keys()):
        # Sort files for that specific day (earliest time first)
        daily_files = sorted(files_by_date[date_key])
        
        # TAKE ONLY THE FIRST FILE OF THE DAY
        first_file = daily_files[0]
        unique_files.append(first_file)
        
        # Debug Info(if theres no file or more than one)
        if len(daily_files) > 1:
            print(f"  [Filtered] Date {date_key}: Found {len(daily_files)} images. Keeping only: {os.path.basename(first_file)}")

    print(f"\nProcessing {len(unique_files)} unique frames (1 per day)...")
    # ==================================
    
    frames = []
    
    for i, filepath in enumerate(unique_files):
        filename = os.path.basename(filepath)
        print(f"[{i+1}/{len(unique_files)}] Processing: {filename}...", end="\r")
        
        try:
            if not icuc: break
            result = icuc.Raw2Npy(filepath)
            if result is None: continue
            _, raw_data = result

            # Applying the defined functions(you can use more like brightness, contrast etc.)
            # Convert's to 8-bit file and It handles any input type (float/int) 
            # and forces it to a standard 0-255 image.
            # We will use a=15.0 (Standard stretch) or a=40.0 (Strong stretch for dark ice images)
            bayer = asinh_stretch(raw_data, p_black=0.5, a=40.0)

            # Debayering an image file (8-bit -> 8-bit BGR), opencv supports 8-bit img files
            bgr = cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2BGR)

            # apply CLAHE
            final_bgr = clahe(bgr, clipLimit=3.0)

            # Add Timestamp information
            try:
                match = re.search(r"(\d{8}-\d{2}-\d{2}-\d{2})", filename)
                if match:
                    ts_str = match.group(1)
                    date_text = f"{ts_str[:4]}-{ts_str[4:6]}-{ts_str[6:8]} {ts_str[9:11]}:{ts_str[12:14]}:{ts_str[15:17]}"
                    cv2.putText(final_bgr, date_text, (30, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            except:
                pass 

            # GIF file creation

            final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(final_rgb)
            frames.append(pil_img.convert("P", palette=Image.ADAPTIVE))
            
        except Exception as e:
            print(f"\n  [Skipping] {filename}: {e}")

    print(f"\n\nSaving GIF with {len(frames)} frames...")
    
    if frames:
        durations = [FRAME_DURATION_MS] * len(frames)
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=True
        )
        print(f"Successfully made the GIF file! Output: {output_gif}")
    else:
        print("No frames processed.")

if __name__ == "__main__":
    main()
