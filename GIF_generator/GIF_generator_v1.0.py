# IceCube Upgrade Image GIF file generator(VISUAL PURPOSE ONLY)
# Version: v1.0
#Last Updated: 01 Feb 2026
# Written by: Shouvik Mondal(smondal@icecube.wisc.edu) based on Seowon Choi's[schoi1@icecube.wisc.edu] scripts

import os  #help search the subfolders
import glob   #help search the subfolders
import cv2      #OpenCV library for image processing
import numpy as np
from PIL import Image

# === make sure you run this script in the same directory as ICUCamera.py ===
try:
    import ICUCamera as icuc  #specific to ICU_camera settings
except ImportError:
    icuc = None

# ==================== INPUT DIRECTORY settings ====================
root_dir = r"E:\ICUCamera_data" #all the different run directories are saved in a same directory like the input"E:\ICUCamera_data"
####################"E:\ICUCamera_data\0102";"E:\ICUCamera_data\0105";"E:\ICUCamera_data\0108"...etc.
input_file = "Camera-Run_IIIB_string88_mDOM_port5244_cam2_illum2_gain0_exposure3700ms"
output_gif = r"E:\ICUCamera_data\Gif_files_for_press_release\timelapse_Run_IIIB_string88_mDOM_port5244_cam2_illum2_gain0_exposure3700ms.gif"
frame_duration = 500
# =======================================================

# non-linear transformation of raw pixels
def asinh_stretch(x, p_black=0.5, p_white=99.9, a=15.0):
    """
    Directly converts raw data (float or int) to 8-bit using the ICUCamera's logic.
    """
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

def clahe(bgr_img, clipLimit=2.0): #CLAHE stands for "Contrast Limited Adaptive Histogram Equalization"
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
    
    # Filter 'trial0' files
    valid_files = sorted([f for f in found_files if os.path.isfile(f) and "trial0" not in f.lower()])
    print(f"Found {len(valid_files)} valid files. Processing the images...")
    
    frames = []
    
    for i, filepath in enumerate(valid_files):
        filename = os.path.basename(filepath)
        print(f"[{i+1}/{len(valid_files)}] Processing: {filename}...", end="\r")
        
        try:
            # READ RAW (Using ICUCAMera.py Libraries)
            if icuc:
                result = icuc.Raw2Npy(filepath)
            else:
                print("\n[Error] ICUCamera.py missing.")
                break
                
            if result is None: continue
            _, raw_data = result

            # Convert's to 8-bit file and It handles any input type (float/int) 
            # and forces it to a standard 0-255 image.
            # We will use a=15.0 (Standard stretch) or a=40.0 (Strong stretch for dark ice images)
            bayer = asinh_stretch(raw_data, p_black=0.5, a=40.0)

            # Debayering an image file (8-bit -> 8-bit BGR), opencv supports 8-bit img files
            bgr = cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2BGR)
            
            # 4. apply CLAHE
            final_bgr = clahe(bgr, clipLimit=3.0)
            
            # 5. GIF file creation
            final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(final_rgb)
            frames.append(pil_img.convert("P", palette=Image.ADAPTIVE))
            
        except Exception as e:
            print(f"\n  [ERROR] {filename}: {e}")

    print(f"\n\nSaving GIF with {len(frames)} frames...")
    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True
        )
        print(f"Successfully saved the gif file: {output_gif}")
    else:
        print("No frames are being processed.")

if __name__ == "__main__":
    main()

