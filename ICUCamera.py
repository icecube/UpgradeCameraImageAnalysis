# ============================================================
# Version and Contact Info
# ============================================================
ver = "1.0.0"
# This script supersedes PrintingTools.py
# Written by Seowon Choi [schoi1@icecube.wisc.edu] / [choi940927@gmail.com]

import os
import glob
import gzip
from struct import unpack_from

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import cv2
from astropy.io import fits




def get_version():
    return "v"+ver

def _decompress_aggregate_readout(buffer: bytes) -> bytes:
    # Unlike FAT or CAT systems, the DOM SW uses a different format to store the compressed image data.

    #uncompressed_common_header = buffer[: CommonHeader.length()]
    #compressed_data = buffer[CommonHeader.length() :]
    headerlength = 12  # Originally, the CommonHeader.length() is defined in STM32Tools/xdomapp.py, but to avoid the dependency, I just hardcoded it here.
    uncompressed_common_header = buffer[: headerlength] 
    compressed_data = buffer[headerlength :]

    print(f"Decompressing {len(compressed_data)} bytes of compressed image data")
    return uncompressed_common_header + gzip.decompress(compressed_data)


def Raw2Npy(filename):

    # Read your file (.RAW or .gz or etc.) into 2D numpy array of size:
    H_size = 1312
    candidate_V_sizes = [979, 993] # depends on your windowing

    # different file formats require different reading methods
    # First step is to read the file into a 1D numpy array
    if filename.endswith(".RAW"):
        #print("Image format : .RAW")
        np_1d = np.fromfile(filename, dtype=np.uint16)

    elif filename.endswith(".gz"):
        #print("Image format : .RAW.gz")
        with gzip.open(filename, 'rb') as f:
            data = f.read()
        np_1d = np.frombuffer(data, np.uint16)
    else:
        try:
            with open(filename, "rb") as f:
                buffer = f.read()
            buffer = _decompress_aggregate_readout(buffer)[40:]  # header is 40 bytes long
            np_1d = np.frombuffer(buffer, np.uint16)
        except Exception as e:
            raise ValueError(f"Unsupported file format: {e}")

    np_1d = np_1d >> 4  # Convert from 16-bit to 12-bit (original bit depth of the image sensor is 12-bit)

    # Second step is to reshape the 1D numpy array into a 2D numpy array with correct size/shape
    for V_size in candidate_V_sizes:
        if len(np_1d) == H_size * V_size:
            np_2d = np.reshape(np_1d, (V_size, H_size), 'C')
            print(f"image size: {H_size} * {V_size}")
            
            return (np_2d.shape, np_2d)

    print("Invalid image size. Please check your image size")
    return None

def header_info(filename):
    with open(filename, "rb") as f:
        data = _decompress_aggregate_readout(f.read())
        # Unpack header
        hdr = data[:40]
        (
            nblo,
            nbhi,
            rectype,
            icm_lo,
            icm_hi,
            encdesc,
            dp_status,
            *cam_id,
            cam_num,
            capture_mode,
            spx,
            spy,
            wx,
            wy,
            vob,
            gain,
            conv_mode,
            inv_mode,
            exposure,
            enable_mask,
            heater_state,
        ) = unpack_from("<HBBLHBB8BBBHHHHBHBBHBB", hdr)
        nbytes = nbhi << 16 | nblo
        icm_ts = icm_hi << 32 | icm_lo
        cam_id_str = "".join(["%x" % c for c in cam_id])
        return(f"""Image length: {nbytes} B
                Record type: 0x{rectype:02X}
                ICM timestamp: {icm_ts}
                Encoding desc: {encdesc}
                DP status: 0x{dp_status:02X}
                Camera ID: {cam_id_str}
                Camera number: {cam_num}
                Capture mode: {capture_mode}
                Custom window: {spx}, {spy}, {wx}, {wy}, {vob}
                Gain: {gain}
                Conversion mode: {conv_mode}
                Inversion mode: {inv_mode}
                Exposure: {exposure} msec
                Enable mask: 0x{enable_mask:02X}
                Heater status: {"ON" if heater_state else "off"}"""
                )


################################################################################################################################################

def Npy2Bgr(npy):
    npy = npy >> 4                                                          # Yes, you need to bit shift, 
    bgr = cv2.cvtColor(npy.astype(np.uint8), cv2.COLOR_BAYER_BG2BGR)        # Because the cv2.cvtColor reauires all values in uint8
    return bgr

def Npy2Rgb(npy):
    npy = npy >> 4                                                          # Just in case you need RGB, rather than BGR
    rgb = cv2.cvtColor(npy.astype(np.uint8), cv2.COLOR_BAYER_BG2RGB)        
    return rgb


# However, since the Bayer (RGGB) has 2 Green, 1 Red, and 1 Blue pixels, the image will appear greenish after the convert above.
# Therefore, it is typical to suppress Green, and boost up Red and Blue colors
######## the input image should be in BGR format, not in RGB channel format ##########

def BgrCorrection(bgr):
    # Color Correction Weights
    correction_factors = np.array([1.5, 0.8, 1.5])                          # The plan is to multiply different factros for each color channels. here 1.5, 0.8, and 1.5 are selected manually. You can change as you want.
    #bgr=np.clip(bgr,0,255)                                                 # If value exceeds 255 (maximum of 8-bit), Clip it!
    #for i in range(3):
    #    bgr:,:,i] = np.clip(rgb[:,:,i] * correction_factors[i], 0, 255)
    bgr = bgr * correction_factors[None, None, :]                           
    bgr = np.clip(bgr, 0, 255)
    bgr = bgr.astype('uint8')                                                # The dtype of the multiplication above will be float32. change back to uint8.
    return bgr


################################################################################################################################################

# If the image is too dark (due to short exposure, low luminous condition, etc..) you can modify the brightness with the script here.
## Two different approaches are given here.

def BrightenBgr(bgr, alpha=1.2, beta=30):
    """
    alpha: contrast (bigger than 1.0)
    beta: brightness shift (0~100)
    """
    bright = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
    return bright

################################################################################################################################################

def BrightenHSV(bgr, factor=1.5):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] = hsv[:,:,2] * factor
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    hsv = hsv.astype(np.uint8)
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bright


################################################################################################################################################

# The image sensor has color of RGGB Bayer pattern. And these scripts are to demosaic them, and create 3-channel image
# Numpy array -> BGR channel image
# Check https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html for Bayer -> BGR 


def Npy2Rgb_numpy(npy):
    # Only for visualization purpose
    # npy: 2D numpy array of raw image data
    # 12bit → 8bit
    npy = (npy >> 4).astype(np.float32)

    H, W = npy.shape

    R  = np.zeros_like(npy)
    G  = np.zeros_like(npy)
    B  = np.zeros_like(npy)

    # RGGB
    R[0::2, 0::2] = npy[0::2, 0::2]
    G[0::2, 1::2] = npy[0::2, 1::2]
    G[1::2, 0::2] = npy[1::2, 0::2]
    B[1::2, 1::2] = npy[1::2, 1::2]

    # very simple interpolation
    def interp(channel):
        mask = channel == 0
        channel[mask] = np.mean(channel[~mask])
        return channel

    R = interp(R)
    G = interp(G)
    B = interp(B)

    rgb = np.stack([R, G, B], axis=-1).astype(np.uint8)
    return rgb

def get_gray(npy, pedestal=235.0, sat_thr=4090, dead_value=None):
    # This script corrects the saturated blue pixels using the other channels.
    # crop to even shape
    H, W = npy.shape
    npy = npy[:H - (H % 2), :W - (W % 2)]   # To make sure even shape between 4 channels

    B  = npy[1::2, 1::2].astype(np.float32)
    G1 = npy[0::2, 1::2].astype(np.float32)
    G2 = npy[1::2, 0::2].astype(np.float32)
    R  = npy[0::2, 0::2].astype(np.float32)

    # masks
    B_sat  = B  >= sat_thr
    #print(B_sat.sum())
    G_bad  = (G1 >= sat_thr) | (G2 >= sat_thr)   # if any of G1/G2 is sat, G is bad
    R_sat  = R  >= sat_thr

    # G to use only when not bad
    G_use = 0.5 * (G1 + G2)

    P = float(pedestal)
    #P = float(np.min(npy))

    # scaled-to-B estimates (pedestal 제거 후 스케일링)
    weights = {"G":10.0, "R":25.0}
    B_from_R = weights["R"] * (R - P)
    B_from_G = weights["G"] * (G_use - P)

    B_new = B.copy()

    # case 1: B sat, but G & R usable -> mix
    m = B_sat & (~G_bad) & (~R_sat)
    B_new[m] = 0.5 * (B_from_G[m] + B_from_R[m])

    # case 2: B & G bad, R usable -> R only
    m = B_sat & (G_bad) & (~R_sat)
    B_new[m] = B_from_R[m]

    # case 3: B sat, G usable, R sat -> G only  
    # Don't expect this case to be happen.
    # If this case happens, check the channel division first, and then illuminate conditions.
    m = B_sat & (~G_bad) & (R_sat)
    if np.sum(m) > 0: print(f"B & G usable, but R saturated: {np.sum(m)}. \nCheck channel division and illumination conditions.")
    B_new[m] = B_from_G[m]

    # case 4: B sat, G bad, R sat -> unrecoverable marker
    m = B_sat & (G_bad) & (R_sat)
    if dead_value is None:
        B_new[m] = weights["R"] * (4094.0 - P)
    else:
        B_new[m] = dead_value

    return B_new



################################################################################################################################################
# Under Development ###

def Npy2Bgr16(npy, bayer_code=cv2.COLOR_BAYER_BG2BGR):
    if npy.dtype != np.uint16:
        npy = npy.astype(np.uint16)
    bgr16 = cv2.cvtColor(npy, bayer_code)  
    return bgr16

def to_uint8(bgr01):
    return (np.clip(bgr01, 0, 1) * 255.0 + 0.5).astype(np.uint8)

def asinh_stretch01(x, p_black=0.5, p_white=99.9, a=15.0):
    x = x.astype(np.float32)
    black = np.percentile(x, p_black)
    white = np.percentile(x, p_white)
    x = (x - black) / max(white - black, 1e-6)
    x = np.clip(x, 0, 1)
    y = np.arcsinh(a * x) / np.arcsinh(a)
    return y

def stretch_preserve_color(bgr16, p_black=0.5, p_white=99.9, a=15.0):
    bgr = bgr16.astype(np.float32)
    wR, wG, wB = 4, 10, 100
    lum = (wB*bgr[:,:,0] + wG*bgr[:,:,1] + wR*bgr[:,:,2]) / (wB + wG + wR)
    lum_s = asinh_stretch01(lum, p_black=p_black, p_white=p_white, a=a)
    lum_safe = np.maximum(lum, 1e-6)
    bgr_out = bgr * (lum_s / lum_safe)[:,:,None]
    scale = np.percentile(bgr_out, 99.9) + 1e-6
    bgr01 = np.clip(bgr_out / scale, 0, 1)
    return bgr01

def log_stretch(x, p_black=0.5, p_white=99.9, k=300.0):
    x = x.astype(np.float32)
    black = np.percentile(x, p_black)
    white = np.percentile(x, p_white)
    x = (x - black) / max(white - black, 1e-6)
    x = np.clip(x, 0, 1)
    y = np.log1p(k * x) / np.log1p(k)
    return y

def stretch_preserve_color_log(bgr16, p_black=0.5, p_white=99.9, k=300.0):
    bgr = bgr16.astype(np.float32)
    wR, wG, wB = 4, 10, 100
    lum = (wB*bgr[:,:,0] + wG*bgr[:,:,1] + wR*bgr[:,:,2]) / (wB + wG + wR)
    lum_s = log_stretch(lum, p_black=p_black, p_white=p_white, k=k)
    lum_safe = np.maximum(lum, 1e-6)
    bgr_out = bgr * (lum_s / lum_safe)[:,:,None]
    scale = np.percentile(bgr_out, 99.9) + 1e-6
    bgr01 = np.clip(bgr_out / scale, 0, 1)
    return bgr01

def clahe_on_l_channel(bgr8, clipLimit=2.0, tileGridSize=(8,8)):
    lab = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2,a,b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def gamma(img01, gamma=0.7):
    img01 = np.clip(img01, 0, 1).astype(np.float32)
    return np.power(img01, gamma)

def save_fits(filepath, savedir):
    shape, npy = Raw2Npy(filepath)
    hdr_txt = header_info(filepath)
    print(hdr_txt)

    hdu = fits.PrimaryHDU(npy)
    hdr = hdu.header
    
    # Bayer pattern metadata
    hdr['BAYERPAT'] = 'RGGB'
    hdr['XBAYROFF'] = 0
    hdr['YBAYROFF'] = 0
    
    # Optional but useful
    hdr['COLOR'] = True
    hdr['BITPIX'] = 16

    ##other metadata for header
    hdr['EXPTIME']= 0.1
    ###you can define any other header information here...
    
    hdu.writeto(savedir + filepath.split("/")[-1].replace(".raw",".fits"), overwrite=True)