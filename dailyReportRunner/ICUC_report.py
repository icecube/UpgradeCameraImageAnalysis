# ============================================================
# Version and Contact Info
# ============================================================
# version 1.0.0
ver = "v1.0.3"
# Published @ 30DEC2025
# Last Updated @ 15JAN2026
# Originally written by Seowon Choi [schoi1@icecube.wisc.edu] / [choi940927@gmail.com]

#fname = "/Users/seowonchoi/Documents/NAPPL/Operation/drts/ICUCamera4Pole/real_data/Camera-images_cam1/Run_IIB_string87_mDOM_port5306_cam1_illum1_gain0_exposure3700ms_20260102-05-33-39.raw"
# ============================================================
import ICUCamera as icuc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import argparse
import cv2
import csv
from pathlib import Path


def make_report(fname, outputdir):

    # fname = "/home/jtorresespinosa/jtorresespinosa/UpgradeCamera/UpgradeCameraImageAnalysis/dailyReportRunner/output/Camera-Run_IIB_string87_mDOM_port5306_cam1_illum1_gain0_exposure3700ms_20260112-19-47-11.raw"
    rawname = fname.split("/")[-1]
    output = os.path.join(outputdir, rawname.replace(".raw", "_report.pdf"))

    if "dark" in rawname.lower(): dark_flag = True
    else: dark_flag = False

    # ============================================================

    def add_section_title(fig, ref_ax, title, x=0.02, dy=0.012, fontsize=16):
        bb = ref_ax.get_position()
        y = min(bb.y1 + dy, 0.995)
        fig.text(x, y, title, fontsize=fontsize, fontweight="bold",
                ha="left", va="bottom")

    # -------------------------------
    # [0] Header Info
    hdr_txt = icuc.header_info(fname)
    common_hdr_lst = hdr_txt.split("\n")[0:5]
    common_hdr_txt = "\n".join(line.strip() for line in common_hdr_lst)
    camera_hdr_lst = hdr_txt.split("\n")[5:]
    camera_hdr_txt = "\n".join(line.strip() for line in camera_hdr_lst)

    # -------------------------------
    # [1] Original Data
    shape, npy = icuc.Raw2Npy(fname)

    B_channel_flat  = npy[1::2, 1::2].astype(np.float32).flatten()
    G1_channel_flat = npy[0::2, 1::2].astype(np.float32).flatten()
    G2_channel_flat = npy[1::2, 0::2].astype(np.float32).flatten()
    R_channel_flat  = npy[0::2, 0::2].astype(np.float32).flatten()

    R_channel  = npy[0::2, 0::2]
    G1_channel = npy[0::2, 1::2]
    G2_channel = npy[1::2, 0::2]
    B_channel  = npy[1::2, 1::2]

    # -------------------------------
    # [3] Visualization Purpose Only
    gray = icuc.get_gray(npy)
    rgb  = icuc.Npy2Rgb_numpy(npy)
    #### or you can use above or below as you want
    bgr16 = icuc.Npy2Bgr16(npy, bayer_code=cv2.COLOR_BAYER_BG2BGR)
    bgr01_asinh = icuc.stretch_preserve_color(bgr16, p_black=0.1, p_white=99, a=40.0)
    out8_asinh = icuc.to_uint8(bgr01_asinh)
    out8_asinh_clahe = icuc.clahe_on_l_channel(out8_asinh, clipLimit=3.0)
    rgb = cv2.cvtColor(out8_asinh_clahe, cv2.COLOR_BGR2RGB)
    # ============================================================
    # Generating PDF
    # ============================================================
    FIG_W = 12
    height_ratios = [2.5, 5.0, 12.0, 5.0, 10.0] 
    FIG_H = float(np.sum(height_ratios))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=height_ratios, hspace=0.35)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.02)

    # ============================================================
    # [0] Header Info (1 row, 2 cols)
    # ============================================================
    gs0 = gs[0].subgridspec(2, 2, height_ratios=[0.5, 1.0],wspace=0.1, hspace=0.25)

    ax0_txt = fig.add_subplot(gs0[0, :]) 
    ax0_txt.axis("off")
    ax0_txt.text(0.0, 1.0, f"File name : {rawname}\nResolution : {shape}", ha="left", va="top", fontsize=13, fontweight='bold')

    ax1_l = fig.add_subplot(gs0[1, 0])
    ax1_r = fig.add_subplot(gs0[1, 1])
    ax1_l.text(0.0, 0.95, common_hdr_txt, va="top", ha="left", fontsize=12, family="monospace")
    ax1_l.axis("off")
    ax1_l.set_title("[Common Header]", loc="left", pad=6)
    ax1_r.text(0.0, 0.95, camera_hdr_txt, va="top", ha="left", fontsize=12, family="monospace")
    ax1_r.axis("off")
    ax1_r.set_title("[Camera Header]", loc="left", pad=6)

    add_section_title(fig, ax0_txt, "[0] General Info", dy=0.015)


    # ============================================================
    # [1] Original Data (1 row, 2 cols)
    # ============================================================
    gs1 = gs[1].subgridspec(1, 2, wspace=0.25)
    ax1_l = fig.add_subplot(gs1[0, 0])
    ax1_r = fig.add_subplot(gs1[0, 1])

    im1 = ax1_l.imshow(npy, cmap='gray', vmin=0, vmax=4095)
    fig.colorbar(im1, ax=ax1_l, fraction=0.046, pad=0.04)
    ax1_l.set_title('Numpy array from RAW file')

    ax1_r.hist(B_channel_flat,  bins=56, range=(0,4095), color='b',    alpha=0.5, label='B channel',  histtype='step')
    ax1_r.hist(G1_channel_flat, bins=56, range=(0,4095), color='g',    alpha=0.5, label='G1 channel', histtype='step')
    ax1_r.hist(G2_channel_flat, bins=56, range=(0,4095), color='lime', alpha=0.5, label='G2 channel', histtype='step')
    ax1_r.hist(R_channel_flat,  bins=56, range=(0,4095), color='r',    alpha=0.5, label='R channel',  histtype='step')
    ax1_r.set_yscale('log')
    ax1_r.set_xlabel('Pixel Value')
    ax1_r.set_ylabel('Number of Pixels (Log scale)')
    ax1_r.set_title('Histogram of Pixel Values for Each Channel')
    ax1_r.legend()
    ax1_r.grid(True)

    add_section_title(fig, ax1_l, "[1] Original Data", dy=0.015)

    # ============================================================
    # [2] RGGB Bayer Channel Splitting (FIXED spacing)
    # ============================================================

    gs2 = gs[2].subgridspec(2, 2, wspace=0.25, hspace=0.35)
    axes2 = [
        fig.add_subplot(gs2[0, 0]),
        fig.add_subplot(gs2[0, 1]),
        fig.add_subplot(gs2[1, 0]),
        fig.add_subplot(gs2[1, 1]),
    ]

    channel_lst  = [R_channel, G1_channel, G2_channel, B_channel]
    channel_name = ['R', 'G1', 'G2', 'B']
    cmap_lst    = ['Reds_r','Greens_r','Greens_r','Blues_r']
    color_lst    = ['red','green','green','blue']

    for i in range(4):
        ax = axes2[i]
        channel = channel_lst[i].astype(np.float32)
        ax.set_title(f"{channel_name[i]}_channel",color=color_lst[i], pad=8)
        ax.tick_params(axis='y', pad=1)
        im2 = ax.imshow(channel, cmap=cmap_lst[i])
        fig.colorbar(im2,ax=ax,fraction=0.046,pad=0.02 )
    add_section_title(fig, axes2[0], "[2] RGGB Bayer Channel Splitting", dy=0.018)


    # ============================================================
    # [3] Visualization Purpose Only
    # ============================================================
    gs3 = gs[3].subgridspec(1, 2, wspace=0.25, width_ratios=[1.25, 1.15])
    ax3_l = fig.add_subplot(gs3[0, 0])
    ax3_r = fig.add_subplot(gs3[0, 1])

    img = gray.astype(np.float32)
    img_disp = np.maximum(img, 1e-3)

    norm = colors.LogNorm(
        vmin=np.percentile(img_disp, 1),
        vmax=np.percentile(img_disp, 99)
    )

    im3 = ax3_l.imshow(img_disp, cmap='gray', norm=norm)
    fig.colorbar(im3, ax=ax3_l, fraction=0.046, pad=0.04, label='log-scale intensity')
    ax3_l.set_title(
        'Grayscale image with inter-channel correction\n'
        '(ONLY FOR VISUALIZATION PURPOSE)',
        pad=10
    )

    ax3_r.imshow(rgb)
    ax3_r.set_title(
        'RGB image with processing\n'
        '(ONLY FOR VISUALIZATION PURPOSE)',
        pad=10
    )
    #ax3_r.axis("off")

    add_section_title(fig, ax3_l, "[3] Visualization Purpose Only", dy=0.018)


    # ============================================================
    # [4] Saturation Check by Channels
    # ============================================================
    gs4 = gs[4].subgridspec(2, 2, wspace=0.25, hspace=0.35)
    axes4 = [
        fig.add_subplot(gs4[0, 0]),
        fig.add_subplot(gs4[0, 1]),
        fig.add_subplot(gs4[1, 0]),
        fig.add_subplot(gs4[1, 1]),
    ]

    channel_lst  = [R_channel, G1_channel, G2_channel, B_channel]
    channel_name = ['R', 'G1', 'G2', 'B']
    color_lst    = ['red','green','green','blue']

    cmap = colors.ListedColormap(['lightgray', 'red', 'blue'])
    norm_lbl = colors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)

    for i in range(4):
        ax = axes4[i]
        channel = channel_lst[i].astype(np.float32)

        dn_mask  = channel < 245
        sat_mask = channel > 4090

        label = np.zeros_like(channel, dtype=int)
        label[dn_mask]  = 1
        label[sat_mask] = 2

        total = 1312*992/4
        dn_ratio = np.sum(dn_mask)/total
        sat_ratio = np.sum(sat_mask)/total

        ax.set_title(f"{channel_name[i]}_channel\n(Dark: {np.round(dn_ratio*100, 2)}% ,  Saturate: {np.round(sat_ratio*100, 2)}%)", color=color_lst[i], pad=8)
        im4 = ax.imshow(label, cmap=cmap, norm=norm_lbl, interpolation='nearest')
        #ax.axis('off')

        cbar = fig.colorbar(im4, ax=ax, ticks=[0, 1, 2], shrink=0.9)
        cbar.ax.set_yticklabels(['Normal', 'Dark Signal', 'Saturation'],
                                rotation=90, va='center', fontsize=6)

    add_section_title(fig, axes4[0], "[4] Saturation Check by Channels", dy=0.018)


    fig.text(
        0.5, 0.995,
        "RAW Image Report",
        ha="center", va="top",
        fontsize=20, fontweight="bold"
    )

    fig.text(
        0.02, 0.995 ,
        f"software ver : \n(ICUCamera4Pole.py {icuc.get_version()})\n(ICRC_report.py {ver})",
        ha="left", va="top",
        fontsize=8
    )

    # ============================================================
    # Save ONE stitched image
    # ============================================================

    fig.savefig(output, dpi=300, pad_inches=0.2)
    plt.close('all')
    print(f"Saved: {output}")

with open('newDirs.txt', newline='') as csvfile:
    next(csvfile)
    data = list(csv.reader(csvfile))
data = [int(item[0]) for item in data]
for day in data:
    folder = Path("/home/jtorresespinosa/jtorresespinosa/UpgradeCamera/UpgradeCameraImageAnalysis/dailyReportRunner/output/" + str(day).zfill(4))
    folder.mkdir(parents=True, exist_ok=True)
    file_list = sorted(folder.glob("*.raw"))
    outputdir = Path("/data/ana/Calibration/upgrade-camera/reports/"+str(day).zfill(4))
    print("Making directory:", outputdir)
    outputdir.mkdir(parents=True, exist_ok=True, mode=0o750)
    for file in file_list:
        fname = str(file)
        make_report(fname, outputdir)