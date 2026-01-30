import traceback
import ICUCamera as icuc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import argparse
import cv2
import csv
from pathlib import Path
import matplotlib as mpl

mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.rm'] = 'Verdana'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

mpl.rc('font', family='sans-serif', size=14)
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8

mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['axes.linewidth'] = 1.2 # set the value globally

# mpl.rc('font', size=16)
mpl.rc('axes', titlesize=22)

rawFolder = Path("/data/user/jtorresespinosa/UpgradeCamera/UpgradeCameraImageAnalysis/dailyReportRunner/output/0126")
string = "Camera-Run_IIIB_string88_DEgg_port5151_cam1_"
for filename in rawFolder.iterdir():
    if filename.name.startswith(string):
        print(filename.name)
        shape, npy = icuc.Raw2Npy(str(filename))
        B_channel_flat  = npy[1::2, 1::2].astype(np.float32).flatten()
        G1_channel_flat = npy[0::2, 1::2].astype(np.float32).flatten()
        G2_channel_flat = npy[1::2, 0::2].astype(np.float32).flatten()
        R_channel_flat  = npy[0::2, 0::2].astype(np.float32).flatten()

        R_channel  = npy[0::2, 0::2]
        G1_channel = npy[0::2, 1::2]
        G2_channel = npy[1::2, 0::2]
        B_channel  = npy[1::2, 1::2]

        plt.figure(figsize=(10, 7))
        plt.hist(B_channel_flat,  bins=56, range=(0,4095), color='b',    alpha=0.5, label='B channel',  histtype='step', lw = 2)
        plt.hist(G1_channel_flat, bins=56, range=(0,4095), color='g',    alpha=0.5, label='G1 channel', histtype='step', lw = 2)
        plt.hist(G2_channel_flat, bins=56, range=(0,4095), color='lime', alpha=0.5, label='G2 channel', histtype='step', lw = 2)
        plt.hist(R_channel_flat,  bins=56, range=(0,4095), color='r',    alpha=0.5, label='R channel',  histtype='step', lw = 2)
        plt.yscale('log')
        plt.xlabel('Pixel Value')
        plt.ylabel('Number of Pixels (Log scale)')
        plt.title('Histogram of Pixel Values for Each Channel')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./plots/{filename.stem}_pixel_histogram.png")

