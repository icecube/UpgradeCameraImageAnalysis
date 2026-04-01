import ICUCamera as icuc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os


inputdir = "/Users/seowonchoi/Documents/NAPPL/Operation/data/geometry/2026-03-01-20-36-57/"
outputdir = "/Users/seowonchoi/Documents/NAPPL/UpgradeCameraImageAnalysis/data/mdom/pngs/"
# os.mkdir(outputdir)

for raw in os.listdir(inputdir):
    if raw.endswith(".raw"):
        path = os.path.join(inputdir, raw)
        shape, npy = icuc.Raw2Npy(path)
        gray = icuc.get_gray(npy)
        img = gray.astype(np.float32)
        img_disp = np.maximum(img, 1e-3)

        norm = colors.LogNorm(
            vmin=np.percentile(img_disp, 1),
            vmax=np.percentile(img_disp, 99)
        )
        plt.imshow(img_disp, cmap='gray', norm=norm)
        plt.colorbar(fraction=0.046, pad=0.04, label='log-scale intensity')
        outpath = os.path.join(outputdir, f"{raw[:-4]}.png")
        plt.savefig(outpath, dpi=300)
        plt.close()