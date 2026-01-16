# UpgradeCameraImageAnalysis
Tools for analysis of RAW data coming from the IceCube Upgrade cameras deployed during the 25-26 season.

## Version Info

| File | Version | Last Updated |
|------|---------|--------------|
| `ICUCamera.py` | v1.0.0 | 2026-01-15 |
| `ICUC_report.py` | v1.0.3 | 2026-01-15 |

-------------
This is different from [ICUCamera package for people at SouthPole](https://github.com/WIPACrepo/hole-freeze-operations/tree/main/UpgradeCamera/ImageVerfication/ICUCamera4Pole). 

Check the dependency list below, and if you have any restrictions to use modules other than `os`, `numpy`, `matplotlib`, please check the link above.

-------------
## Dependencies

This project requires the following Python packages:

### Required

- Python >= 3.8
- numpy
- matplotlib
- opencv-python
- astropy

### Standard Python libraries used (no installation needed):

- os
- sys
- glob
- gzip
- struct
- argparse
- tarfile
- pathlib

-------------

## How to generate the image report.py

```python ICUC_report --input [full/path/of/your/rawfile.raw] --outputdir [directory/path/to/store/pdfs/]```

If you have multiple `.raw` files in a directory, and want to generate report files for all of them, checkout `pdf_generate_loop.sh` script.

## ICUCamera.py

This is a script which includes functions processing the images from IceCube Upgrade Camera System (a.k.a. Fixed Focus Camera, SKKU Camera, South Korea Camera, etc.,)

As we develope better processing methods, this script will be updated.

Specific use of the modules are explaind in `example.ipynb`

## example.ipynb

Here you can find out how to use each function in `ICUCamera.py`.

Whenever you add new functions in `ICUCamera.py`, it is strongly recommended to add a blocks in this notebook

## example_data

This is a folder with example `.raw` file and intermediate image files generated in the `example.ipynb`. 

Let's keep this directory as light as possible.

If you'd like to use data other than that, please use other directory, or make `/data/` directory inside this repo.

since `/data/` is a part of the `.gitignore`, it won't be uploaded in the github.

## tar_unzip.py / tar_unzip.ipynb

As real data is transfered via JADE, it is sent out as a `.tar.gz` file which contains a `.raw` file and a `.sem`file.

These scripts explains how to unzip and extract the `.raw` file only.

