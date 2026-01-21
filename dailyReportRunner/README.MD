
# Daily report runner 

This folder contains tools to identify if a new upgrade-camera run has arrived from the South Pole and produce reports for each of the captured images.

## Authors

- [@toej93](https://www.github.com/toej93)

Using code developed by Seowon Choi
## Documentation

[Documentation](https://linktodocumentation)

The main code is `runRoutine.sh`, which will do the following (in order):

1. Compare against a list of older runs to see if there's a new run that appeared [`checkNewDirs.sh`]
2. Untar/unzip transferred files for newer run(s) [`tar_unzip.py`]
3. Produce report for each capture from the newer run(s) [`ICUC_report.py`]

The code is smart enought that no inputs are required, but it will identify the newly created files and run the code over all of them. The reports will be stored in `/data/ana/Calibration/upgrade-camera/reports/`, where the subfolders correspond to `MM/DD`, the date when the run happened.