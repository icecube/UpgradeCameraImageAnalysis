for f in /Users/seowonchoi/Documents/NAPPL/Operation/data/geometry/2026-03-01-20-36-57/*.raw; do
    echo "Processing $f"
    python ICUC_report.py --input "$f" --outputdir /Users/seowonchoi/Documents/NAPPL/UpgradeCameraImageAnalysis/data/mdom/
done
