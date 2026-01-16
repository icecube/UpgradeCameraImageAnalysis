#/!/bin/bash
for f in /Users/seowonchoi/Documents/NAPPL/Operation/data/0115/raw/*; do
    python ICUC_report.py --input "$f" --outputdir /Users/seowonchoi/Documents/NAPPL/Operation/data/0115/reports/
done
--