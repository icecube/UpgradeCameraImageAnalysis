#!/bin/bash
#Check if there are new directories and run report if there are new directories
./checkNewDirs.sh
return_code=$?

if [ $return_code -eq 1 ]; then
    echo "New directories detected. Running the routine..."

    echo "Untaring files..."

    # python3 ./tar_unzip.py
    if python3 tar_unzip.py; then
        echo "Untarred. Making reports now..."
    else
        echo "Untaring Script failed. Exiting."
    fi

    if python3 ICUC_report.py; then
        echo "Reports made successfully."
    else
        echo "Report Script failed. Exiting."
    fi
    
# rm -r ./output/*

else
    echo "No new directories detected. Routine will not run."
fi


