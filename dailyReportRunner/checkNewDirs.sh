#!/bin/bash

# Define the directory to monitor and the file to store the list
MONITOR_DIR="/data/exp/IceCube/2026/internal-system/upgrade-camera/"
LIST_FILE="./listOfCurrentDirs.txt"
NEW_LIST_FILE="./newDirs.txt"
DATE_STR=$(date +"%Y-%m-%d %T")

monitor_new_dirs() {
    # 1. Initial list of dirs
    if [[ ! -f ${LIST_FILE} ]]; then
        find "${MONITOR_DIR}" -maxdepth 1 -type d -printf "%f\\n" | sort > "${LIST_FILE}"
        echo "Initial directory list created: ${LIST_FILE}"
        exit 0
    fi

    CURRENT_DIRS=$(find "${MONITOR_DIR}" -maxdepth 1 -type d -printf "%f\\n" | sort)

    # 2. Use `comm` to find new directories not in text file
    NEW_DIRS=$(comm -13 "${LIST_FILE}" <(echo "${CURRENT_DIRS}"))

    # 3. Check if any new directories were found
    if [[ -n "${NEW_DIRS}" ]]; then
        echo "New directories found:"
        echo "${NEW_DIRS}"

        # Update the list file to reflect the new state
        echo "${CURRENT_DIRS}" > "${LIST_FILE}"
        echo "$DATE_STR" > "${NEW_LIST_FILE}"
        echo "${NEW_DIRS}" >> "${NEW_LIST_FILE}"
        echo "Directory list updated."

        # Perform actions on the new directories here, e.g., in a loop
        # for dir in "${NEW_DIRS}"; do
        #   echo "Processing new directory: $dir"
        #   # your commands here
        # done
        return 1

    else
        # echo "No new directories found."
        return 0
    fi
}

monitor_new_dirs