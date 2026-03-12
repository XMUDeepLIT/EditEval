#!/bin/bash

# Download MLLM results from Google Drive
# Link: https://drive.google.com/file/d/1KOjR4FG-coqLrK06ldPs7T8pmO_LVR3p/view?usp=sharing

FILE_ID="1KOjR4FG-coqLrK06ldPs7T8pmO_LVR3p"
OUTPUT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_FILE="${OUTPUT_DIR}/mllm_results.zip"
TEMP_DIR="${OUTPUT_DIR}/_tmp_extract"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown not found, installing..."
    pip install gdown
fi

echo "Downloading MLLM results from Google Drive..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUTPUT_FILE}"

# Check if download was successful
if [ $? -eq 0 ] && [ -f "${OUTPUT_FILE}" ]; then
    echo "Download complete: ${OUTPUT_FILE}"
    echo "Extracting to ${OUTPUT_DIR}..."

    # Extract to a temp directory first
    mkdir -p "${TEMP_DIR}"
    unzip -o "${OUTPUT_FILE}" -d "${TEMP_DIR}"

    # Move contents up: flatten one nested level if exists
    # e.g. _tmp_extract/mllm_results/* -> OUTPUT_DIR/
    for dir in "${TEMP_DIR}"/*/; do
        dir_name=$(basename "$dir")
        # Skip __MACOSX
        if [ "$dir_name" = "__MACOSX" ]; then
            continue
        fi
        # Move all contents inside the nested folder up to OUTPUT_DIR
        mv -f "$dir"* "${OUTPUT_DIR}/" 2>/dev/null
        mv -f "$dir".* "${OUTPUT_DIR}/" 2>/dev/null
    done

    # Cleanup
    rm -rf "${TEMP_DIR}"
    rm -f "${OUTPUT_FILE}"
    echo "Done!"
else
    echo "Download failed. Please check the sharing link and permissions."
    exit 1
fi
