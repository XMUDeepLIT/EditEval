#!/bin/bash

# Download source videos from Google Drive
# Link: https://drive.google.com/file/d/1zy17YLjtnchUYyN4BOAciZRteDp3SSHJ/view?usp=sharing

FILE_ID="1zy17YLjtnchUYyN4BOAciZRteDp3SSHJ"
OUTPUT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_FILE="${OUTPUT_DIR}/source_videos.zip"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown not found, installing..."
    pip install gdown
fi

echo "Downloading source videos from Google Drive..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUTPUT_FILE}"

# Check if download was successful
if [ $? -eq 0 ] && [ -f "${OUTPUT_FILE}" ]; then
    echo "Download complete: ${OUTPUT_FILE}"
    echo "Extracting..."
    unzip -o -j "${OUTPUT_FILE}" -d "${OUTPUT_DIR}"
    echo "Extraction complete. Removing zip file..."
    rm -f "${OUTPUT_FILE}"
    echo "Done!"
else
    echo "Download failed. Please check the sharing link and permissions."
    exit 1
fi
