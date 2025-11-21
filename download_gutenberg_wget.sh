#!/bin/bash
#
# Download Project Gutenberg books using wget and the official robot/harvest endpoint.
#
# This script uses Project Gutenberg's official bulk download endpoint, which is
# the recommended way to download multiple books. The robot/harvest endpoint is
# specifically designed for automated downloads and respects their terms of service.
#
# Usage:
#     ./download_gutenberg_wget.sh [output_dir]
#
# Arguments:
#     output_dir: Directory to save downloaded books (default: raw_data/gutenberg)
#
# Note: On macOS, install wget with: brew install wget
#

set -e  # Exit on error

# Default output directory
OUTPUT_DIR="${1:-raw_data/gutenberg}"

# Check if wget is available
if ! command -v wget &> /dev/null; then
    echo "Error: wget is not installed."
    echo ""
    echo "To install wget:"
    echo "  macOS:   brew install wget"
    echo "  Ubuntu:  sudo apt-get install wget"
    echo "  Fedora:  sudo dnf install wget"
    echo ""
    echo "Alternatively, you can use the Python downloader:"
    echo "  python download_gutenberg.py --k 5"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Project Gutenberg Bulk Download (wget)"
echo "============================================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "This script uses Project Gutenberg's official robot/harvest endpoint."
echo "This is the recommended method for bulk downloads."
echo ""
echo "Download settings:"
echo "  - Wait time: 2 seconds between requests"
echo "  - Mirror mode: Recursive download"
echo "  - File type: Plain text (.txt)"
echo "  - Language: English"
echo ""
echo "Starting download..."
echo "============================================================"

# Change to output directory for wget mirror
cd "$OUTPUT_DIR"

# Download using wget with Project Gutenberg's robot/harvest endpoint
# The harvest endpoint returns an HTML page with links to files.
# We need to follow those links recursively.
#
# -w 2: wait 2 seconds between requests (be polite to server)
# -r: recursive (follow links)
# -l 2: recursion depth (harvest page -> file links)
# -H: span hosts (allow downloading from different hosts if needed)
# -np: no parent (don't go up in directory structure)
# -nH: no host directories (don't create hostname directories)
# --cut-dirs=1: cut one directory level from the path
# --accept-regex: only keep files matching the pattern (pg*.txt files in cache/epub directories)
# --no-directories: don't create directory structure, save all .txt files in output dir
# --convert-links: convert links to local references
# Note: We allow HTML to be downloaded temporarily so wget can parse it for links,
# but we only keep files matching the accept-regex pattern
wget -w 2 -r -l 2 -H -np -nH --cut-dirs=1 \
     --accept-regex='.*/cache/epub/[0-9]+/pg[0-9]+\.txt$' \
     --no-directories \
     --convert-links \
     "https://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en" \
     2>&1 | tee wget_download.log

# Clean up: remove any HTML files or harvest page files that were downloaded
echo "Cleaning up temporary files..."
find . -maxdepth 1 -name "*.html" -type f -delete 2>/dev/null || true
find . -maxdepth 1 -name "harvest*" -type f -delete 2>/dev/null || true
find . -maxdepth 1 -name "*.tmp" -type f -delete 2>/dev/null || true

echo ""
echo "============================================================"
echo "Download complete!"
echo "============================================================"
echo "Files saved to: $OUTPUT_DIR"
echo "Download log saved to: $OUTPUT_DIR/wget_download.log"
echo ""
echo "Note: wget creates a mirror structure. You may need to find"
echo "the actual .txt files in subdirectories."
