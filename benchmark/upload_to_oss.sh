#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# OSS configuration
OSS_BUCKET="oss://morphling-prod/482607"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to show usage
usage() {
    echo "Usage: $0 <file_or_directory> [remote_path]"
    echo ""
    echo "Arguments:"
    echo "  file_or_directory  Path to file or directory to upload"
    echo "  remote_path        Optional remote path in OSS (default: basename of source)"
    echo ""
    echo "Examples:"
    echo "  $0 result/benchmarks                    # Upload directory as result_benchmarks.tar.gz"
    echo "  $0 config.yaml                          # Upload file as config.yaml"
    echo "  $0 result/benchmarks my_results.tar.gz  # Upload with custom name"
    exit 1
}

# Check arguments
if [ $# -lt 1 ]; then
    print_error "Missing required argument"
    usage
fi

SOURCE_PATH="$1"
REMOTE_PATH="${2:-}"

# Check if source exists
if [ ! -e "$SOURCE_PATH" ]; then
    print_error "Source path does not exist: $SOURCE_PATH"
    exit 1
fi

# Check if ossutil is available
if ! command -v ossutil &> /dev/null; then
    print_error "ossutil command not found. Please install ossutil first."
    exit 1
fi

# Determine if we need to compress
CLEANUP_FILE=""
UPLOAD_FILE=""

if [ -d "$SOURCE_PATH" ]; then
    # It's a directory - compress it
    print_info "Compressing directory: $SOURCE_PATH"

    # Generate archive name
    DIR_BASENAME=$(basename "$SOURCE_PATH")
    ARCHIVE_NAME="${DIR_BASENAME}.tar.gz"

    # If remote path is specified, use it; otherwise use generated name
    if [ -n "$REMOTE_PATH" ]; then
        ARCHIVE_NAME="$REMOTE_PATH"
    fi

    # Create temporary archive
    TEMP_ARCHIVE="/tmp/${ARCHIVE_NAME}"

    # Compress (exclude hidden files and common large directories)
    tar -czf "$TEMP_ARCHIVE" \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='.venv' \
        --exclude='node_modules' \
        -C "$(dirname "$SOURCE_PATH")" \
        "$(basename "$SOURCE_PATH")"

    UPLOAD_FILE="$TEMP_ARCHIVE"
    CLEANUP_FILE="$TEMP_ARCHIVE"

    # Show archive size
    ARCHIVE_SIZE=$(du -h "$TEMP_ARCHIVE" | cut -f1)
    print_info "Archive created: $TEMP_ARCHIVE (${ARCHIVE_SIZE})"

elif [ -f "$SOURCE_PATH" ]; then
    # It's a file - upload directly
    print_info "Uploading file: $SOURCE_PATH"
    UPLOAD_FILE="$SOURCE_PATH"

    # Determine remote filename
    if [ -n "$REMOTE_PATH" ]; then
        ARCHIVE_NAME="$REMOTE_PATH"
    else
        ARCHIVE_NAME=$(basename "$SOURCE_PATH")
    fi
else
    print_error "Source is neither a file nor a directory: $SOURCE_PATH"
    exit 1
fi

# Construct OSS destination path
OSS_DEST="${OSS_BUCKET}/${ARCHIVE_NAME}"

# Upload to OSS
print_info "Uploading to: $OSS_DEST"
if ossutil cp "$UPLOAD_FILE" "$OSS_DEST"; then
    print_info "Upload successful!"
    print_info "OSS URL: $OSS_DEST"
else
    print_error "Upload failed"
    # Cleanup on failure
    if [ -n "$CLEANUP_FILE" ] && [ -f "$CLEANUP_FILE" ]; then
        rm -f "$CLEANUP_FILE"
        print_info "Cleaned up temporary file: $CLEANUP_FILE"
    fi
    exit 1
fi

# Cleanup temporary archive if created
if [ -n "$CLEANUP_FILE" ] && [ -f "$CLEANUP_FILE" ]; then
    rm -f "$CLEANUP_FILE"
    print_info "Cleaned up temporary file: $CLEANUP_FILE"
fi

print_info "Done!"
