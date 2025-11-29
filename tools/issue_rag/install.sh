#!/bin/bash
# This install script runs in the container after the bundle is copied.
# The csv0_memory_1k directory should be copied as part of the bundle on the host.
# If it's not present, the Python script will handle path resolution.
bundle_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Fix line endings for Python scripts (Windows CRLF -> Unix LF)
# This is critical when running on Windows with Docker Desktop
if command -v dos2unix >/dev/null 2>&1; then
    find "$bundle_dir/bin" -type f -name "*.py" -o -name "issue_rag" | while read -r file; do
        dos2unix "$file" 2>/dev/null || true
    done
else
    # Fallback: use sed to convert CRLF to LF
    find "$bundle_dir/bin" -type f \( -name "*.py" -o -name "issue_rag" \) -exec sed -i 's/\r$//' {} \;
fi

# Check if csv0_memory_1k exists in the bundle or mounted location
if [ ! -d "$bundle_dir/csv0_memory_1k" ]; then
    if [ -n "$ISSUE_RAG_MEMORY_DIR" ] && [ -d "$ISSUE_RAG_MEMORY_DIR" ]; then
        echo "Info: Using csv0_memory_1k from mounted directory: $ISSUE_RAG_MEMORY_DIR"
    else
        echo "Warning: csv0_memory_1k not found in bundle."
        echo "  The tool will try to locate it automatically."
        echo "  To mount from host, set ISSUE_RAG_MEMORY_DIR and use docker volume mount."
    fi
fi

