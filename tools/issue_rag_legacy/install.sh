#!/usr/bin/env bash
set -euo pipefail

# This install script runs inside the Linux container once the bundle is copied.
# The csv0_memory_1k directory can either be provided in the bundle or mounted
# into the container and referenced via ISSUE_RAG_MEMORY_DIR.

bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)

script_is_sourced=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    script_is_sourced=1
fi

finish() {
    local status=$1
    if [[ $script_is_sourced -eq 1 ]]; then
        return "$status"
    fi
    exit "$status"
}

if [[ -d "$bundle_dir/csv0_memory_1k" ]]; then
    echo "issue_rag: found csv0_memory_1k in bundle."
    finish 0
fi

if [[ -n "${ISSUE_RAG_MEMORY_DIR:-}" && -d "${ISSUE_RAG_MEMORY_DIR}" ]]; then
    echo "issue_rag: using mounted csv0_memory_1k from ${ISSUE_RAG_MEMORY_DIR}."
    finish 0
fi

cat <<'EOF'
issue_rag warning: csv0_memory_1k not found in bundle or via ISSUE_RAG_MEMORY_DIR.
Provide the memories by either copying them into tools/issue_rag/csv0_memory_1k
or by mounting the directory into the container and exporting ISSUE_RAG_MEMORY_DIR.
EOF
