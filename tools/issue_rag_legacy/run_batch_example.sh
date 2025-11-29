#!/usr/bin/env bash
set -euo pipefail

# Example bash script to run SWE-agent batch mode with the issue_rag tool on Ubuntu.
# Update HOST_MEMORY_DIR to point to the csv0_memory_1k directory on the host.

HOST_MEMORY_DIR=${HOST_MEMORY_DIR:-/home/you/path/csv0_memory_1k}
MOUNT_POINT=${MOUNT_POINT:-/mnt/csv0_memory_1k}

sweagent run-batch \
  --config config/gpt5nano_issuerag.yaml \
  --agent.model.per_instance_cost_limit 0 \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split test \
  --instances.deployment.docker_args="[\"-v\", \"${HOST_MEMORY_DIR}:${MOUNT_POINT}\"]" \
  --instances.slice :3 \
  --instances.shuffle=True

