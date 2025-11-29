#!/usr/bin/env bash
set -euo pipefail

# Example bash script to run SWE-agent with the issue_rag tool on Ubuntu.
# Update HOST_MEMORY_DIR to point to the csv0_memory_1k directory on the host.

HOST_MEMORY_DIR=${HOST_MEMORY_DIR:-/home/you/path/csv0_memory_1k}
MOUNT_POINT=${MOUNT_POINT:-/mnt/csv0_memory_1k}

sweagent run \
  --config config/gpt5nano_issuerag.yaml \
  --env.deployment.docker_args="[\"-v\", \"${HOST_MEMORY_DIR}:${MOUNT_POINT}\"]" \
  --agent.model.name=openai/gpt-5.1 \
  --env.repo.github_url=YOUR_REPO_URL \
  --problem_statement.text="YOUR_PROBLEM_STATEMENT"

