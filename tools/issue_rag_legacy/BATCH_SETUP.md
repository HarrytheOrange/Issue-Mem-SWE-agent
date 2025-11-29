# Issue RAG Tool - Batch Mode Setup

## Quick Fix for Your Current Issue

When running `sweagent run-batch`, pass the volume mount through `--instances.deployment.docker_args` so every container gets access to the same memory directory.

## Command to Run

```bash
HOST_MEMORY_DIR=/home/you/path/csv0_memory_1k
MOUNT_POINT=/mnt/csv0_memory_1k

sweagent run-batch \
  --config config/gpt5nano_issuerag.yaml \
  --agent.model.per_instance_cost_limit 0 \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split test \
  --instances.deployment.docker_args="[\"-v\", \"${HOST_MEMORY_DIR}:${MOUNT_POINT}\"]" \
  --instances.slice :3 \
  --instances.shuffle=True
```

## Key Points

1. Use `--instances.deployment.docker_args` for batch mode.
2. Keep `/mnt/csv0_memory_1k` in sync with `ISSUE_RAG_MEMORY_DIR` inside the config.
3. Make sure `${HOST_MEMORY_DIR}` exists on the host and contains the `*_memory.json` files.

## Verify It Works

Check the generated trace log; the tool should report that it located the memory directory instead of raising the "Memory directory not found" error.

