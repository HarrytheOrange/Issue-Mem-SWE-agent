# Issue RAG Tool Setup Guide

## Quick Setup for Ubuntu + Docker

### Step 1: Ensure Configuration Matches the Container Mount

The `config/gpt5nano_issuerag.yaml` file already sets `ISSUE_RAG_MEMORY_DIR: /mnt/csv0_memory_1k`. Keep this value in sync with the mount point that you pass to Docker.

### Step 2: Run with a Docker Volume Mount

When starting SWE-agent, mount your local `csv0_memory_1k` directory into the container and reuse the same mount point as the environment variable:

```bash
sweagent run \
  --config config/gpt5nano_issuerag.yaml \
  --env.deployment.docker_args='["-v", "/home/you/path/csv0_memory_1k:/mnt/csv0_memory_1k"]' \
  --agent.model.name=openai/gpt-5.1 \
  ...
```

For batch runs, swap `--env.deployment.docker_args` with `--instances.deployment.docker_args`.

### Alternative: Copy Directory to the Bundle

If you prefer not to mount, copy the directory into the bundle before running:

```bash
cp -r /home/you/path/csv0_memory_1k tools/issue_rag/csv0_memory_1k
```

The tool will automatically detect this bundled directory.

## Verification

After setup, the tool should locate the memory directory. If you see errors, double-check:
1. The host path passed to `docker_args` exists and is readable
2. `ISSUE_RAG_MEMORY_DIR` matches the mount point inside the container
3. `csv0_memory_1k` exists either in the bundle directory or at the mounted path

