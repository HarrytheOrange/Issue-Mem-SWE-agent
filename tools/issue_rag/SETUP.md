# Issue RAG Tool Setup Guide

## Quick Setup for Windows + Docker Desktop

### Step 1: Configuration is Already Set

The `config/gpt5nano_issuerag.yaml` already has `ISSUE_RAG_MEMORY_DIR: /mnt/csv0_memory_1k` configured.

### Step 2: Run with Docker Volume Mount

**IMPORTANT**: The docker volume mount configuration depends on which command you use:

#### For `sweagent run` (single instance):

When running SWE-agent, add the docker volume mount argument. **For Windows + Docker Desktop**, use one of these formats:

**Option A: Using forward slashes (recommended for PowerShell):**
```powershell
sweagent run `
  --config config/gpt5nano_issuerag.yaml `
  --env.deployment.docker_args='["-v", "C:/Users/Admin/Documents/GitHub/Issue-Mem-SWE-agent/csv0_memory_1k:/mnt/csv0_memory_1k"]' `
  ...
```

**Option B: Using escaped backslashes:**
```powershell
sweagent run `
  --config config/gpt5nano_issuerag.yaml `
  --env.deployment.docker_args='["-v", "C:\\Users\\Admin\\Documents\\GitHub\\Issue-Mem-SWE-agent\\csv0_memory_1k:/mnt/csv0_memory_1k"]' `
  ...
```

**Option C: Using WSL path format (if Docker Desktop is configured for WSL):**
```powershell
sweagent run `
  --config config/gpt5nano_issuerag.yaml `
  --env.deployment.docker_args='["-v", "/mnt/c/Users/Admin/Documents/GitHub/Issue-Mem-SWE-agent/csv0_memory_1k:/mnt/csv0_memory_1k"]' `
  ...
```

#### For `sweagent run-batch` (batch mode):

**CRITICAL**: Use `--instances.deployment.docker_args` instead of `--env.deployment.docker_args`:

**Option 1: Using PowerShell variable (Recommended):**
```powershell
$dockerArgs = '["-v", "C:/Users/Admin/Documents/GitHub/Issue-Mem-SWE-agent/csv0_memory_1k:/mnt/csv0_memory_1k"]'
sweagent run-batch `
  --config config/gpt5nano_issuerag.yaml `
  --instances.type swe_bench `
  --instances.subset lite `
  --instances.split test `
  --instances.deployment.docker_args=$dockerArgs `
  --agent.model.name=openai/gpt-5.1 `
  ...
```

**Option 2: Using escaped quotes:**
```powershell
sweagent run-batch `
  --config config/gpt5nano_issuerag.yaml `
  --instances.type swe_bench `
  --instances.subset lite `
  --instances.split test `
  --instances.deployment.docker_args="[`"-v`", `"C:/Users/Admin/Documents/GitHub/Issue-Mem-SWE-agent/csv0_memory_1k:/mnt/csv0_memory_1k`"]" `
  --agent.model.name=openai/gpt-5.1 `
  ...
```

**Important Notes for Windows + Docker Desktop**: 
- Replace the path with your actual `csv0_memory_1k` directory path
- Use forward slashes `/` (Option A) - this is the most reliable on Windows
- The path after `:` is the mount point inside the container (`/mnt/csv0_memory_1k`)
- This mount point must match `ISSUE_RAG_MEMORY_DIR` in the config
- Ensure Docker Desktop has access to the drive (check Docker Desktop Settings > Resources > File Sharing)
- **For batch mode**: Use `--instances.deployment.docker_args` (not `--env.deployment.docker_args`)

### Alternative: Copy Directory to Bundle

If you prefer not to use volume mounts, copy the directory:

```powershell
Copy-Item -Path csv0_memory_1k -Destination tools\issue_rag\csv0_memory_1k -Recurse
```

Then the tool will automatically find it in the bundle directory.

## Verification

After setup, the tool should be able to find the memory directory. If you see an error, check:
1. The docker volume mount path is correct
2. The `ISSUE_RAG_MEMORY_DIR` environment variable matches the mount point
3. The directory exists on your host system

