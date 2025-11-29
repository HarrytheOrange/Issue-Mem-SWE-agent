# Issue RAG Tool - Batch Mode Setup

## Quick Fix for Your Current Issue

You're using `sweagent run-batch`, so you need to configure `instances.deployment.docker_args` instead of `env.deployment.docker_args`.

## Command to Run

**Option 1: Using PowerShell variable (Recommended):**

```powershell
$dockerArgs = '["-v", "C:/Users/Admin/Documents/GitHub/Issue-Mem-SWE-agent/csv0_memory_1k:/mnt/csv0_memory_1k"]'
sweagent run-batch `
  --config config/gpt5nano_issuerag.yaml `
  --agent.model.per_instance_cost_limit 0 `
  --instances.type swe_bench `
  --instances.subset lite `
  --instances.split test `
  --instances.deployment.docker_args=$dockerArgs `
  --instances.slice :3 `
  --instances.shuffle=True
```

**Option 2: Using escaped quotes:**

```powershell
sweagent run-batch `
  --config config/gpt5nano_issuerag.yaml `
  --agent.model.per_instance_cost_limit 0 `
  --instances.type swe_bench `
  --instances.subset lite `
  --instances.split test `
  --instances.deployment.docker_args="[`"-v`", `"C:/Users/Admin/Documents/GitHub/Issue-Mem-SWE-agent/csv0_memory_1k:/mnt/csv0_memory_1k`"]" `
  --instances.slice :3 `
  --instances.shuffle=True
```

**Option 3: Using cmd.exe (if PowerShell is problematic):**

```cmd
sweagent run-batch --config config/gpt5nano_issuerag.yaml --agent.model.per_instance_cost_limit 0 --instances.type swe_bench --instances.subset lite --instances.split test --instances.deployment.docker_args="[\"-v\", \"C:/Users/Admin/Documents/GitHub/Issue-Mem-SWE-agent/csv0_memory_1k:/mnt/csv0_memory_1k\"]" --instances.slice :3 --instances.shuffle=True
```

## Key Points

1. **Use `--instances.deployment.docker_args`** (not `--env.deployment.docker_args`)
2. **Path format**: Use forward slashes `/` for Windows paths
3. **Mount point**: `/mnt/csv0_memory_1k` must match `ISSUE_RAG_MEMORY_DIR` in config
4. **Docker Desktop**: Ensure C: drive is shared (Settings > Resources > File Sharing)

## Verify It Works

After running, check the trace log. You should see the tool successfully finding the memory directory instead of the "Memory directory not found" error.

