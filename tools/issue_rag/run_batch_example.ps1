# Example PowerShell script to run SWE-agent batch mode with issue_rag tool
# This demonstrates the correct format for Windows + Docker Desktop

# Your actual path (adjust if needed)
$memoryPath = "C:/Users/Admin/Documents/GitHub/Issue-Mem-SWE-agent/csv0_memory_1k"
$mountPoint = "/mnt/csv0_memory_1k"

# Build the docker_args JSON string
# IMPORTANT: Use single quotes for the JSON string to avoid PowerShell parsing issues
# Escape internal quotes with backticks when using double quotes, or use single quotes for the whole string
$dockerArgs = '["-v", "' + "$memoryPath" + ':' + "$mountPoint" + '"]'

# Run SWE-agent batch mode with volume mount
sweagent run-batch `
  --config config/gpt5nano_issuerag.yaml `
  --agent.model.per_instance_cost_limit 0 `
  --instances.type swe_bench `
  --instances.subset lite `
  --instances.split test `
  --instances.deployment.docker_args=$dockerArgs `
  --instances.slice :3 `
  --instances.shuffle=True

# Note: Make sure Docker Desktop has access to C: drive
# Check: Docker Desktop Settings > Resources > File Sharing

