# Example PowerShell script to run SWE-agent with issue_rag tool
# This demonstrates the correct format for Windows + Docker Desktop

# Your actual path (adjust if needed)
$memoryPath = "C:/Users/Admin/Documents/GitHub/Issue-Mem-SWE-agent/csv0_memory_1k"
$mountPoint = "/mnt/csv0_memory_1k"

# Build the docker_args JSON string
$dockerArgs = "[`"-v`", `"$memoryPath`:$mountPoint`"]"

# Run SWE-agent with volume mount
# Adjust other parameters as needed
sweagent run `
  --config config/gpt5nano_issuerag.yaml `
  --env.deployment.docker_args="$dockerArgs" `
  --agent.model.name=openai/gpt-5.1 `
  --env.repo.github_url=YOUR_REPO_URL `
  --problem_statement.text="YOUR_PROBLEM_STATEMENT"

# Note: Make sure Docker Desktop has access to C: drive
# Check: Docker Desktop Settings > Resources > File Sharing

