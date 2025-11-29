# Issue RAG Tool

This tool automatically retrieves relevant agent memories from `csv0_memory_1k` based on semantic similarity to the issue description.

## Setup

### Option 1: Mount Host Directory (Recommended for Large Files)

To avoid copying large memory files into the container, mount the host directory:

1. Set the environment variable in your config (e.g., `config/gpt5nano.yaml`):
```yaml
tools:
  env_variables:
    ISSUE_RAG_MEMORY_DIR: /mnt/csv0_memory_1k
```

2. Run SWE-agent with docker volume mount:
```bash
sweagent run \
  --config config/gpt5nano.yaml \
  --env.deployment.docker_args='["-v", "C:/Users/Admin/Documents/GitHub/Issue-Mem-SWE-agent/csv0_memory_1k:/mnt/csv0_memory_1k"]' \
  ...
```

**Note**: Adjust the Windows path format for your system. Use forward slashes or escaped backslashes in the docker_args.

### Option 2: Copy to Bundle (For Small Files)

If the memory files are small, you can copy them to the bundle directory:
```bash
# On Windows (PowerShell)
Copy-Item -Path csv0_memory_1k -Destination tools\issue_rag\csv0_memory_1k -Recurse
```

The tool will automatically find `csv0_memory_1k` in the bundle directory.

## Usage

Simply call:
```bash
issue_rag "Your issue description here"
```

The tool will:
1. Automatically find the most similar memories using semantic search
2. Return the top 3 matches (by default)
3. Display episodic, semantic, and procedural memories for context

## Dependencies

Requires `sentence-transformers` package. Install with:
```bash
pip install sentence-transformers
```

