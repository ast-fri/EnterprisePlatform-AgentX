 EnterprisePlatform-AgentX 🟢

Enterprise evaluation agent (green agent) for benchmarking AI agents on real-world enterprise tool usage tasks using the Model Context Protocol (MCP).

## Overview

EnterprisePlatform-AgentX is a green agent that evaluates purple agents (participants) on their ability to:
- Select appropriate enterprise tools for tasks
- Execute tool calls with correct parameters
- Handle multi-step workflows
- Provide accurate final answers

The agent uses MCP to connect to enterprise services like RocketChat, Plane, and OwnCloud, creating a realistic evaluation environment.

## Architecture

┌─────────────────┐
│ Green Agent │ ◄── Orchestrates evaluation
│ (Evaluator) │
└────────┬────────┘
│
├─► Purple Agent (Participant being evaluated)
│
├─► MCP Servers (RocketChat, Plane, OwnCloud)
│
└─► Judge (Scores tool use & answer quality)

text

## Features

- **Auto-discovery**: Automatically finds purple agent on Docker network
- **Multi-MCP Support**: Connects to multiple MCP servers (69+ tools)
- **Comprehensive Judging**: Evaluates both tool usage and answer quality
- **A2A Protocol**: Standard Agent-to-Agent communication
- **Detailed Metrics**: Per-task and aggregate performance scores
- **Artifact Generation**: Saves structured evaluation results

## Prerequisites

- Docker & Docker Compose
- Python 3.13+
- MCP servers running (RocketChat, Plane, OwnCloud)
- Azure OpenAI API access (for judging)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/ast-fri/EnterprisePlatform-AgentX.git
cd EnterprisePlatform-AgentX
```

## Build Docker Image
```bash
docker build -t ghcr.io/ast-fri/enterpriseplatform-agentx:latest .
```

## Message Format
### The green agent accepts messages in this format:

```xml
<white_agent_url>http://purple-agent:9009</white_agent_url>
<env_config>
{
  "tasks_file": "tasks.json",
  "mcp_config_path": "mcp_configs_http.json",
  "task_indices":,[1]
  "max_steps": 15
}
</env_config>
```

Project Structure
text
EnterprisePlatform-AgentX/
├── src/
│   ├── agent.py           # Main evaluation logic
│   ├── env.py            # Environment setup
│   ├── judge.py          # Scoring logic
│   ├── messenger.py      # A2A communication
│   ├── mcp_tools.py      # MCP integration
│   ├── my_util.py        # Utilities
│   └── server.py         # A2A server
├── tasks.json            # Evaluation tasks
├── mcp_configs_http.json # MCP server config
├── Dockerfile
├── pyproject.toml
└── README.md
Local Development
bash
# Install dependencies
uv sync

# Run locally
uv run src/server.py --host 0.0.0.0 --port 9009

# Run tests
uv run pytest

License
MIT License - See LICENSE file for details

Citation
If you use EnterprisePlatform-AgentX in your research, please cite:

text
@software{enterpriseplatform_agentx,
  title = {EnterprisePlatform-AgentX: Enterprise AI Agent Evaluation Framework},
  author = {Fujitsu Research India},
  year = {2026},
  url = {https://github.com/ast-fri/EnterprisePlatform-AgentX}
}