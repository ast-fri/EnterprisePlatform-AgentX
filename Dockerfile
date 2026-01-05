FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install system dependencies needed for MCP and enterprise tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

# Copy Python project files
COPY --chown=agent:agent pyproject.toml uv.lock README.md ./
COPY --chown=agent:agent src src

# Copy EnterpriseArena specific files
COPY --chown=agent:agent tasks.json src/tasks.json
COPY --chown=agent:agent mcp_configs_http.json src/mcp_configs_http.json

# Create results directory
RUN mkdir -p /home/agent/results

# Install dependencies with cache
RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

# Environment variables
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
EXPOSE 9009
