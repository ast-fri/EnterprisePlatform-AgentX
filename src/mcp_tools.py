# src/green_agent/mcp_tools.py

import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict, Any, List, Tuple

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool


def extract_tool_info(tool: BaseTool) -> Dict[str, Any]:
    """Serialize a LangChain tool into a schema that any white agent can consume."""
    # Case 1: args_schema is a Pydantic BaseModel
    if hasattr(tool.args_schema, "model_json_schema"):
        schema = tool.args_schema.model_json_schema()
        args_required = schema.get("required", [])
        properties = schema.get("properties", {})
    else:
        # assume dict format
        schema = tool.args_schema
        args_required = schema.get("required", [])
        properties = schema.get("properties", {})

    args = {}
    for field, meta in properties.items():
        args[field] = {
            "type": meta.get("type", "unknown"),
            "description": meta.get("description", ""),
            "required": field in args_required,
        }

    return {
        "name": tool.name,
        "description": tool.description or "",
        "arguments": args,
    }


async def load_mcp_tools_with_sessions(
    mcp_config_path: str,
) -> Tuple[List[BaseTool], Dict[str, Any], AsyncExitStack, List[Dict[str, Any]]]:
    """
    - Load MCP config JSON like your original client.
    - Connect to all servers, keep sessions via AsyncExitStack.
    - Return:
        tools: LangChain tools for potential green-side execution.
        sessions: {server_name: session}
        stack: the AsyncExitStack, to be closed by caller.
        tools_schema: list[dict] for sending to white agent.
    """
    cfg_path = Path(mcp_config_path)
    with cfg_path.open("r") as f:
        config = json.load(f)  # expects { "mcpServers": { ... } }

    client = MultiServerMCPClient(config["mcpServers"])
    tools: List[BaseTool] = []

    stack = AsyncExitStack()
    await stack.__aenter__()

    sessions: Dict[str, Any] = {}

    for server_name in config["mcpServers"].keys():
        try:
            session = await stack.enter_async_context(client.session(server_name))
            sessions[server_name] = session
            server_tools = await load_mcp_tools(session)
            tools.extend(server_tools)
            print(f"✓ Connected to {server_name} - Loaded {len(server_tools)} tools")
            print("  Tools:", ", ".join(tool.name for tool in server_tools))
        except Exception as e:
            print(f"✗ Failed to connect to {server_name}: {e}")

    if not tools:
        raise RuntimeError("No tools loaded from provided MCP servers")

    tools_schema = [extract_tool_info(t) for t in tools]

    return tools, sessions, stack, tools_schema
