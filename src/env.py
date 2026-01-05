import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from mcp_tools import load_mcp_tools_with_sessions


def load_env_config(env_config_str: str) -> Dict[str, Any]:
    """
    env_config comes from <env_config>...</env_config>.
    Example:
      {
        "tasks_file": "...",
        "task_indices": [0, 1, 2],   # optional, default: all tasks
        "mcp_config_path": "...json",
        "max_steps": 10
      }
    """
    return json.loads(env_config_str)


def load_all_tasks(tasks_file: str) -> List[Dict[str, Any]]:
    tasks_path = Path(tasks_file)
    with tasks_path.open("r") as f:
        return json.load(f)  # list of { "query": ... }


async def prepare_env_for_single_task(
    env_config: Dict[str, Any],
    task_index: int,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], Any]:
    """
    Prepare environment for a single task index:
      - Load that task's query.
      - Load MCP tools/sessions.
    Returns:
      task_query,
      tools_schema,
      env_runtime (contains tools/sessions),
      stack (AsyncExitStack to close after the task).
    """
    tasks_file = env_config["tasks_file"]
    mcp_config_path = env_config.get(
        "mcp_config_path",
        "/mnt/home-ldap/suraj_ldap/projects/MCP/chat/mcp_configs_http.json",
    )

    tasks = load_all_tasks(tasks_file)
    task = tasks[task_index]
    task_query = task["query"]

    tools, sessions, stack, tools_schema = await load_mcp_tools_with_sessions(
        mcp_config_path
    )

    env_runtime = {
        "tools": tools,
        "sessions": sessions,
    }

    return task_query, tools_schema, env_runtime, stack


def build_white_agent_task_prompt(
    task_query: str,
    tools_schema: List[Dict[str, Any]],
    observation: str | None = None,
) -> str:
    tools_schema_json = json.dumps({"tools": tools_schema}, indent=2)
    observation_text = observation or "No previous tool calls have been made yet."

    prompt = f"""
You are an enterprise assistant evaluated on your ability to select and use tools correctly.

You have access to the following tools, described as JSON:

<tools_schema_json>
{tools_schema_json}
</tools_schema_json>

The user's task is:

<task>
{task_query}
</task>

The latest observation (e.g., previous tool result or user message) is:

<observation>
{observation_text}
</observation>

At each step, respond with exactly ONE JSON object wrapped in <action>...</action>.

If you want to call a tool:

<action>
{{
  "type": "tool_call",
  "tool": "<tool_name>",
  "args": {{ ... }}
}}
</action>

If you want to stop and give your final answer:

<action>
{{
  "type": "final_answer",
  "content": "your final answer here"
}}
</action>

Do not output anything outside the <action>...</action> block.
"""
    return prompt.strip()
