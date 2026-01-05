# src/server.py
import os
import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the EnterpriseArena Green Agent")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind")
    parser.add_argument("--card-url", type=str, help="URL for agent card")
    args = parser.parse_args()

    print("Starting the Green Agent")

    skill = AgentSkill(
        id="host_assess_enterprise_mcp",
        name="Enterprise MCP Assessment Hosting",
        description=(
            "Assess the tool-using ability of an A2A-compatible agent on enterprise MCP tasks. "
            "The caller should provide a <white_agent_url>...</white_agent_url> and an "
            "<env_config>...</env_config> JSON."
        ),
        tags=["green agent", "assessment hosting", "enterprise", "mcp"],
        examples=["""
Your task is to assess the agent located at:
<white_agent_url>http://localhost:9002</white_agent_url>
You should use the following env configuration:
<env_config>
{
  "tasks_file": "tasks.json",
  "task_indices": [0, 1, 2],
  "mcp_config_path": "/app/mcp_configs_http.json",
  "max_steps": 15
}
</env_config>
"""]
    )

    agent_card = AgentCard(
        name="EnterpriseArena Green Agent",
        description="Assessment hosting agent for MCP-based enterprise tasks.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    # Build and run the Starlette app
    starlette_app = app.build()
    
   # Debug: Show all registered routes
    print("\n" + "="*60)
    print("🔍 Registered Routes:")
    print("="*60)
    for route in starlette_app.routes:
        if hasattr(route, 'path'):
            methods = getattr(route, 'methods', ['*'])
            print(f"  {', '.join(methods):8s} {route.path}")
    print("="*60 + "\n")

    print(f"✅ Agent Card URL: {agent_card.url}")
    print(f"✅ Server starting on http://{args.host}:{args.port}")
    print(f"✅ Agent Card endpoint: http://{args.host}:{args.port}/.well-known/agent-card.json")
    print()
    
    uvicorn.run(starlette_app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
