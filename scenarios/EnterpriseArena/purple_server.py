# scenarios/enterprise-mcp/purple_server.py
import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from purple_agent import BaselinePurpleExecutor


def main():
    parser = argparse.ArgumentParser(description="Baseline Enterprise Purple Agent")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9002, help="Port to bind")
    parser.add_argument("--card-url", type=str, help="URL for agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="enterprise_tool_usage",
        name="Enterprise Tool Usage",
        description="Solve enterprise tasks using MCP tools through multi-step reasoning",
        tags=["tools", "reasoning", "enterprise", "mcp"],
        examples=["Query a database to find customer information", 
                  "Access web APIs to fetch real-time data"]
    )

    agent_card = AgentCard(
        name="Baseline Enterprise Agent",
        description="Simple ReAct-style agent for enterprise MCP tool usage tasks",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=BaselinePurpleExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
