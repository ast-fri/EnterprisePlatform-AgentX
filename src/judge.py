# src/green_agent/judge.py

from typing import Dict, Any
import json

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()


_llm_judge = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-02-01",
    temperature=0,
)


async def judge_task(
    task_query: str,
    tools_schema: list[dict],
    tool_trace: list[dict],
    final_answer: str,
) -> Dict[str, Any]:
    """
    Use an LLM-as-a-judge to score:
      - tool selection & arguments
      - final answer quality
    Returns a dict with numeric scores and explanation.
    """

    tools_schema_json = json.dumps({"tools": tools_schema}, indent=2)
    tool_trace_json = json.dumps(tool_trace, indent=2)

    system_prompt = """
You are an impartial evaluator of an AI agent that solves tasks by calling tools.

You will receive:
- The original task.
- A catalog of available tools and their argument schemas.
- The sequence of tool calls the agent made (name, args, result).
- The agent's final answer.

Your job is to assign **partial scores** for:
1. tool_use: Did the agent choose appropriate tools and arguments, in a reasonable order?
2. answer_quality: Is the final answer correct, complete, and well-justified given the tool results?

Scoring rules (0 to 1):
- 1.0 = excellent, 0.75 = good, 0.5 = partially correct, 0.25 = poor, 0.0 = unusable.
Return a single JSON object with fields:
{
  "tool_use": float,
  "answer_quality": float,
  "overall": float,
  "explanation": "short justification"
}
Do not add any text outside of this JSON.
    """.strip()

    user_prompt = f"""
[Task]
{task_query}

[Available Tools Schema]
{tools_schema_json}

[Tool Trace]
{tool_trace_json}

[Final Answer]
{final_answer}
"""

    resp = await _llm_judge.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    content = str(resp.content)

    # try to parse JSON; if it fails, fall back to zeros
    try:
        parsed = json.loads(content)
        tool_use = float(parsed.get("tool_use", 0.0))
        answer_quality = float(parsed.get("answer_quality", 0.0))
        overall = float(parsed.get("overall", (tool_use + answer_quality) / 2))
        explanation = str(parsed.get("explanation", ""))
    except Exception:
        tool_use = 0.0
        answer_quality = 0.0
        overall = 0.0
        explanation = "Judge failed to return valid JSON."

    return {
        "tool_use": tool_use,
        "answer_quality": answer_quality,
        "overall": overall,
        "explanation": explanation,
    }
