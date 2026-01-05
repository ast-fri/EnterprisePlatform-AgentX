# src/agent.py
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from contextlib import AsyncExitStack

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

# Import your existing modules
from env import (
    load_env_config,
    load_all_tasks,
    prepare_env_for_single_task,
    build_white_agent_task_prompt,
)
from judge import judge_task
from my_util import parse_tags


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Main evaluation flow for EnterpriseArena green agent.
        
        Auto-discovers purple agent from message or environment.
        """
        input_text = get_message_text(message)
        
        # Try to parse XML tags first
        tags = parse_tags(input_text)
        white_agent_url = tags.get("white_agent_url")
        env_config_str = tags.get("env_config")
        
        # If no XML tags, try to discover purple agent
        if not white_agent_url:
            white_agent_url = await self._discover_purple_agent(message)
            
            if not white_agent_url:
                await updater.failed(
                    new_agent_text_message(
                        "❌ Error: Could not discover purple agent URL. "
                        "Please provide <white_agent_url> in message or check network configuration.",
                        context_id=message.context_id,
                        task_id=message.task_id
                    )
                )
                return
        
        # Use default config if not provided
        if not env_config_str:
            env_config_str = json.dumps({
                "tasks_file": "tasks.json",
                "mcp_config_path": "/app/mcp_configs_http.json",
                "task_indices": [0, 1, 2],
                "max_steps": 15
            })
            print(f"⚠️  No env_config found in message, using default")
        
        print(f"🟢 Configuration:")
        print(f"  - Purple agent URL: {white_agent_url}")
        print(f"  - Environment config: {env_config_str}")
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"🟢 Starting EnterpriseArena evaluation...\n"
                f"Purple agent: {white_agent_url}",
                context_id=message.context_id
            )
        )
        
        try:
            # Run evaluation
            summary = await self._run_evaluation(
                white_agent_url=white_agent_url,
                env_config_str=env_config_str,
                updater=updater,
                context_id=message.context_id
            )
            
            # Save results as artifact
            await self._save_results_artifact(summary, updater, message)
            
            # Send summary message
            await self._send_summary(summary, updater, message.context_id)
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            await updater.failed(
                new_agent_text_message(
                    f"❌ Evaluation failed: {type(e).__name__}: {e}",
                    context_id=message.context_id,
                    task_id=message.task_id
                )
            )


    async def _discover_purple_agent(self, message: Message) -> str | None:
        """
        Auto-discover purple agent URL from environment.
        
        Tries multiple methods:
        1. Environment variable
        2. Docker network scan
        3. Extract from message text
        """
        import os
        import re
        import socket
        
        # Method 1: Check environment variable
        purple_url = os.getenv("PURPLE_AGENT_URL")
        if purple_url:
            print(f"🔍 Found purple agent from env: {purple_url}")
            return purple_url
        
        # Method 2: Extract URL from message text
        input_text = get_message_text(message)
        urls = re.findall(r'https?://[^\s<>"]+', input_text)
        if urls:
            print(f"🔍 Found purple agent from message: {urls[0]}")
            return urls[0]
        
        # Method 3: Scan common Docker container names on port 9009
        common_names = [
            "EnterprisePurpleAgent",
            "purple-agent",
            "white-agent",
            "participant",
            "solver",
        ]
        
        for name in common_names:
            try:
                # Try to resolve hostname
                socket.gethostbyname(name)
                candidate_url = f"http://{name}:9009"
                
                # Verify it has an agent-card
                try:
                    import httpx
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        response = await client.get(f"{candidate_url}/.well-known/agent-card.json")
                        if response.status_code == 200:
                            print(f"🔍 Discovered purple agent at: {candidate_url}")
                            return candidate_url
                except Exception:
                    pass
            except socket.gaierror:
                continue
        
        print("❌ Could not auto-discover purple agent")
        return None


    async def _run_evaluation(
        self,
        white_agent_url: str,
        env_config_str: str,
        updater: TaskUpdater,
        context_id: str
    ) -> Dict[str, Any]:
        """Execute the full EnterpriseArena evaluation."""
        env_config = load_env_config(env_config_str)
        
        tasks_file = env_config.get("tasks_file", "tasks.json")
        all_tasks = load_all_tasks(tasks_file)
        
        # Determine which tasks to run
        if "task_indices" in env_config:
            task_indices = env_config["task_indices"]
        else:
            task_indices = list(range(len(all_tasks)))
        
        max_steps = env_config.get("max_steps", 10)
        
        all_metrics: Dict[int, Dict[str, Any]] = {}
        benchmark_start = time.time()
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"📋 Running {len(task_indices)} tasks with max {max_steps} steps each",
                context_id=context_id
            )
        )
        
        # Evaluate each task
        for idx, task_index in enumerate(task_indices):
            print(f"\n{'='*60}")
            print(f"🟢 Task {idx+1}/{len(task_indices)} (index={task_index})")
            print(f"{'='*60}")
            
            task_metrics = await self._evaluate_single_task(
                task_index=task_index,
                white_agent_url=white_agent_url,
                env_config=env_config,
                max_steps=max_steps,
                updater=updater,
                context_id=context_id
            )
            
            all_metrics[task_index] = task_metrics
            
            # Progress update
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"✓ Task {idx+1}/{len(task_indices)} complete "
                    f"(score: {task_metrics['judge'].get('overall', 0):.3f})",
                    context_id=context_id
                )
            )
        
        # Calculate aggregate metrics
        total_elapsed = time.time() - benchmark_start
        num_success = sum(1 for m in all_metrics.values() if m["success"])
        
        overall_scores = [m["judge"].get("overall", 0.0) for m in all_metrics.values()]
        tool_use_scores = [m["judge"].get("tool_use", 0.0) for m in all_metrics.values()]
        answer_quality_scores = [m["judge"].get("answer_quality", 0.0) for m in all_metrics.values()]
        
        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        avg_tool_use = sum(tool_use_scores) / len(tool_use_scores) if tool_use_scores else 0.0
        avg_answer_quality = sum(answer_quality_scores) / len(answer_quality_scores) if answer_quality_scores else 0.0
        
        summary = {
            "metadata": {
                "purple_agent_url": white_agent_url,
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_time_seconds": round(total_elapsed, 2),
            },
            "aggregate_metrics": {
                "num_tasks": len(task_indices),
                "num_success": num_success,
                "success_rate": round(num_success / len(task_indices), 3) if task_indices else 0,
                "avg_overall_score": round(avg_overall, 3),
                "avg_tool_use_score": round(avg_tool_use, 3),
                "avg_answer_quality_score": round(avg_answer_quality, 3),
            },
            "per_task_results": all_metrics,
        }
        
        return summary

    async def _evaluate_single_task(
        self,
        task_index: int,
        white_agent_url: str,
        env_config: Dict[str, Any],
        max_steps: int,
        updater: TaskUpdater,
        context_id: str
    ) -> Dict[str, Any]:
        """Evaluate a single task."""
        task_start = time.time()
        
        # Prepare environment for this task
        task_query, tools_schema, env_runtime, stack = await prepare_env_for_single_task(
            env_config,
            task_index=task_index,
        )
        tools = env_runtime.get("tools", [])
        
        observation: str | None = None
        final_answer: str | None = None
        tool_trace: List[Dict[str, Any]] = []
        
        try:
            # Multi-step interaction loop
            for step in range(max_steps):
                # Build prompt for purple agent
                step_prompt = build_white_agent_task_prompt(
                    task_query=task_query,
                    tools_schema=tools_schema,
                    observation=observation,
                )
                
                print(f"  Step {step}: Querying purple agent...")
                
                try:
                    # Use messenger to talk to purple agent
                    purple_response = await self.messenger.talk_to_agent(
                        message=step_prompt,
                        url=white_agent_url,
                        new_conversation=(step == 0),
                        timeout=120
                    )
                except Exception as e:
                    observation = f"Error: Purple agent failed at step {step}: {type(e).__name__}: {e}"
                    print(f"  ❌ {observation}")
                    break
                
                # Parse action from response
                action_tags = parse_tags(purple_response)
                action_json_str = action_tags.get("action")
                
                if not action_json_str:
                    observation = "Error: No <action> block in purple agent response"
                    print(f"  ⚠️  {observation}")
                    continue
                
                try:
                    action = json.loads(action_json_str)
                except Exception as e:
                    observation = f"Error: Invalid action JSON: {e}"
                    print(f"  ⚠️  {observation}")
                    continue
                
                action_type = action.get("type")
                
                if action_type == "tool_call":
                    tool_name = action.get("tool")
                    args = action.get("args", {}) or {}
                    
                    # Find and execute tool
                    tool_obj = next(
                        (t for t in tools if getattr(t, "name", None) == tool_name),
                        None,
                    )
                    
                    if tool_obj is None:
                        observation = f"Error: Tool '{tool_name}' not found"
                        print(f"  ⚠️  {observation}")
                        continue
                    
                    try:
                        result = await tool_obj.ainvoke(args)
                        observation = str(result)
                        print(f"  ✓ Tool '{tool_name}' executed successfully")
                    except Exception as e:
                        observation = f"Tool '{tool_name}' error: {type(e).__name__}: {e}"
                        print(f"  ⚠️  {observation}")
                    
                    tool_trace.append({
                        "step": step,
                        "tool": tool_name,
                        "args": args,
                        "result": observation,
                    })
                
                elif action_type == "final_answer":
                    final_answer = action.get("content", "")
                    print(f"  ✅ Final answer received")
                    break
                
                else:
                    observation = f"Error: Unknown action type '{action_type}'"
                    print(f"  ⚠️  {observation}")
            
            # Judge the task
            task_elapsed = time.time() - task_start
            success = bool(final_answer)
            
            judge_scores = await judge_task(
                task_query=task_query,
                tools_schema=tools_schema,
                tool_trace=tool_trace,
                final_answer=final_answer or "",
            )
            
            return {
                "task_index": task_index,
                "task_query": task_query,
                "success": success,
                "time_used": round(task_elapsed, 2),
                "num_steps": len(tool_trace),
                "final_answer": final_answer or "",
                "tool_trace": tool_trace,
                "judge": judge_scores,
            }
        
        finally:
            # Clean up MCP sessions
            await stack.aclose()

    async def _save_results_artifact(
        self,
        summary: Dict[str, Any],
        updater: TaskUpdater,
        message: Message
    ) -> None:
        """Save results as a structured artifact."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"assessment_{timestamp}.json"
        
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"🟢 Saved results to {results_path}")
        
        # Add as artifact
        await updater.add_artifact(
            parts=[Part(root=DataPart(
                data=summary,
                media_type="application/json"
            ))],
            name=f"assessment_results_{timestamp}",
        )

    async def _send_summary(
        self,
        summary: Dict[str, Any],
        updater: TaskUpdater,
        context_id: str
    ) -> None:
        """Send human-readable summary."""
        metrics = summary["aggregate_metrics"]
        
        result_emoji = "✅" if metrics["success_rate"] == 1.0 else "❌"
        
        summary_text = f"""{result_emoji} **EnterpriseArena Evaluation Complete**

**Overall Performance:**
- Tasks Completed: {metrics['num_success']}/{metrics['num_tasks']}
- Success Rate: {metrics['success_rate']*100:.1f}%
- Average Score: {metrics['avg_overall_score']:.3f}/1.0
- Total Time: {summary['metadata']['total_time_seconds']:.1f}s

**Detailed Scores:**
- Tool Use: {metrics['avg_tool_use_score']:.3f}/1.0
- Answer Quality: {metrics['avg_answer_quality_score']:.3f}/1.0

**Per-Task Breakdown:**
"""
        
        for task_idx, task_result in summary["per_task_results"].items():
            status = "✓" if task_result["success"] else "✗"
            score = task_result["judge"].get("overall", 0)
            summary_text += f"\n{status} Task {task_idx}: {score:.3f} ({task_result['num_steps']} steps)"
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(summary_text, context_id=context_id)
        )
