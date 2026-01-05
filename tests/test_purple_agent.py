# test_purple_direct.py
import asyncio
import httpx
from uuid import uuid4
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, TextPart, Role


async def test_purple():
    async with httpx.AsyncClient(timeout=60) as httpx_client:
        # Connect to purple agent
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://127.0.0.1:9002")
        agent_card = await resolver.get_agent_card()
        print(f"✅ Purple agent: {agent_card.name}\n")
        
        # Create client
        config = ClientConfig(httpx_client=httpx_client, streaming=False)
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        
        # Simple test message
        test_prompt = """<task>Say hello</task>
<tools_schema_json>
{
  "tools": []
}
</tools_schema_json>
<observation>No previous tool calls have been made yet.</observation>"""
        
        message = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(kind="text", text=test_prompt))],
            message_id=uuid4().hex,
        )
        
        print("📤 Sending test message to purple agent...")
        
        async for event in client.send_message(message):
            print(f"\n{'='*60}")
            print(f"📨 Event type: {type(event)}")
            print(f"{'='*60}")
            
            if isinstance(event, tuple):
                task, _ = event
                print(f"Task status: {task.status.state}")
                
                # Print artifacts
                if task.artifacts:
                    print("\n📦 Artifacts received:")
                    for artifact in task.artifacts:
                        print(f"  Name: {artifact.name}")
                        for part in artifact.parts:
                            if hasattr(part.root, 'text'):
                                print(f"  Content:\n{part.root.text}")
                else:
                    print("⚠️  No artifacts returned!")
                
                # Print status message
                if task.status.message:
                    from a2a.utils import get_text_parts
                    text = '\n'.join(get_text_parts(task.status.message.parts))
                    print(f"\nStatus message:\n{text}")


asyncio.run(test_purple())
