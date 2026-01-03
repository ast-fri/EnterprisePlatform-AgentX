# test_green_agent.py
import asyncio
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, TextPart, Role


async def test_green_agent():
    base_url = "http://127.0.0.1:9009"
    
    async with httpx.AsyncClient(timeout=300) as httpx_client:
        # Get agent card
        print("📋 Fetching agent card...")
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        print(f"✅ Agent: {agent_card.name}\n")
        
        # Create client
        config = ClientConfig(httpx_client=httpx_client, streaming=False)
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        
        # Create test message
        # Note: You'll need a purple agent running on port 9002 for this to actually work
        assessment_message = """<white_agent_url>http://localhost:9002</white_agent_url>
<env_config>
{
  "tasks_file": "tasks.json",
  "task_indices": [0],
  "max_steps": 10
}
</env_config>"""
        
        message = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(kind="text", text=assessment_message))],
            message_id=uuid4().hex,
        )
        
        print("📤 Sending assessment request...")
        print(f"   Purple agent: http://localhost:9002")
        print(f"   Tasks: [0]\n")
        
        # Send and receive
        async for event in client.send_message(message):
            print(f"\n{'='*60}")
            print(f"📨 Event: {type(event).__name__}")
            print(f"{'='*60}")
            
            # Pretty print the event
            if hasattr(event, 'status'):
                print(f"Status: {event.status.state}")
                if event.status.message:
                    from a2a.utils import get_text_parts
                    text = '\n'.join(get_text_parts(event.status.message.parts))
                    print(f"Message:\n{text}")
            elif hasattr(event, 'parts'):
                from a2a.utils import get_text_parts
                text = '\n'.join(get_text_parts(event.parts))
                print(f"Message:\n{text}")
            else:
                print(event)


if __name__ == "__main__":
    asyncio.run(test_green_agent())
