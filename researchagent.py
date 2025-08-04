import asyncio
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from tavily import AsyncTavilyClient

# LLM setup
llm = OpenAI(model="gpt-4o-mini")

# Define tools
async def search_web(query: str) -> str:
    client = AsyncTavilyClient(api_key="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")
    return str(await client.search(query))

async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    async with ctx.store.edit_state() as ctx_state:
        if "research_notes" not in ctx_state["state"]:
            ctx_state["state"]["research_notes"] = {}
        ctx_state["state"]["research_notes"][notes_title] = notes
    return "Notes recorded."

async def write_report(ctx: Context, report_content: str) -> str:
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["report_content"] = report_content
    return "Report written."

async def review_report(ctx: Context, review: str) -> str:
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["review"] = review
    return "Report reviewed."

# Define agents
from llama_index.core.agent.workflow import FunctionAgent

research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Useful for searching the web for information on a given topic and recording notes on the topic.",
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "Once notes are recorded and you are satisfied, you should hand off control to the WriteAgent to write a report on the topic. "
        "You should have at least some notes on a topic before handing off control to the WriteAgent."
    ),
    llm=llm,
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Useful for writing a report on a given topic.",
    system_prompt=(
        "You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Once the report is written, you should get feedback at least once from the ReviewAgent."
    ),
    llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Useful for reviewing a report and providing feedback.",
    system_prompt=(
        "You are the ReviewAgent that can review the write report and provide feedback. "
        "Your review should either approve the current report or request changes for the WriteAgent to implement. "
        "If you have feedback that requires changes, you should hand off control to the WriteAgent to implement the changes after submitting the review."
    ),
    llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent"],
)

# Create workflow
agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)

# Streaming and output logic
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)

async def main():
    handler = agent_workflow.run(
        user_msg=(
            "Write me a report on the history of the internet. "
            "Briefly describe the history of the internet, including the development of the internet, the development of the web, "
            "and the development of the internet in the 21st century."
        )
    )

    current_agent = None

    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"\n{'='*50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'='*50}\n")

        if isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print("🛠️  Planning to use tools:", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")
              

if __name__ == "__main__":
    asyncio.run(main())
