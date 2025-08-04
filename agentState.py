from llama_index.core.workflow import Context
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent

ctx = Context(workflow)


async def main():
    response = await workflow.run(user_msg="What's the current stock price of NVIDIA?")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
