from dotenv import load_dotenv
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

workflow = FunctionAgent(
    llm=GoogleGenAI(model="models/gemini-3.1-flash-lite-preview"),
    system_prompt="You are a helpful assistant.",
)


ctx = Context(workflow)


async def main():
    response = await workflow.run(user_msg="Hi, my name is Laurie!", ctx=ctx)
    print(response)

    response2 = await workflow.run(user_msg="What's my name?", ctx=ctx)
    print(response2)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
