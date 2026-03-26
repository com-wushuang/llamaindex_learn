from dotenv import load_dotenv
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import FunctionAgent


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


load_dotenv()

# 使用 YahooFinanceToolSpec 来获取股票价格
finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([multiply, add])

workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations.",
    llm=GoogleGenAI(model="models/gemini-3.1-flash-lite-preview"),
    tools=finance_tools,
    system_prompt="You are a helpful assistant.",
)


async def main():
    response = await workflow.run(user_msg="What's the current stock price of NVIDIA?")
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
