from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.base.llms.types import ChatMessage ,ImageBlock,TextBlock
from llama_index.core.tools import FunctionTool  


"""
可以看到其实llamaindex 不仅仅是一个rag的框架,还是一个agent开发的框架
"""

load_dotenv()
llm = GoogleGenAI(model="models/gemini-3.1-flash-lite-preview")

# 普通的文本生成
response = llm.complete("你好")
print(response)


# 对话生成
messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Tell me a joke."),
]
chat_response = llm.chat(messages)
print(chat_response)

# 多模态输入
messages = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path="image.jpg"),
            TextBlock(text="描述一下图中的场景"),
        ],
    )
]
chat_response = llm.chat(messages)
print(chat_response.message.content)

# 工具调用
class Song:
    def __init__(self, name: str, artist: str):
        self.name = name
        self.artist = artist


def generate_song(name: str, artist: str) -> Song:
    """Generates a song with provided name and artist."""
    return {"name": name, "artist": artist}


tool = FunctionTool.from_defaults(fn=generate_song)

response = llm.predict_and_call(
    [tool],
    "Pick a random song for me",
)
print(str(response))