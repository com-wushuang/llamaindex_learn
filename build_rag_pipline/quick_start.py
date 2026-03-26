from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings

load_dotenv()

llm = GoogleGenAI(model="models/gemini-3.1-flash-lite-preview")
embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2-preview")

Settings.llm = llm
Settings.embed_model = embed_model


documents = SimpleDirectoryReader("./data").load_data()

# 建立索引的时候需要使用大模型厂商提供的embedding模型,否则会报错
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("讲了一个什么故事？")
print(response)