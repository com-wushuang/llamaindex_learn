#!/Users/chengjun/private/RAG/llamaindex_learn/.venv/bin/python
import os
from dotenv import load_dotenv
import chromadb
import nest_asyncio
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.cli.rag import RagCLI
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_parse import LlamaParse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
nest_asyncio.apply()

llm = GoogleGenAI(
    model="models/gemini-3.1-flash-lite-preview",
    api_key=os.getenv("GOOGLE_API_KEY"),
)
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",
    device="cpu",
    trust_remote_code=True,
)

Settings.llm = llm
Settings.embed_model = embed_model


chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_collection")
vec_store = ChromaVectorStore(chroma_collection=chroma_collection)

custom_ingestion_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=200),
        embed_model,
    ],
    vector_store=vec_store,
    cache=IngestionCache(),
)

# 1. 初始化解析器
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),  # 替换为你的 Key
    result_type="markdown",       # 推荐 Markdown，对 AI 最友好
    language="ch_sim",           # 如果文档是中文，建议指定语言
    use_vendor_multimodal_model=True,
)

file_extractor = {
    ".pdf": parser,
}

rag_cli_instance = RagCLI(
    ingestion_pipeline=custom_ingestion_pipeline,
    llm=llm,
    file_extractor=file_extractor,
)

if __name__ == "__main__":
    rag_cli_instance.cli()
