from llama_cloud import AsyncLlamaCloud
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

async def main():
    client = AsyncLlamaCloud(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))

    # Upload and parse a document
    file_obj = await client.files.create(
        file="data/chart.pdf", purpose="parse"
    )

    result = await client.parsing.parse(
        file_id=file_obj.id,
        # The parsing tier. Options: fast, cost_effective, agentic, agentic_plus,
        tier="agentic",
        # The version of the parsing tier to use. Use 'latest' for the most recent version,
        version="latest",
        # 'expand' controls which result fields are returned in the response.,
        # Without it, only job metadata is returned. Common fields:,
        # - "markdown_full", "text_full": Full document content,
        # - "markdown", "text", "items": Page-level content,
        # - "images_content_metadata": Presigned URLs for images,
        expand=["markdown_full", "text_full"],
    )

    # Access the full document content
    # print("Full markdown:")
    # print(result.markdown_full)

    # print("\nFull text:")
    # print(result.text_full)

    # Save to markdown file
    output_file = "output/chart_parsed.md"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.markdown_full)
    print(f"\n解析结果已保存到: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
