import os
import sys
import types

_fake_trace_exporter = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
_fake_trace_exporter.OTLPSpanExporter = object
sys.modules["opentelemetry.exporter.otlp.proto.grpc.trace_exporter"] = _fake_trace_exporter

import chromadb
from chromadb.utils import embedding_functions
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()
print(__file__)
key = os.getenv("CHROMA_OPENAI_API_KEY")

print("KEY FOUND:", key is not None)
print("KEY PREFIX:", key[:10] if key else None)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=key,
    model_name="text-embedding-3-small"
)
client = chromadb.PersistentClient(path="./chroma_db")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("CHROMA_OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="marketing_knowledge",
    embedding_function=openai_ef
)


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the marketing knowledge base for templates, frameworks, and examples.
    Use this when you need content templates, copywriting frameworks,
    or real examples of high-performing marketing content.
    Example queries: 'LinkedIn post template', 'email subject line formula',
    'B2B copywriting framework'.
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=5
        )

        if not results["documents"][0]:
            return "No relevant knowledge found for that query."

        output = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output.append(f"Source: {meta['source']}\n{doc}")

        return "\n\n---\n\n".join(output)

    except Exception as e:
        return f"Knowledge base error: {e}"