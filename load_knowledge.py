import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

client = chromadb.PersistentClient(path="./chroma_db")

openai_ef =  embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("CHROMA_OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="marketing_knowledge",
    embedding_function=openai_ef
)

knowledge_folder = "./knowledge"
documents = []
ids = []
metadatas = []

structured_knowledge = [
    # =====================
    # 🧠 FRAMEWORKS
    # =====================
    {
        "id": "framework_aida",
        "content": "Attention → Interest → Desire → Action. Use for ads, landing pages, email copy.",
        "metadata": {"type": "framework", "name": "AIDA"}
    },
    {
        "id": "framework_pas",
        "content": "Problem → Agitation → Solution. Use for persuasive copywriting.",
        "metadata": {"type": "framework", "name": "PAS"}
    },
    {
        "id": "framework_storybrand",
        "content": "Customer is hero, brand is guide. Used for messaging and positioning.",
        "metadata": {"type": "framework", "name": "StoryBrand"}
    },

    # =====================
    # ✍️ TEMPLATES
    # =====================
    {
        "id": "template_linkedin_hook",
        "content": "Controversial statement + Proof + Promise",
        "metadata": {"type": "template", "name": "LinkedIn Hook"}
    },
    {
        "id": "template_email_subject",
        "content": "Pain + Specific insight (under 8 words)",
        "metadata": {"type": "template", "name": "Cold Email Subject"}
    },

    # =====================
    # 📊 EXAMPLES
    # =====================
    {
        "id": "example_linkedin",
        "content": "Grew email list from 0 to 10,000 in 90 days. No ads. System: [steps].",
        "metadata": {
            "type": "example",
            "name": "LinkedIn Viral Post",
            "why_it_works": "specific numbers + curiosity gap + authority"
        }
    }
]

for item in structured_knowledge:
    documents.append(item["content"])
    ids.append(item["id"])
    metadatas.append(item["metadata"])

for filename in os.listdir(knowledge_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(knowledge_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

        for i, chunk in enumerate(chunks):
            doc_id = f"{filename}_{i}"
            documents.append(chunk)
            ids.append(doc_id)
            metadatas.append({"source": filename})

collection.upsert(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"✅ Loaded {len(documents)} chunks into ChromaDB")
