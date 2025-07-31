import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def retrieve_symptoms_tool(user_input: str) -> str:
    """Retrieve diseases that match the given symptom description using ChromaDB."""
    client = chromadb.PersistentClient(path="chroma_db")
    embed_fn = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"),
                                       model_name="text-embedding-ada-002")
    collection = client.get_or_create_collection(name="disease_symptom_collection",
                                                 embedding_function=embed_fn)
    result = collection.query(query_texts=[user_input], n_results=2)
    docs = result["documents"][0]
    return "\n".join([f"Symptoms & Disease: {d}" for d in docs])
