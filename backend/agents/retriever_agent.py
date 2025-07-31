import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

class SymptomRetrieverAgent:
    def __init__(self, collection_name="disease_symptom_collection", db_path="chroma_db"):
        embedding_function = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    def retrieve_symptoms(self, user_input: str, k: int = 3):
        results = self.collection.query(query_texts=[user_input], n_results=k)
        matches = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            matches.append((doc, meta['disease']))
        return matches
