import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# CONFIG
DATA_PATH = "data/DiseaseAndSymptoms.csv"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "disease_symptom_collection"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("🔍 Starting debug ingestion...")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Prepare documents
documents, metadatas, ids = [], [], []

for idx, row in df.iterrows():
    disease = row["Disease"]
    symptoms = [str(sym).strip() for sym in row[1:] if pd.notna(sym) and str(sym).strip()]
    if not symptoms:
        continue
    symptom_text = ", ".join(symptoms).replace("\n", " ").replace("\r", " ").strip()
    text = f"Symptoms: {symptom_text}. Disease: {disease}"
    text = text.strip()
    if 20 < len(text) < 1000:
        documents.append(text)
        metadatas.append({"disease": disease})
        ids.append(f"doc_{idx}")
    if len(documents) == 5:  # limit for debug
        break

print("\n🧪 Preview of documents to embed:")
for i, doc in enumerate(documents):
    print(f"Doc {i+1}: {doc[:200]}...")

if not OPENAI_API_KEY:
    print("\n❌ OPENAI_API_KEY not found in environment. Please set it.")
    exit(1)

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embed_fn = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-ada-002")
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

# Clear and ingest
try:
    if len(collection.get()["ids"]) > 0:
        collection.delete(ids=collection.get()["ids"])
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print("\n✅ Ingestion successful with 5 documents.")
except Exception as e:
    print(f"\n❌ Error during ingestion: {e}")
