import streamlit as st
import os
import sys
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Ensure root path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.langgraph_flow.flow import build_graph

DATA_PATH = "data/DiseaseAndSymptoms.csv"
COLLECTION_NAME = "disease_symptom_collection"
CHROMA_PATH = "chroma_db"

st.set_page_config(page_title="Symptom Checker", layout="centered")
st.title("🧠 Symptom Checker - RAG with LangGraph")

if "dataset_ingested" not in st.session_state:
    st.session_state.dataset_ingested = False

# Ingest Dataset
if st.button("Ingest Dataset"):
    try:
        df = pd.read_csv(DATA_PATH)
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        embed_fn = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-ada-002")
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

        documents, metadatas, ids = [], [], []

        for idx, row in df.iterrows():
            disease = row["Disease"]
            symptoms = [str(sym).strip() for sym in row[1:] if pd.notna(sym) and str(sym).strip()]
            if not symptoms:
                continue
            symptom_text = ", ".join(symptoms)
            text = f"Symptoms: {symptom_text}. Disease: {disease}"
            if 20 < len(text) < 1000:  # filter short or very long text
                documents.append(text)
                metadatas.append({"disease": disease})
                ids.append(f"doc_{idx}")

        if len(collection.get()["ids"]) > 0:
            collection.delete(ids=collection.get()["ids"])

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        st.session_state.dataset_ingested = True
        st.success("Dataset successfully ingested and vectorized!")

    except Exception as e:
        st.error(f"Failed to ingest dataset: {e}")

# Input interface
if st.session_state.dataset_ingested:
    st.markdown("### Describe your symptoms (e.g., 'fever and headache'):")
    user_input = st.text_area("Symptoms Input:")

    if st.button("Get Diagnosis"):
        if user_input.strip() == "":
            st.warning("Please enter a description of your symptoms.")
        else:
            graph = build_graph()
            state = {"input": user_input}
            result = graph.invoke(state)
            st.success("Diagnosis complete!")
            st.markdown(f"### Possible Conditions:\n{result['diagnosis']}")
else:
    st.info("Please ingest the dataset before entering symptoms.")
