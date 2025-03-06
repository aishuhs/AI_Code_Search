import streamlit as st
import os
import ollama
import chromadb
from chromadb.utils import embedding_functions


# Initialize ChromaDB
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")


def get_collection(client):
    return client.get_or_create_collection(name="codebase")


# Function to extract code from Python files
def extract_code_from_folder(folder_path):
    code_files = []
    if not os.path.exists(folder_path):
        st.error("Folder path does not exist!")
        return []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code_files.append((file, f.read()))
                except Exception as e:
                    st.warning(f"Could not read {file}: {e}")
    return code_files


# Generate embeddings using Ollama
import requests


def generate_embedding(text):
    try:
        # Check if Ollama server is reachable
        test_url = "http://localhost:11434"
        response = requests.get(test_url)
        if response.status_code != 200:
            st.error("Ollama server is not responding. Ensure it is running with 'ollama serve'.")
            return []

        # Generate embeddings
        response = ollama.embeddings(model="mistral", prompt=text)
        return response.get("embedding", [])

    except requests.ConnectionError:
        st.error("Failed to connect to Ollama. Ensure it is running with 'ollama serve'.")
        return []

    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return []


# Ingest code into ChromaDB
def ingest_code(folder_path, collection):
    code_files = extract_code_from_folder(folder_path)
    if not code_files:
        st.warning("No Python files found!")
        return

    for file_name, code in code_files:
        embedding = generate_embedding(code)
        if embedding:
            collection.add(documents=[code], metadatas=[{"file": file_name}], ids=[file_name])

    st.success("Code files successfully indexed!")
def query_ollama(model, prompt):
    """Sends a query to Ollama and returns the response."""
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]  # Extract the explanation
    except Exception as e:
        st.error(f"Error querying Ollama: {e}")
        return "Failed to generate an explanation."


# Search codebase
def search_code(query, collection):
    try:
        results = collection.query(query_texts=[query], n_results=3)
        documents, metadatas = results["documents"], results["metadatas"]

        explanations = []
        for doc in documents:
            explanation = query_ollama("mistral", f"Explain this Python code: {doc}")
            explanations.append(explanation)

        return documents, metadatas, explanations
    except Exception as e:
        st.error(f"Error searching codebase: {e}")
        return [], [], []




# Streamlit UI
st.title("AI Code Search with Vector Database")

# File Upload Section
folder_path = st.text_input("Enter the folder path containing Python files:")
if st.button("Ingest Code") and folder_path:
    client = get_chroma_client()
    collection = get_collection(client)
    ingest_code(folder_path, collection)

# Search Section
# Search Section
query = st.text_input("Ask a question about the code:")
if st.button("Search") and query:
    client = get_chroma_client()
    collection = get_collection(client)
    results, metadata, explanations = search_code(query, collection)

    st.write("### Search Results")
    if results:
        for i, result in enumerate(results):
            if metadata and len(metadata) > i and isinstance(metadata[i], dict):
                file_name = metadata[i].get("file", "Unknown file")
            else:
                file_name = "Unknown file"
            st.subheader(f"🔍 Found in: {file_name}")
            st.code(result, language="python")

            # Show explanation
            st.write("💡 **AI Explanation:**")
            st.write(explanations[i])  # Display explanation from Ollama
    else:
        st.warning("No relevant code found!")

