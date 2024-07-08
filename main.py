import gradio as gr
import concurrent.futures
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
import ollama
import os

# Function to load a single file (CSV or Excel)


def load_file(filepath):
    if filepath.endswith(".xlsx"):
        data = pd.read_excel(filepath)
    elif filepath.endswith(".csv"):
        data = pd.read_csv(filepath)
    else:
        return None
    text = data.to_string(index=False)  # Convert the DataFrame to a string
    return Document(page_content=text)

# Function to load files from a folder in parallel


def load_files_from_folder_parallel(folder_path):
    docs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_file, os.path.join(folder_path, filename))
                   for filename in os.listdir(folder_path) if filename.endswith((".xlsx", ".csv"))]
        for future in concurrent.futures.as_completed(futures):
            doc = future.result()
            if doc is not None:
                docs.append(doc)
    return docs

# Function to load, split, and retrieve documents


def load_and_retrieve_docs(folder_path):
    docs = load_files_from_folder_parallel(folder_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# Function to format documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain


def rag_chain(folder_path, question):
    retriever = load_and_retrieve_docs(folder_path)
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}\n\nNote: Only use the provided context to answer the question. Do not use any external knowledge."
    response = ollama.chat(model='llama3', messages=[
                           {'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']


# Gradio interface
iface = gr.Interface(
    fn=rag_chain,
    inputs=["text", "text"],
    outputs="text",
    title="RAG Chain Question Answering",
    description="Enter a folder path containing CSV and Excel files and a query to get answers from the RAG chain."
)

# Launch the app
iface.launch()
