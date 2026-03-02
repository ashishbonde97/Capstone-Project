import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store():
    """Reads PDFs from the documents folder, chunks them, and saves to FAISS."""
    docs_dir = "documents"
    
    # Ensure the documents directory exists
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Created '{docs_dir}' directory. Please place your PDFs there and run again.")
        return

    print("Loading PDFs...")
    loader = PyPDFDirectoryLoader(docs_dir)
    docs = loader.load()
    
    if not docs:
        print("No PDFs found in the 'documents' directory. Please add some and retry.")
        return

    print(f"Loaded {len(docs)} document pages. Chunking...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks. Generating embeddings...")
    # Using open-source local embeddings so you don't need OpenAI keys
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the vector store locally in the vector_db folder
    save_path = "vector_db/faiss_index"
    vector_store.save_local(save_path)
    print(f"Vector store successfully saved to '{save_path}'!")

if __name__ == "__main__":
    build_vector_store()