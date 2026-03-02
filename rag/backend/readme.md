🌌 Agentic Corrective RAG (CRAG) Explorer

This repository contains an Agentic Corrective Retrieval-Augmented Generation (CRAG) application. It utilizes a LangGraph-based agentic workflow to smartly route user queries, evaluate the relevance of local context, and seamlessly fallback to web searches when local knowledge is insufficient.

🛠️ Technologies & Libraries Used

LangGraph: For orchestrating the agentic workflow and defining the state machine (Nodes and Edges).

LangChain: For LLM wrapping, prompting, and chaining operations.

Groq (Llama-3.1-8b-instant): Lightning-fast LLM inference powering the Grader, Query Transformer, and Answer Generation nodes.

FAISS: Local, lightweight vector database for fast similarity search of PDF embeddings.

HuggingFace Embeddings (all-MiniLM-L6-v2): Open-source, local embedding model to vectorize PDF chunks.

DuckDuckGo Search: Free, unauthenticated web search API for external knowledge fallback.

Streamlit: For the interactive, visually engaging chat frontend.

🧠 Architecture Logic (How it Works)

Retrieve Node: Fetches the top 3 most relevant document chunks from the local FAISS vector store.

Grade Node (Evaluation): An LLM acts as a grader. It reads the retrieved documents and gives a binary (yes/no) score on their relevance to the user's query.

Conditional Routing: * If any document is relevant, the agent routes directly to the Generate Node.

If all documents are irrelevant, the agent triggers a web fallback.

Transform Query Node (Correction): The LLM rewrites the original question into an optimized web search query.

Web Search Node: Queries DuckDuckGo with the optimized question to fetch external context.

Generate Node: Synthesizes the final answer using the filtered context (either from local PDFs or the web).

(See architecture.mmd for the visual flow diagram).

🚀 Setup & Execution Instructions

1. Prerequisites

Ensure you have Python 3.9+ installed.

2. Install Dependencies

pip install -r requirements.txt


3. Environment Variables

Create a .env file in the root directory and add your Groq API key:

GROQ_API_KEY=your_groq_api_key_here


4. Ingest Your Documents

Create a folder named documents in the root directory and place your PDF files inside it. Then, build the local vector database:

python vector_db/ingest.py


(This will create a vector_db/faiss_index folder containing your embeddings).

5. Run the Application

Launch the interactive Streamlit UI:

streamlit run app.py
