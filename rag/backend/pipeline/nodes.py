import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .state import GraphState
from dotenv import load_dotenv

# --- Configuration & Setup ---
# Initialize LLM via Groq (Ensure GROQ_API_KEY is in your environment/.env)
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0,groq_api_key=groq_api_key)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_retriever():
    """Helper to load the FAISS DB"""
    db_path = "vector_db/faiss_index"
    if not os.path.exists(db_path):
        raise FileNotFoundError("FAISS index not found. Run vector_db/ingest.py first.")
    # Allow dangerous deserialization is required for local FAISS loading in newer versions
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Nodes ---

def retrieve(state: GraphState):
    """Retrieves documents from the local FAISS vector store."""
    print("---NODE: RETRIEVE FROM LOCAL DB---")
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state: GraphState):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    # Prompt to evaluate relevance
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        Provide ONLY 'yes' or 'no' as your output, without any extra text.
        """,
        input_variables=["question", "document"],
    )
    
    grader_chain = prompt | llm | StrOutputParser()

    filtered_docs = []
    web_fallback = False

    for d in documents:
        score = grader_chain.invoke({"question": question, "document": d.page_content})
        grade = score.strip().lower()
        if grade == "yes":
            print("  - Grader: Document is relevant")
            filtered_docs.append(d)
        else:
            print("  - Grader: Document is IRRELEVANT")

    # If no relevant documents are found, trigger web search fallback via correction
    if len(filtered_docs) == 0:
        print("  - Grader: No relevant documents found. Triggering correction/fallback.")
        web_fallback = True

    return {"documents": filtered_docs, "question": question, "web_fallback": web_fallback}


def transform_query(state: GraphState):
    """Correction Node: Rewrites the question to improve retrieval/search."""
    print("---NODE: TRANSFORM QUERY (CORRECTION)---")
    question = state["question"]

    prompt = PromptTemplate(
        template="""You are an expert at crafting search queries.
        The user asked a question, but our local database couldn't find a good match. 
        Please rewrite the following question to be a better, more detailed web search query.
        Original question: {question}
        Provide ONLY the rewritten question as your output without any quotes or prefixes.
        """,
        input_variables=["question"],
    )

    rewrite_chain = prompt | llm | StrOutputParser()
    better_question = rewrite_chain.invoke({"question": question})
    print(f"  - Original: {question}")
    print(f"  - Rewritten: {better_question}")
    
    return {"documents": state["documents"], "question": better_question}


def web_search(state: GraphState):
    """Web search based on the re-phrased question."""
    print("---NODE: WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
    tool = DuckDuckGoSearchResults(api_wrapper=wrapper)
    
    docs = tool.invoke(question)
    # DuckDuckGo returns a string of results, we wrap it in a Document
    web_results = Document(page_content=docs, metadata={"source": "duckduckgo"})
    documents.append(web_results)

    return {"documents": documents, "question": question}


def generate(state: GraphState):
    """Generates the final answer based on the context documents."""
    print("---NODE: GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]

    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        Question: {question} 
        
        Context: {context} 
        
        Answer:
        """,
        input_variables=["question", "context"],
    )

    rag_chain = prompt | llm | StrOutputParser()
    
    # Combine the documents into a single context string
    context = "\n\n".join(doc.page_content for doc in documents)
    
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}