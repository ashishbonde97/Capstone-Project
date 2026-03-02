import streamlit as st
import os
from dotenv import load_dotenv
from pipeline.graph import build_crag_graph

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic CRAG Explorer",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom Colorful CSS ---
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1e3f 50%, #2e0854 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    h1 {
        background: -webkit-linear-gradient(45deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Chat bubble styling for User */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* Chat bubble styling for Assistant */
    [data-testid="stChatMessage"]:nth-child(even) {
        background: rgba(168, 85, 247, 0.1);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* Avatar styling */
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #0ea5e9;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #8b5cf6;
    }

    /* Text input container */
    [data-testid="stChatInput"] {
        border-radius: 25px !important;
        border: 2px solid #8b5cf6 !important;
        background-color: rgba(15, 23, 42, 0.8) !important;
    }
    
    /* Hide top padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("<h1>🌌 Agentic CRAG Explorer</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by LangGraph, FAISS & Groq</div>", unsafe_allow_html=True)

# --- Initialization ---
load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    st.error("🔑 GROQ_API_KEY is missing. Please add it to your .env file.")
    st.stop()

@st.cache_resource
def load_graph():
    """Caches the graph compilation so it doesn't rebuild on every UI interaction."""
    try:
        return build_crag_graph()
    except Exception as e:
        st.error(f"Failed to build graph: {e}\nDid you run `vector_db/ingest.py` first?")
        st.stop()

app_graph = load_graph()

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me a question. I will search your local PDFs first, and fallback to the web if needed!"}
    ]

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input & Execution ---
if prompt := st.chat_input("Ask something about your documents..."):
    # Append user message to state and UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process via LangGraph
    with st.chat_message("assistant"):
        # UI placeholder for the agent's thought process
        status_container = st.status("🧠 Agent is thinking...", expanded=True)
        
        try:
            inputs = {"question": prompt}
            final_generation = ""
            
            # Stream the agent's steps
            for output in app_graph.stream(inputs):
                for node_name, node_state in output.items():
                    # Display the steps nicely in the UI
                    if node_name == "retrieve":
                        status_container.write("📚 Retrieved documents from local FAISS database.")
                    elif node_name == "grade_documents":
                        if node_state.get("web_fallback"):
                            status_container.write("⚠️ Local docs irrelevant. Triggering web fallback...")
                        else:
                            status_container.write("✅ Local documents are relevant!")
                    elif node_name == "transform_query":
                        status_container.write(f"🔄 Rewriting query for web search: *{node_state.get('question')}*")
                    elif node_name == "web_search":
                        status_container.write("🌐 Searched DuckDuckGo for missing context.")
                    elif node_name == "generate":
                        status_container.write("✍️ Generating final response...")
                        final_generation = node_state.get("generation", "Could not generate an answer.")
            
            status_container.update(label="✨ Task Complete!", state="complete", expanded=False)
            
            # Display Final Answer
            st.markdown(final_generation)
            
            # Save assistant response to state
            st.session_state.messages.append({"role": "assistant", "content": final_generation})
            
        except Exception as e:
            status_container.update(label="❌ Error occurred", state="error")
            st.error(f"An error occurred: {e}")