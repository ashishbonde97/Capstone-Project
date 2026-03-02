from langgraph.graph import END, StateGraph
from .state import GraphState
from .nodes import retrieve, grade_documents, transform_query, web_search, generate

def decide_to_generate(state: GraphState):
    """
    Conditional edge function that determines the next step after grading.
    """
    print("---CONDITIONAL EDGE: ASSESS GRADED DOCUMENTS---")
    web_fallback = state.get("web_fallback", False)
    
    if web_fallback:
        print("---DECISION: ALL DOCS ARE IRRELEVANT. ROUTING TO QUERY TRANSFORM---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def build_crag_graph():
    """Builds and compiles the Corrective RAG LangGraph."""
    
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    # Build Graph Edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    # Conditional Edge: Generate or Transform (Correct)
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    
    # If we had to correct the query, perform web search next
    workflow.add_edge("transform_query", "web_search")
    
    # After web search, generate the final answer
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    # Compile the workflow into a runnable app
    app = workflow.compile()
    return app