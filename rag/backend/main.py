import os
from dotenv import load_dotenv
from pipeline.graph import build_crag_graph

# Load environment variables (Make sure .env exists with GROQ_API_KEY)
load_dotenv()

def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found in environment variables. Please check your .env file.")
        return

    print("Initializing CRAG pipeline...")
    app = build_crag_graph()
    
    while True:
        print("\n" + "="*50)
        user_query = input("Enter your question (or type 'quit' to exit): ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_query.strip():
            continue

        inputs = {"question": user_query}
        
        # Stream the graph execution to see the agentic process
        for output in app.stream(inputs):
            for key, value in output.items():
                # Print node state keys just to track progress visually
                pass 
                
        
        # Workaround for getting the final result if stream finishes
        print("\n---FINAL ANSWER---")
        if "generate" in output:
            print(output["generate"]["generation"])
        else:
            # Fallback if streaming exits differently
            for key, value in output.items():
                if "generation" in value:
                    print(value["generation"])

if __name__ == "__main__":
    main()