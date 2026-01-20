from rag_system.core import OptimizationRAG

def main():
    print("🚀 Starting Pure Python RAG Demo (Framework-Free)...")
    print("-" * 50)

    knowledge_base = (
        "Antigravity is an AI agent designed by Google Deepmind. "
        "It specializes in pair programming and helping users build web applications. "
        "The primary goal of Antigravity is to write high-quality, production-ready code. "
        "It prefers using Vanilla CSS over Tailwind unless requested. "
        "It runs on Windows OS. "
    )

    rag = OptimizationRAG()

    # Ingest
    print("\n[Step 1] Ingesting Knowledge Base...")
    rag.ingest_text(knowledge_base)

    # Query
    question = "What OS does it run on?"
    print(f"\n[Step 2] Asking Question: '{question}'")
    
    response = rag.query(question)
    
    print("\n[Step 3] Retrieval Results:")
    print(response['source_documents'])
    
    print("\n[Step 4] Simulated Answer:")
    print(f"{response['result']}")
    
    print(f"\n[Step 5] Evaluation (Faithfulness Score): {response['eval_score']}")
    print("(A score closer to 1.0 means the answer is well-supported by the document)")

if __name__ == "__main__":
    main()
