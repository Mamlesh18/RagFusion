"""
quickstart.py
Quick demonstration of RAGFusion capabilities.
"""
from ragfusion import chromadb, faiss, configure

def quickstart():
    """Quick demonstration of RAGFusion."""
    
    print("RAGFusion Quickstart")
    print("=" * 50)
    
    configure(
        embedding__provider="sentence-transformers",
        embedding__model_name="all-MiniLM-L6-v2",
        chunking__chunk_size=300,
        verbose=True
    )
    
    documents = [
        """Artificial Intelligence is revolutionizing technology. 
        Machine learning models can now understand and generate human language,
        recognize images, and make complex decisions.""",
        
        """Climate change is one of the most pressing challenges of our time.
        Rising temperatures, melting ice caps, and extreme weather events
        are becoming more frequent.""",
        
        """Quantum computing represents a fundamental shift in computation.
        By leveraging quantum mechanics, these computers can solve certain
        problems exponentially faster than classical computers.""",
    ]
    
    print("\n1. Using ChromaDB")
    print("-" * 30)
    
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection("demo")
    
    chroma_collection.add_documents(
        documents=documents,
        ids=["doc1", "doc2", "doc3"],
        metadata=[
            {"topic": "AI"},
            {"topic": "Climate"},
            {"topic": "Quantum"},
        ]
    )
    print(f"[OK] Added {len(documents)} documents to ChromaDB")
    
    query = "artificial intelligence and machine learning"
    results = chroma_collection.rag_search(query, k=2)
    
    print(f"\nSearch results for: '{query}' (k={len(results)})")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"  Score: {result.score:.3f}")
        print(f"  Topic: {result.metadata.get('topic')}")
        print(f"  Preview: {result.content[:100]}...")
    
    print("\n" + "=" * 50)
    print("2. Using FAISS")
    print("-" * 30)
    
    faiss_index = faiss.Index(dimension=384) 
    
    faiss_index.add_documents(
        documents=documents,
        ids=["doc1", "doc2", "doc3"],
        metadata=[
            {"topic": "AI"},
            {"topic": "Climate"},
            {"topic": "Quantum"},
        ]
    )
    print(f"[OK] Added {len(documents)} documents to FAISS")
    
    results = faiss_index.rag_search(query, k=2)
    
    print(f"\nSearch results for: '{query}' (k={len(results)})")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"  Score: {result.score:.3f}")
        print(f"  Topic: {result.metadata.get('topic')}")
        print(f"  Preview: {result.content[:100]}...")


if __name__ == "__main__":
    quickstart()