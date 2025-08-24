"""
examples/basic_usage.py
Basic usage examples for RAGFusion.
"""

from ragfusion import chromadb, faiss, configure
from ragfusion.config import Config, EmbeddingConfig, ChunkingConfig

# ============================================================================
# Example 1: Basic ChromaDB Usage
# ============================================================================

def example_chromadb():
    """Example using ChromaDB with RAG enhancements."""
    print("=" * 50)
    print("ChromaDB Example")
    print("=" * 50)
    
    # Initialize ChromaDB client
    client = chromadb.Client()
    
    # Create a collection
    collection = client.create_collection("my_documents")
    
    # Add documents with automatic chunking and embedding
    documents = [
        "Machine learning is a subset of artificial intelligence...",
        "Deep learning uses neural networks with multiple layers...",
        "Natural language processing enables computers to understand human language...",
    ]
    
    # Add documents - they will be automatically chunked and embedded
    doc_ids = collection.add_documents(
        documents=documents,
        ids=["doc1", "doc2", "doc3"],
        metadata=[
            {"topic": "ML", "source": "wikipedia"},
            {"topic": "DL", "source": "textbook"},
            {"topic": "NLP", "source": "blog"},
        ],
        chunk=True,  # Enable automatic chunking
    )
    
    print(f"Added {len(doc_ids)} documents")
    
    # Search with RAG features
    results = collection.rag_search(
        query="What is deep learning?",
        k=5,
        strategy="similarity",
        rerank=False,
    )
    
    print(f"\nSearch results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.score:.3f}")
        print(f"   Content: {result.content[:100]}...")
        print(f"   Metadata: {result.metadata}")
    
    # Hybrid search (combines semantic and keyword search)
    hybrid_results = collection.hybrid_search(
        query="neural networks",
        k=3,
        alpha=0.7,  # Weight for semantic search
    )
    
    print(f"\nHybrid search results: {len(hybrid_results)} found")


# ============================================================================
# Example 2: Basic FAISS Usage
# ============================================================================

def example_faiss():
    """Example using FAISS with document storage."""
    print("\n" + "=" * 50)
    print("FAISS Example")
    print("=" * 50)
    
    # Initialize FAISS index with automatic dimension detection
    index = faiss.Index(
        dimension=None,  # Will be auto-detected from embedding model
        index_type="Flat",  # Use Flat index for exact search
        metric="L2",
    )
    
    # Add documents
    documents = [
        {
            "id": "doc1",
            "content": "Python is a high-level programming language...",
            "metadata": {"category": "programming", "language": "en"}
        },
        {
            "id": "doc2",
            "content": "Data science combines statistics and programming...",
            "metadata": {"category": "data science", "language": "en"}
        },
    ]
    
    # Add documents with automatic embedding
    doc_ids = index.add_documents(documents, chunk=True)
    print(f"Added {len(doc_ids)} documents to FAISS")
    
    # Search
    results = index.rag_search(
        query="programming with Python",
        k=3,
        filters={"language": "en"},  # Metadata filtering
    )
    
    print(f"\nSearch results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.score:.3f}")
        print(f"   Document ID: {result.document_id}")
        print(f"   Content: {result.content[:100]}...")
    
    # Save the index
    index.save("./my_faiss_index")
    print("\nIndex saved to ./my_faiss_index")
    
    # Load the index
    loaded_index = faiss.Index.load("./my_faiss_index")
    print("Index loaded successfully")


# ============================================================================
# Example 3: Configuration
# ============================================================================

def example_configuration():
    """Example of configuring RAGFusion."""
    print("\n" + "=" * 50)
    print("Configuration Example")
    print("=" * 50)
    
    # Method 1: Configure globally using keyword arguments
    configure(
        verbose=True,
        embedding__model_name="all-MiniLM-L6-v2",
        embedding__provider="sentence-transformers",
        chunking__strategy="recursive",
        chunking__chunk_size=300,
        chunking__chunk_overlap=50,
        retrieval__strategy="similarity",
        retrieval__top_k=10,
    )
    
    print("Global configuration updated")
    
    # Method 2: Create a custom configuration
    custom_config = Config(
        embedding=EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-ada-002",
            api_key="your-api-key-here",  # Or use environment variable
        ),
        chunking=ChunkingConfig(
            strategy="semantic",
            chunk_size=500,
            chunk_overlap=100,
        ),
        verbose=True,
    )
    
    # Save configuration
    custom_config.save("./my_config.json")
    print("Configuration saved to ./my_config.json")
    
    # Load configuration
    loaded_config = Config.load("./my_config.json")
    print("Configuration loaded")


# ============================================================================
# Example 4: Migration Between Stores
# ============================================================================

def example_migration():
    """Example of migrating between vector stores."""
    print("\n" + "=" * 50)
    print("Migration Example")
    print("=" * 50)
    
    # Create and populate ChromaDB
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection("source_collection")
    
    documents = [
        "Document 1 content...",
        "Document 2 content...",
        "Document 3 content...",
    ]
    
    chroma_collection.add_documents(documents, ids=["d1", "d2", "d3"])
    print(f"Added {len(documents)} documents to ChromaDB")
    
    # Migrate to FAISS
    print("\nMigrating ChromaDB -> FAISS...")
    chroma_client.migrate_to_faiss(
        collection_name="source_collection",
        faiss_index_path="./migrated_index",
        batch_size=100,
    )
    
    # Load the migrated FAISS index
    faiss_index = faiss.Index.load("./migrated_index")
    print(f"FAISS index loaded with {faiss_index.count()} documents")
    
    # You can also migrate from FAISS to ChromaDB
    print("\nMigrating FAISS -> ChromaDB...")
    new_chroma_client = chromadb.Client()
    faiss_index.migrate_to_chromadb(
        client=new_chroma_client,
        collection_name="migrated_collection",
        batch_size=100,
    )


# ============================================================================
# Example 5: Advanced RAG Features
# ============================================================================

def example_advanced_rag():
    """Example of advanced RAG features."""
    print("\n" + "=" * 50)
    print("Advanced RAG Example")
    print("=" * 50)
    
    # Configure for advanced usage
    configure(
        embedding__provider="sentence-transformers",
        embedding__model_name="all-mpnet-base-v2",  # Better model
        chunking__strategy="semantic",  # Semantic chunking
        retrieval__strategy="mmr",  # Maximum Marginal Relevance
    )
    
    # Create a collection
    client = chromadb.Client()
    collection = client.create_collection("advanced_rag")
    
    # Add a longer document that will be chunked
    long_document = """
    Artificial Intelligence (AI) is transforming the world in unprecedented ways.
    Machine learning, a subset of AI, enables computers to learn from data without
    explicit programming. Deep learning, using neural networks, has achieved
    remarkable results in computer vision, natural language processing, and more.
    
    The applications of AI are vast and growing. In healthcare, AI assists in
    diagnosis, drug discovery, and personalized treatment plans. In finance,
    it powers fraud detection, algorithmic trading, and risk assessment.
    Transportation is being revolutionized by autonomous vehicles and smart
    traffic management systems.
    
    However, AI also presents challenges. Ethical considerations around bias,
    privacy, and job displacement need careful attention. The explainability
    of AI decisions, especially in critical applications, remains a key concern.
    Ensuring AI safety and alignment with human values is crucial as these
    systems become more powerful.
    """
    
    # Add with semantic chunking
    collection.add_documents(
        [long_document],
        ids=["ai_overview"],
        metadata=[{"source": "technical_report", "year": 2024}],
        chunk=True,
    )
    
    # Search with different strategies
    
    # 1. Standard similarity search
    similarity_results = collection.rag_search(
        query="What are the applications of AI?",
        k=3,
        strategy="similarity",
    )
    print("\nSimilarity search results:", len(similarity_results))
    
    # 2. MMR search for diversity
    mmr_results = collection.rag_search(
        query="AI challenges and applications",
        k=3,
        strategy="mmr",
    )
    print("MMR search results (diverse):", len(mmr_results))
    
    # 3. Search with reranking
    reranked_results = collection.rag_search(
        query="ethical AI",
        k=5,
        strategy="similarity",
        rerank=True,  # Use cross-encoder for reranking
    )
    print("Reranked results:", len(reranked_results))
    
    # 4. Evaluate retrieval performance
    test_queries = [
        "AI in healthcare",
        "machine learning applications",
        "AI ethics and challenges",
    ]
    
    # Expected relevant chunks for each query
    ground_truth = [
        ["ai_overview_chunk_1"],
        ["ai_overview_chunk_0", "ai_overview_chunk_1"],
        ["ai_overview_chunk_2"],
    ]
    
    metrics = collection.evaluate_retrieval(
        test_queries=test_queries,
        ground_truth=ground_truth,
        k=3,
        metrics=["precision", "recall", "f1"],
    )
    
    print("\nRetrieval evaluation metrics:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.3f}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run examples
    example_chromadb()
    example_faiss()
    example_configuration()
    example_migration()
    example_advanced_rag()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)