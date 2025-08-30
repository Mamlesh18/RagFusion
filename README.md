
"""
# RAGFusion 🚀

A unified RAG (Retrieval-Augmented Generation) framework that provides seamless switching between vector stores while adding powerful RAG-specific features.


## Quick Start

```python
from ragfusion import chromadb, faiss

# Use ChromaDB with all its native features + RAG enhancements
client = chromadb.Client()
collection = client.create_collection("my_docs")

# Add documents with automatic chunking and embedding
collection.add_documents([
    "Your first document...",
    "Your second document...",
])

# Enhanced search with reranking
results = collection.rag_search(
    query="your query",
    k=5,
    strategy="hybrid",
    rerank=True
)

# Switch to FAISS with one line
index = faiss.Index(dimension=768)
index.add_documents(["Same documents..."])
results = index.rag_search("same query", k=5)
```

## Key Advantages

### 1. No Feature Loss
Unlike traditional abstraction layers, RAGFusion preserves 100% of each vector store's native capabilities:

```python
from ragfusion import chromadb

# All ChromaDB features work exactly as expected
client = chromadb.Client()
collection = client.create_collection(
    name="test",
    metadata={"hnsw:space": "cosine"}  # Native ChromaDB configuration
)
collection.peek()  # Native method
collection.query()  # Native method
collection.rag_search()  # RAGFusion enhancement
```

### 2. Enhanced RAG Features

Every vector store gets powerful RAG capabilities:

- **Smart Chunking**: Multiple strategies (recursive, semantic, sentence-based)
- **Embedding Management**: Support for OpenAI, Cohere, SentenceTransformers
- **Advanced Search**: Hybrid search, MMR, cross-encoder reranking
- **Document Management**: Automatic handling of documents and metadata
- **Evaluation Tools**: Built-in metrics for retrieval quality

### 3. Easy Migration

Migrate between stores with built-in tools:

```python
# Migrate ChromaDB -> FAISS
chroma_client.migrate_to_faiss(
    collection_name="source",
    faiss_index_path="./migrated"
)

# Migrate FAISS -> ChromaDB
faiss_index.migrate_to_chromadb(
    client=chroma_client,
    collection_name="destination"
)
```

## Configuration

```python
from ragfusion import configure

# Global configuration
configure(
    embedding__provider="openai",
    embedding__model_name="text-embedding-ada-002",
    chunking__strategy="semantic",
    chunking__chunk_size=500,
    retrieval__strategy="hybrid"
)

# Or use configuration files
from ragfusion.config import Config

config = Config()
config.embedding.provider = "sentence-transformers"
config.save("./my_config.json")
```

## Advanced Usage

### Custom Embeddings

```python
from ragfusion.common.embeddings import CustomEmbedding

def my_embed_function(texts):
    # Your embedding logic
    return embeddings

custom_embedding = CustomEmbedding(
    embed_fn=my_embed_function,
    dimension=768
)

collection.set_embedding_model(custom_embedding)
```

### Evaluation

```python
metrics = collection.evaluate_retrieval(
    test_queries=["query1", "query2"],
    ground_truth=[["doc1"], ["doc2"]],
    metrics=["precision", "recall", "f1"]
)
```

## Architecture

RAGFusion uses a wrapper/enhancement pattern rather than abstraction:

```
Your Code
    ↓
RAGFusion Enhanced Interface
    ↓
Native Vector Store (ChromaDB/FAISS)
```

This means:
- No performance overhead
- No feature limitations
- Easy debugging (you're using the real store)
- Gradual adoption (start with native, add RAG features as needed)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

