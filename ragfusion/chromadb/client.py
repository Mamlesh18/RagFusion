# ============================================================================
# ragfusion/chromadb/client.py
# ============================================================================

"""
Enhanced ChromaDB client with RAG features.
"""

import chromadb
from chromadb import Collection
from typing import List, Dict, Any, Optional, Union, Sequence
import numpy as np
import uuid
from pathlib import Path

from ragfusion.common.base import (
    BaseVectorStore, Document, DocumentChunk, SearchResult
)
from ragfusion.common.embeddings import get_embedding_model, BaseEmbedding
from ragfusion.common.chunking import get_chunker
from ragfusion.config import get_config


class EnhancedCollection(Collection, BaseVectorStore):
    """
    Enhanced ChromaDB Collection with RAG features.
    Inherits from both ChromaDB Collection and RAGFusion BaseVectorStore.
    """
    
    def __init__(self, collection: Collection, embedding_model: Optional[BaseEmbedding] = None):
        """
        Initialize enhanced collection.
        
        Args:
            collection: Original ChromaDB collection
            embedding_model: Optional embedding model for RAG features
        """
        # Copy all attributes from the original collection
        self.__dict__.update(collection.__dict__)
        self._original_collection = collection
        self._embedding_model = embedding_model or get_embedding_model()
        self._chunker = get_chunker()
        self._config = get_config()
    
    # ========== Native ChromaDB methods (inherited) ==========
    # All original ChromaDB methods work as expected
    
    # ========== RAG-enhanced methods ==========
    
    def add_documents(
        self,
        documents: List[Union[Document, str, Dict[str, Any]]],
        embeddings: Optional[np.ndarray] = None,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        chunk: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Enhanced document addition with automatic chunking and embedding.
        
        Args:
            documents: Documents to add
            embeddings: Pre-computed embeddings (optional)
            ids: Document IDs
            metadata: Metadata for each document
            chunk: Whether to chunk documents
            **kwargs: Additional ChromaDB parameters
        
        Returns:
            List of document IDs
        """
        # Convert to Document objects
        docs = []
        for i, doc in enumerate(documents):
            if isinstance(doc, Document):
                docs.append(doc)
            elif isinstance(doc, str):
                doc_id = ids[i] if ids else str(uuid.uuid4())
                docs.append(Document(
                    id=doc_id,
                    content=doc,
                    metadata=metadata[i] if metadata else {}
                ))
            elif isinstance(doc, dict):
                docs.append(Document(**doc))
            else:
                raise ValueError(f"Invalid document type: {type(doc)}")
        
        # Chunk documents if requested
        if chunk and self._config.chunking.chunk_size > 0:
            chunks = self._chunker.chunk_documents(docs)
            
            # Prepare data for ChromaDB
            chunk_texts = [chunk.content for chunk in chunks]
            chunk_ids = [chunk.id for chunk in chunks]
            chunk_metadata = [chunk.metadata for chunk in chunks]
            
            # Generate embeddings if not provided
            if embeddings is None:
                chunk_embeddings = self._embedding_model.embed_documents(chunk_texts)
            else:
                chunk_embeddings = embeddings
            
            # Add to ChromaDB
            self.add(
                documents=chunk_texts,
                embeddings=chunk_embeddings.tolist(),
                ids=chunk_ids,
                metadatas=chunk_metadata,
                **kwargs
            )
            
            return [doc.id for doc in docs]
        else:
            # Add documents without chunking
            texts = [doc.content for doc in docs]
            doc_ids = [doc.id for doc in docs]
            doc_metadata = [doc.metadata for doc in docs]
            
            # Generate embeddings if not provided
            if embeddings is None:
                doc_embeddings = self._embedding_model.embed_documents(texts)
            else:
                doc_embeddings = embeddings
            
            # Add to ChromaDB
            self.add(
                documents=texts,
                embeddings=doc_embeddings.tolist(),
                ids=doc_ids,
                metadatas=doc_metadata,
                **kwargs
            )
            
            return doc_ids
    
    def rag_search(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        strategy: str = None,
        rerank: bool = False,
        include_context: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Enhanced search with RAG features.
        
        Args:
            query: Query string or embedding
            k: Number of results
            strategy: Retrieval strategy (similarity, mmr, hybrid)
            rerank: Whether to rerank results
            include_context: Include surrounding context
            filters: Metadata filters
            **kwargs: Additional parameters
        
        Returns:
            List of search results
        """
        strategy = strategy or self._config.retrieval.strategy
        
        # Generate query embedding if needed
        if isinstance(query, str):
            query_embedding = self._embedding_model.embed_query(query)
            query_text = query
        else:
            query_embedding = query
            query_text = None
        
        # Perform search based on strategy
        if strategy == "mmr":
            # Maximum Marginal Relevance search
            results = self.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k * 3,  # Fetch more for MMR
                where=filters,
                **kwargs
            )
            # TODO: Implement MMR selection
            # For now, just take top k
            results['ids'] = [results['ids'][0][:k]]
            results['documents'] = [results['documents'][0][:k]]
            results['metadatas'] = [results['metadatas'][0][:k]]
            results['distances'] = [results['distances'][0][:k]]
        else:
            # Standard similarity search
            results = self.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=filters,
                **kwargs
            )
        
        # Convert to SearchResult objects
        search_results = []
        for i in range(len(results['ids'][0])):
            result = SearchResult(
                document_id=results['metadatas'][0][i].get('document_id', results['ids'][0][i]),
                chunk_id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=1.0 - results['distances'][0][i],  # Convert distance to similarity
            )
            search_results.append(result)
        
        # Rerank if requested
        if rerank and query_text:
            from ragfusion.common.reranking import get_reranker
            reranker = get_reranker()
            search_results = reranker.rerank(query_text, search_results, top_k=k)
        
        return search_results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Hybrid search combining keyword and semantic search.
        
        Args:
            query: Query string
            k: Number of results
            alpha: Weight for semantic search (0-1)
            filters: Metadata filters
            **kwargs: Additional parameters
        
        Returns:
            List of search results
        """
        # Semantic search
        semantic_results = self.rag_search(
            query=query,
            k=k * 2,
            strategy="similarity",
            filters=filters,
            **kwargs
        )
        
        # Keyword search (using ChromaDB's where clause)
        # This is a simplified version - you might want to implement
        # more sophisticated keyword matching
        keyword_results = self.query(
            query_texts=[query],
            n_results=k * 2,
            where=filters,
            **kwargs
        )
        
        # Combine and rerank results
        # TODO: Implement proper hybrid ranking
        return semantic_results[:k]
    
    def evaluate_retrieval(
        self,
        test_queries: List[str],
        ground_truth: List[List[str]],
        k: int = 5,
        metrics: List[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            test_queries: Test queries
            ground_truth: Expected document IDs for each query
            k: Number of results to retrieve
            metrics: Metrics to compute
        
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ["precision", "recall", "f1"]
        
        results = {metric: [] for metric in metrics}
        
        for query, expected in zip(test_queries, ground_truth):
            retrieved = self.rag_search(query, k=k)
            retrieved_ids = [r.document_id for r in retrieved]
            
            # Calculate metrics
            if "precision" in metrics:
                precision = len(set(retrieved_ids) & set(expected)) / len(retrieved_ids)
                results["precision"].append(precision)
            
            if "recall" in metrics:
                recall = len(set(retrieved_ids) & set(expected)) / len(expected)
                results["recall"].append(recall)
            
            if "f1" in metrics:
                if "precision" in metrics and "recall" in metrics:
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0
                    results["f1"].append(f1)
        
        # Average metrics
        return {metric: np.mean(scores) for metric, scores in results.items()}
    
    # ========== BaseVectorStore implementation ==========
    
    def search(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Implement BaseVectorStore search interface."""
        return self.rag_search(query, k=k, filters=filters, **kwargs)
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> int:
        """Implement BaseVectorStore delete interface."""
        if ids:
            self._original_collection.delete(ids=ids, **kwargs)
            return len(ids)
        elif filters:
            # ChromaDB doesn't support delete by filter directly
            # Need to query first then delete
            results = self.get(where=filters)
            if results['ids']:
                self._original_collection.delete(ids=results['ids'])
                return len(results['ids'])
        return 0
    
    def update(
        self,
        ids: List[str],
        documents: Optional[List[Union[Document, str]]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """Implement BaseVectorStore update interface."""
        update_dict = {"ids": ids}
        
        if documents:
            texts = []
            for doc in documents:
                if isinstance(doc, Document):
                    texts.append(doc.content)
                else:
                    texts.append(doc)
            update_dict["documents"] = texts
            
            if embeddings is None:
                embeddings = self._embedding_model.embed_documents(texts)
        
        if embeddings is not None:
            update_dict["embeddings"] = embeddings.tolist()
        
        if metadata:
            update_dict["metadatas"] = metadata
        
        self._original_collection.update(**update_dict, **kwargs)
        return ids
    
    def count(
        self,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> int:
        """Implement BaseVectorStore count interface."""
        return self._original_collection.count()
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        # ChromaDB doesn't have a clear method, so delete all
        all_ids = self._original_collection.get()['ids']
        if all_ids:
            self._original_collection.delete(ids=all_ids)


class EnhancedChromaClient:
    """
    Enhanced ChromaDB Client with RAG features.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced client."""
        self._client = chromadb.Client(*args, **kwargs)
        self._embedding_model = None
        self._config = get_config()
    
    def __getattr__(self, name):
        """Delegate unknown methods to the original client."""
        return getattr(self._client, name)
    
    def create_collection(
        self,
        name: str,
        embedding_function: Optional[Any] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        **kwargs
    ) -> EnhancedCollection:
        """
        Create an enhanced collection.
        
        Args:
            name: Collection name
            embedding_function: ChromaDB embedding function
            embedding_model: RAGFusion embedding model
            **kwargs: Additional ChromaDB parameters
        
        Returns:
            Enhanced collection
        """
        # Create original collection
        collection = self._client.create_collection(
            name=name,
            embedding_function=embedding_function,
            **kwargs
        )
        
        # Wrap with enhancements
        return EnhancedCollection(collection, embedding_model or self._embedding_model)
    
    def get_or_create_collection(
        self,
        name: str,
        embedding_function: Optional[Any] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        **kwargs
    ) -> EnhancedCollection:
        """Get or create an enhanced collection."""
        collection = self._client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
            **kwargs
        )
        return EnhancedCollection(collection, embedding_model or self._embedding_model)
    
    def get_collection(
        self,
        name: str,
        embedding_function: Optional[Any] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        **kwargs
    ) -> EnhancedCollection:
        """Get an enhanced collection."""
        collection = self._client.get_collection(
            name=name,
            embedding_function=embedding_function,
            **kwargs
        )
        return EnhancedCollection(collection, embedding_model or self._embedding_model)
    
    def set_embedding_model(self, model: BaseEmbedding):
        """Set default embedding model for all collections."""
        self._embedding_model = model
    
    def migrate_to_faiss(
        self,
        collection_name: str,
        faiss_index_path: str,
        batch_size: int = 1000,
    ):
        """
        Migrate a collection to FAISS.
        
        Args:
            collection_name: Name of collection to migrate
            faiss_index_path: Path to save FAISS index
            batch_size: Batch size for migration
        """
        from ragfusion import faiss
        
        # Get collection
        collection = self.get_collection(collection_name)
        
        # Get all data
        data = collection.get(include=["documents", "embeddings", "metadatas"])
        
        # Create FAISS index
        dimension = len(data["embeddings"][0])
        faiss_index = faiss.Index(dimension=dimension)
        
        # Add data in batches
        for i in range(0, len(data["ids"]), batch_size):
            batch_docs = data["documents"][i:i+batch_size]
            batch_embeddings = np.array(data["embeddings"][i:i+batch_size])
            batch_metadata = data["metadatas"][i:i+batch_size]
            batch_ids = data["ids"][i:i+batch_size]
            
            faiss_index.add_documents(
                documents=batch_docs,
                embeddings=batch_embeddings,
                ids=batch_ids,
                metadata=batch_metadata,
            )
        
        # Save FAISS index
        faiss_index.save(faiss_index_path)
        
        print(f"Migration complete. {len(data['ids'])} documents migrated to FAISS.")