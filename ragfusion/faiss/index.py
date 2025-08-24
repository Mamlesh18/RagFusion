# ============================================================================
# ragfusion/faiss/index.py
# ============================================================================

"""
Enhanced FAISS index with document storage and RAG features.
"""

import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import uuid
import pickle
from pathlib import Path
import json

from ragfusion.common.base import (
    BaseVectorStore, Document, DocumentChunk, SearchResult
)
from ragfusion.common.embeddings import get_embedding_model, BaseEmbedding
from ragfusion.common.chunking import get_chunker
from ragfusion.config import get_config
from ragfusion.faiss.document_store import DocumentStore


class EnhancedFAISSIndex(BaseVectorStore):
    """
    Enhanced FAISS Index with document storage and RAG features.
    FAISS doesn't store documents natively, so we add this capability.
    """
    
    def __init__(
        self,
        dimension: Optional[int] = None,
        index_type: str = "Flat",
        metric: str = "L2",
        index: Optional[faiss.Index] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        document_store: Optional[DocumentStore] = None,
        **kwargs
    ):
        """
        Initialize enhanced FAISS index.
        
        Args:
            dimension: Vector dimension
            index_type: Type of FAISS index (Flat, IVF, HNSW, etc.)
            metric: Distance metric (L2, IP for inner product)
            index: Pre-existing FAISS index
            embedding_model: Embedding model for documents
            document_store: Document storage backend
            **kwargs: Additional FAISS parameters
        """
        self._config = get_config()
        self._embedding_model = embedding_model or get_embedding_model()
        self._chunker = get_chunker()
        
        # Initialize or use existing index
        if index is not None:
            self.index = index
            self.dimension = index.d
        else:
            if dimension is None:
                # Try to get dimension from embedding model
                dimension = self._embedding_model.dimension
            self.dimension = dimension
            self.index = self._create_index(dimension, index_type, metric, **kwargs)
        
        # Document storage
        self.document_store = document_store or DocumentStore()
        
        # Metadata storage (FAISS doesn't handle metadata)
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        
        # ID mapping (FAISS uses integer IDs, we use strings)
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.next_idx = 0
    
    def _create_index(
        self,
        dimension: int,
        index_type: str,
        metric: str,
        **kwargs
    ) -> faiss.Index:
        """Create a FAISS index based on type and parameters."""
        # Set metric
        if metric == "L2":
            metric_type = faiss.METRIC_L2
        elif metric in ["IP", "inner_product", "cosine"]:
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Create index based on type
        if index_type == "Flat":
            if metric == "L2":
                index = faiss.IndexFlatL2(dimension)
            else:
                index = faiss.IndexFlatIP(dimension)
        
        elif index_type == "IVF":
            nlist = kwargs.get("nlist", 100)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric_type)
        
        elif index_type == "HNSW":
            M = kwargs.get("M", 32)
            index = faiss.IndexHNSWFlat(dimension, M, metric_type)
        
        elif index_type == "LSH":
            nbits = kwargs.get("nbits", dimension * 8)
            index = faiss.IndexLSH(dimension, nbits)
        
        elif index_type == "IVF_PQ":
            nlist = kwargs.get("nlist", 100)
            m = kwargs.get("m", 8)
            nbits = kwargs.get("nbits", 8)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add ID map to support remove operations
        index = faiss.IndexIDMap(index)
        
        return index
    
    # ========== FAISS-like interface ==========
    
    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None):
        """
        Add vectors to the index (FAISS-compatible interface).
        
        Args:
            vectors: Vectors to add
            ids: Optional IDs for the vectors
        """
        if ids is None:
            ids = list(range(self.next_idx, self.next_idx + len(vectors)))
            self.next_idx += len(vectors)
        
        # Normalize for cosine similarity if needed
        if hasattr(self.index, 'metric_type') and self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(vectors)
        
        self.index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
    
    def _faiss_search(
        self,
        query_vectors: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors (FAISS-compatible interface).
        
        Args:
            query_vectors: Query vectors
            k: Number of neighbors
        
        Returns:
            Distances and indices
        """
        # Normalize for cosine similarity if needed
        if hasattr(self.index, 'metric_type') and self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            query_vectors = query_vectors.copy()
            faiss.normalize_L2(query_vectors)
        
        return self.index.search(query_vectors, k)
    
    def train(self, vectors: np.ndarray):
        """Train the index if needed (for IVF indices)."""
        if hasattr(self.index, 'train'):
            self.index.train(vectors)
    
    def reset(self):
        """Reset the index."""
        self.index.reset()
        self.document_store.clear()
        self.metadata_store.clear()
        self.id_to_idx.clear()
        self.idx_to_id.clear()
        self.next_idx = 0
    
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
        Add documents with automatic chunking and embedding.
        
        Args:
            documents: Documents to add
            embeddings: Pre-computed embeddings
            ids: Document IDs
            metadata: Metadata for each document
            chunk: Whether to chunk documents
            **kwargs: Additional parameters
        
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
        
        # Store documents
        for doc in docs:
            self.document_store.add_document(doc)
        
        # Chunk documents if requested
        if chunk and self._config.chunking.chunk_size > 0:
            chunks = self._chunker.chunk_documents(docs)
            
            # Prepare data for FAISS
            chunk_texts = [chunk.content for chunk in chunks]
            chunk_ids = [chunk.id for chunk in chunks]
            chunk_metadata = [chunk.metadata for chunk in chunks]
            
            # Generate embeddings if not provided
            if embeddings is None:
                chunk_embeddings = self._embedding_model.embed_documents(chunk_texts)
            else:
                chunk_embeddings = embeddings
            
            # Add to FAISS
            faiss_ids = []
            for i, (chunk_id, metadata) in enumerate(zip(chunk_ids, chunk_metadata)):
                idx = self.next_idx
                self.next_idx += 1
                
                self.id_to_idx[chunk_id] = idx
                self.idx_to_id[idx] = chunk_id
                self.metadata_store[idx] = metadata
                faiss_ids.append(idx)
            
            self.add(chunk_embeddings, faiss_ids)
            
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
            
            # Add to FAISS
            faiss_ids = []
            for doc_id, metadata in zip(doc_ids, doc_metadata):
                idx = self.next_idx
                self.next_idx += 1
                
                self.id_to_idx[doc_id] = idx
                self.idx_to_id[idx] = doc_id
                self.metadata_store[idx] = metadata
                faiss_ids.append(idx)
            
            self.add(doc_embeddings, faiss_ids)
            
            return doc_ids
    
    def rag_search(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
        **kwargs
    ) -> List[SearchResult]:
        """
        Enhanced search with document retrieval.
        
        Args:
            query: Query string or embedding
            k: Number of results
            filters: Metadata filters (post-filtering)
            include_embeddings: Include embeddings in results
            **kwargs: Additional parameters
        
        Returns:
            List of search results with documents
        """
        # Generate query embedding if needed
        if isinstance(query, str):
            query_embedding = self._embedding_model.embed_query(query)
        else:
            query_embedding = query
        
        # Search in FAISS
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self._faiss_search(query_embedding, k * 3 if filters else k)
        
        # Convert to search results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for unfilled results
                continue
            
            # Get document/chunk ID and metadata
            doc_id = self.idx_to_id.get(idx)
            if doc_id is None:
                continue
            
            metadata = self.metadata_store.get(idx, {})
            
            # Apply filters
            if filters:
                match = all(
                    metadata.get(key) == value
                    for key, value in filters.items()
                )
                if not match:
                    continue
            
            # Get document content
            if "document_id" in metadata:
                # This is a chunk
                document = self.document_store.get_document(metadata["document_id"])
                content = self.document_store.get_chunk(doc_id)
                if content is None and document:
                    # Fallback to document content
                    content = document.content
            else:
                # This is a full document
                document = self.document_store.get_document(doc_id)
                content = document.content if document else ""
            
            # Create search result
            result = SearchResult(
                document_id=metadata.get("document_id", doc_id),
                chunk_id=doc_id if "document_id" in metadata else None,
                content=content,
                metadata=metadata,
                score=1.0 / (1.0 + dist),  # Convert distance to similarity
                embedding=query_embedding[0] if include_embeddings else None
            )
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def save(self, path: Union[str, Path]):
        """
        Save the index and document store to disk.
        
        Args:
            path: Directory to save the index
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save document store
        self.document_store.save(path / "documents.pkl")
        
        # Save metadata and mappings
        metadata = {
            "dimension": self.dimension,
            "metadata_store": self.metadata_store,
            "id_to_idx": self.id_to_idx,
            "idx_to_id": self.idx_to_id,
            "next_idx": self.next_idx,
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        # Save config
        config = {
            "embedding_model": self._config.embedding.model_name,
            "embedding_provider": self._config.embedding.provider,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "EnhancedFAISSIndex":
        """
        Load an index from disk.
        
        Args:
            path: Directory containing the saved index
        
        Returns:
            Loaded index
        """
        path = Path(path)
        
        # Load FAISS index
        index = faiss.read_index(str(path / "index.faiss"))
        
        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create enhanced index
        enhanced = cls(index=index)
        
        # Load document store
        enhanced.document_store = DocumentStore.load(path / "documents.pkl")
        
        # Load metadata and mappings
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        enhanced.dimension = metadata["dimension"]
        enhanced.metadata_store = metadata["metadata_store"]
        enhanced.id_to_idx = metadata["id_to_idx"]
        enhanced.idx_to_id = metadata["idx_to_id"]
        enhanced.next_idx = metadata["next_idx"]
        
        return enhanced
    
    def migrate_to_chromadb(
        self,
        client,
        collection_name: str,
        batch_size: int = 1000,
    ):
        """
        Migrate this FAISS index to ChromaDB.
        
        Args:
            client: ChromaDB client
            collection_name: Name for the new collection
            batch_size: Batch size for migration
        """
        from ragfusion import chromadb
        
        # Create collection
        collection = client.create_collection(collection_name)
        
        # Get all vectors and metadata
        vectors = []
        ids = []
        metadatas = []
        documents = []
        
        for idx, doc_id in self.idx_to_id.items():
            # Reconstruct vector
            vector = self.index.reconstruct(int(idx))
            vectors.append(vector)
            ids.append(doc_id)
            
            # Get metadata
            metadata = self.metadata_store.get(idx, {})
            metadatas.append(metadata)
            
            # Get document
            if "document_id" in metadata:
                # This is a chunk
                content = self.document_store.get_chunk(doc_id)
            else:
                # This is a document
                doc = self.document_store.get_document(doc_id)
                content = doc.content if doc else ""
            documents.append(content)
        
        # Add to ChromaDB in batches
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                embeddings=[v.tolist() for v in vectors[i:batch_end]],
                metadatas=metadatas[i:batch_end],
            )
        
        print(f"Migration complete. {len(ids)} documents migrated to ChromaDB.")
    
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
        """Delete documents from the index."""
        count = 0
        
        if ids:
            faiss_ids = []
            for doc_id in ids:
                if doc_id in self.id_to_idx:
                    idx = self.id_to_idx[doc_id]
                    faiss_ids.append(idx)
                    
                    # Clean up mappings
                    del self.id_to_idx[doc_id]
                    del self.idx_to_id[idx]
                    if idx in self.metadata_store:
                        del self.metadata_store[idx]
                    
                    # Remove from document store
                    self.document_store.delete_document(doc_id)
                    count += 1
            
            # Remove from FAISS index
            if faiss_ids and hasattr(self.index, 'remove_ids'):
                self.index.remove_ids(np.array(faiss_ids, dtype=np.int64))
        
        return count
    
    def update(
        self,
        ids: List[str],
        documents: Optional[List[Union[Document, str]]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """Update existing documents."""
        # FAISS doesn't support in-place updates, so we delete and re-add
        self.delete(ids=ids)
        return self.add_documents(
            documents=documents if documents else ids,
            embeddings=embeddings,
            ids=ids,
            metadata=metadata,
            **kwargs
        )
    
    def get(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents by ID."""
        if ids:
            return [
                self.document_store.get_document(doc_id)
                for doc_id in ids
                if self.document_store.get_document(doc_id)
            ]
        return []
    
    def count(
        self,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> int:
        """Count documents in the index."""
        if filters:
            # Count with filters
            count = 0
            for idx, metadata in self.metadata_store.items():
                match = all(
                    metadata.get(key) == value
                    for key, value in filters.items()
                )
                if match:
                    count += 1
            return count
        else:
            return self.index.ntotal
    
    def clear(self) -> None:
        """Clear all documents from the index."""
        self.reset()