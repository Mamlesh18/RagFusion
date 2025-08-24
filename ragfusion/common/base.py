"""
Base classes and interfaces for RAGFusion.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum


class DocumentStatus(Enum):
    """Status of a document in the store."""
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


@dataclass
class Document:
    """
    Represents a document in the RAG system.
    """
    id: str
    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None
    chunks: Optional[List["DocumentChunk"]] = None
    status: DocumentStatus = DocumentStatus.PENDING
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "status": self.status.value,
        }


@dataclass
class DocumentChunk:
    """
    Represents a chunk of a document.
    """
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    chunk_index: Optional[int] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Ensure document_id is in metadata
        self.metadata["document_id"] = self.document_id
        if self.chunk_index is not None:
            self.metadata["chunk_index"] = self.chunk_index


@dataclass
class SearchResult:
    """
    Represents a search result from the vector store.
    """
    document_id: str
    chunk_id: Optional[str]
    content: str
    metadata: Dict[str, Any]
    score: float
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
        }


class BaseVectorStore(ABC):
    """
    Abstract base class for all vector stores in RAGFusion.
    """
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Union[Document, str, Dict[str, Any]]],
        embeddings: Optional[np.ndarray] = None,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            embeddings: Pre-computed embeddings (optional)
            ids: Document IDs (optional, will be generated if not provided)
            metadata: Additional metadata for each document
            **kwargs: Store-specific parameters
        
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Query string or embedding vector
            k: Number of results to return
            filters: Metadata filters
            **kwargs: Store-specific parameters
        
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> int:
        """
        Delete documents from the store.
        
        Args:
            ids: Document IDs to delete
            filters: Delete documents matching these filters
            **kwargs: Store-specific parameters
        
        Returns:
            Number of documents deleted
        """
        pass
    
    @abstractmethod
    def update(
        self,
        ids: List[str],
        documents: Optional[List[Union[Document, str]]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """
        Update existing documents.
        
        Args:
            ids: Document IDs to update
            documents: New document content
            embeddings: New embeddings
            metadata: New metadata
            **kwargs: Store-specific parameters
        
        Returns:
            List of updated document IDs
        """
        pass
    
    @abstractmethod
    def get(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve documents by ID or filters.
        
        Args:
            ids: Document IDs to retrieve
            filters: Metadata filters
            **kwargs: Store-specific parameters
        
        Returns:
            List of documents
        """
        pass
    
    @abstractmethod
    def count(
        self,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> int:
        """
        Count documents in the store.
        
        Args:
            filters: Optional metadata filters
            **kwargs: Store-specific parameters
        
        Returns:
            Number of documents
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the store."""
        pass
    
    # RAG-specific methods (with default implementations)
    
    def similarity_search_with_score(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Search and return documents with similarity scores.
        """
        results = self.search(query, k=k, filters=filters, **kwargs)
        documents = []
        for result in results:
            doc = Document(
                id=result.document_id,
                content=result.content,
                metadata=result.metadata,
                embedding=result.embedding
            )
            documents.append((doc, result.score))
        return documents
    
    def max_marginal_relevance_search(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform Maximum Marginal Relevance (MMR) search.
        """
        # Default implementation: fall back to regular search
        # Subclasses can override with actual MMR implementation
        results = self.search(query, k=k, filters=filters, **kwargs)
        return [
            Document(
                id=r.document_id,
                content=r.content,
                metadata=r.metadata,
                embedding=r.embedding
            )
            for r in results
        ]


class BaseEmbedding(ABC):
    """
    Abstract base class for embedding models.
    """
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass


class BaseChunker(ABC):
    """
    Abstract base class for text chunking strategies.
    """
    
    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text into chunks."""
        pass
    
    @abstractmethod
    def chunk_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[DocumentChunk]:
        """Split documents into chunks."""
        pass


class BaseReranker(ABC):
    """
    Abstract base class for result reranking.
    """
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Rerank search results based on the query."""
        pass