"""
Document storage for FAISS (since FAISS only stores vectors).
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Any
from ragfusion.common.base import Document


class DocumentStore:
    """
    Simple document storage for FAISS.
    In production, you might want to use a database.
    """
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, str] = {}  # chunk_id -> content
    
    def add_document(self, document: Document):
        """Add a document to the store."""
        self.documents[document.id] = document
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str):
        """Delete a document."""
        if doc_id in self.documents:
            del self.documents[doc_id]
    
    def add_chunk(self, chunk_id: str, content: str):
        """Add a chunk to the store."""
        self.chunks[chunk_id] = content
    
    def get_chunk(self, chunk_id: str) -> Optional[str]:
        """Get chunk content by ID."""
        return self.chunks.get(chunk_id)
    
    def clear(self):
        """Clear all documents."""
        self.documents.clear()
        self.chunks.clear()
    
    def save(self, path: Path):
        """Save document store to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "chunks": self.chunks,
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> "DocumentStore":
        """Load document store from disk."""
        store = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
            store.documents = data["documents"]
            store.chunks = data.get("chunks", {})
        return store