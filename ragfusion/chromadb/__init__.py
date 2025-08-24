"""
ragfusion/chromadb/__init__.py
Enhanced ChromaDB with RAG features.
"""

try:
    import chromadb
    from chromadb import *  # Re-export everything from ChromaDB
except ImportError:
    raise ImportError(
        "ChromaDB not installed. Install with: pip install ragfusion[chromadb]"
    )

from ragfusion.chromadb.client import EnhancedChromaClient, EnhancedCollection

# Replace the default Client and Collection
Client = EnhancedChromaClient
PersistentClient = EnhancedChromaClient

# Make enhanced versions available
__all__ = chromadb.__all__ + ['EnhancedChromaClient', 'EnhancedCollection']