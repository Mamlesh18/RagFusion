"""
ragfusion/faiss/__init__.py
Enhanced FAISS with document storage and RAG features.
"""

try:
    import faiss
    from faiss import *  # Re-export everything from FAISS
except ImportError:
    raise ImportError(
        "FAISS not installed. Install with: pip install ragfusion[faiss] or pip install ragfusion[faiss-gpu]"
    )

from ragfusion.faiss.index import EnhancedFAISSIndex
from ragfusion.faiss.document_store import DocumentStore

# Make Index point to our enhanced version
Index = EnhancedFAISSIndex

# Make enhanced versions available
__all__ = ['EnhancedFAISSIndex', 'DocumentStore', 'Index'] + dir(faiss)