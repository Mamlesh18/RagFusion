"""
RAGFusion - A unified RAG framework for seamless vector store operations.

Usage:
    from ragfusion import chromadb
    client = chromadb.Client()
    
    from ragfusion import faiss
    index = faiss.Index(dimension=768)
"""

from ragfusion.version import __version__
from ragfusion.config import Config, get_config, set_config, configure

# Lazy loading for vector stores to avoid unnecessary imports
_AVAILABLE_STORES = {
    'chromadb': 'ragfusion.chromadb',
    'faiss': 'ragfusion.faiss',
}

_LOADED_MODULES = {}


def __getattr__(name):
    """
    Lazy loading of vector store modules.
    This allows users to only import what they need.
    """
    if name in _AVAILABLE_STORES:
        if name not in _LOADED_MODULES:
            import importlib
            try:
                module = importlib.import_module(_AVAILABLE_STORES[name])
                _LOADED_MODULES[name] = module
            except ImportError as e:
                raise ImportError(
                    f"Cannot import {name}. Please install it with: "
                    f"pip install ragfusion[{name}]"
                ) from e
        return _LOADED_MODULES[name]
    
    raise AttributeError(f"module 'ragfusion' has no attribute '{name}'")


def list_available_stores():
    """List all available vector stores."""
    return list(_AVAILABLE_STORES.keys())


__all__ = [
    '__version__',
    'Config',
    'get_config',
    'set_config',
    'configure',
    'list_available_stores',
]