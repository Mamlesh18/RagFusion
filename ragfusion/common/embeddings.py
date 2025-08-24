"""
Embedding models management for RAGFusion.
"""

import numpy as np
from typing import List, Optional, Union, Callable
from functools import lru_cache
import hashlib
import pickle
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

from ragfusion.common.base import BaseEmbedding
from ragfusion.config import get_config


class EmbeddingCache:
    """Simple cache for embeddings."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or get_config().cache_dir / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        key = self._get_cache_key(text, model_name)
        
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            embedding = np.load(cache_file)
            self.memory_cache[key] = embedding
            return embedding
        
        return None
    
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """Store embedding in cache."""
        key = self._get_cache_key(text, model_name)
        
        # Store in memory
        self.memory_cache[key] = embedding
        
        # Store on disk
        cache_file = self.cache_dir / f"{key}.npy"
        np.save(cache_file, embedding)
    
    def clear(self):
        """Clear all cached embeddings."""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()


class SentenceTransformerEmbedding(BaseEmbedding):
    """Sentence Transformer embedding model."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None,
        **kwargs
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install ragfusion[embeddings]"
            )
        
        self.model_name = model_name
        self.device = device or get_config().embedding.device
        self.model = SentenceTransformer(model_name, device=self.device, **kwargs)
        self.cache = cache if cache is not None else (
            EmbeddingCache() if get_config().cache_embeddings else None
        )
        self._dimension = None
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of documents."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Compute uncached embeddings
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                show_progress_bar=get_config().enable_progress_bar,
                batch_size=get_config().embedding.batch_size,
            )
            
            # Cache new embeddings
            if self.cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.cache.set(text, self.model_name, embedding)
            
            # Combine with cached embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, embedding))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.vstack([e[1] for e in embeddings])
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query."""
        # Check cache
        if self.cache:
            cached = self.cache.get(text, self.model_name)
            if cached is not None:
                return cached
        
        # Compute embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        
        # Cache embedding
        if self.cache:
            self.cache.set(text, self.model_name, embedding)
        
        return embedding
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        if self._dimension is None:
            # Generate a sample embedding to get dimension
            sample = self.embed_query("sample text")
            self._dimension = sample.shape[0]
        return self._dimension


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding model."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None,
        **kwargs
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install with: pip install ragfusion[embeddings]"
            )
        
        self.model_name = model_name
        self.api_key = api_key or get_config().embedding.api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key, **kwargs)
        self.cache = cache if cache is not None else (
            EmbeddingCache() if get_config().cache_embeddings else None
        )
        
        # Model dimensions
        self._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _embed_with_retry(self, texts: List[str]) -> List[np.ndarray]:
        """Embed texts with retry logic."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [np.array(data.embedding) for data in response.data]
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of documents."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Compute uncached embeddings in batches
        if uncached_texts:
            batch_size = 100  # OpenAI limit
            new_embeddings = []
            
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                batch_embeddings = self._embed_with_retry(batch)
                new_embeddings.extend(batch_embeddings)
            
            # Cache new embeddings
            if self.cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.cache.set(text, self.model_name, embedding)
            
            # Combine with cached embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, embedding))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.vstack([e[1] for e in embeddings])
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query."""
        # Check cache
        if self.cache:
            cached = self.cache.get(text, self.model_name)
            if cached is not None:
                return cached
        
        # Compute embedding
        embedding = self._embed_with_retry([text])[0]
        
        # Cache embedding
        if self.cache:
            self.cache.set(text, self.model_name, embedding)
        
        return embedding
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._dimensions.get(self.model_name, 1536)


class CustomEmbedding(BaseEmbedding):
    """Custom embedding model using a user-provided function."""
    
    def __init__(
        self,
        embed_fn: Callable[[List[str]], np.ndarray],
        dimension: int,
        cache: Optional[EmbeddingCache] = None,
        model_name: str = "custom",
    ):
        self.embed_fn = embed_fn
        self._dimension = dimension
        self.model_name = model_name
        self.cache = cache if cache is not None else (
            EmbeddingCache() if get_config().cache_embeddings else None
        )
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of documents."""
        if self.cache:
            embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            if uncached_texts:
                new_embeddings = self.embed_fn(uncached_texts)
                
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.cache.set(text, self.model_name, embedding)
                
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    embeddings.append((idx, embedding))
            
            embeddings.sort(key=lambda x: x[0])
            return np.vstack([e[1] for e in embeddings])
        else:
            return self.embed_fn(texts)
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query."""
        if self.cache:
            cached = self.cache.get(text, self.model_name)
            if cached is not None:
                return cached
        
        embedding = self.embed_fn([text])[0]
        
        if self.cache:
            self.cache.set(text, self.model_name, embedding)
        
        return embedding
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._dimension


def get_embedding_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseEmbedding:
    """
    Factory function to get an embedding model.
    
    Args:
        provider: Embedding provider (openai, sentence-transformers, custom)
        model_name: Model name
        **kwargs: Additional arguments for the embedding model
    
    Returns:
        Embedding model instance
    """
    config = get_config()
    provider = provider or config.embedding.provider
    model_name = model_name or config.embedding.model_name
    
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedding(model_name=model_name, **kwargs)
    elif provider == "openai":
        return OpenAIEmbedding(model_name=model_name, **kwargs)
    elif provider == "custom":
        if "embed_fn" not in kwargs:
            raise ValueError("embed_fn must be provided for custom embeddings")
        return CustomEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")