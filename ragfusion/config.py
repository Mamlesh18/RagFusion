"""
Global configuration management for RAGFusion.
"""

import os
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    provider: Literal["openai", "cohere", "sentence-transformers", "custom"] = "sentence-transformers"
    model_name: str = "all-MiniLM-L6-v2"
    api_key: Optional[str] = None
    dimension: Optional[int] = None
    batch_size: int = 32
    device: str = "cpu"  # cpu, cuda, mps
    
    def __post_init__(self):
        # Auto-load API keys from environment
        if self.provider == "openai" and not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.provider == "cohere" and not self.api_key:
            self.api_key = os.getenv("COHERE_API_KEY")


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    strategy: Literal["fixed", "semantic", "recursive", "sentence"] = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    separators: list = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


@dataclass
class RetrievalConfig:
    """Configuration for retrieval strategies."""
    strategy: Literal["similarity", "mmr", "hybrid", "rerank"] = "similarity"
    top_k: int = 5
    score_threshold: Optional[float] = None
    diversity_factor: float = 0.5  # For MMR
    rerank_model: Optional[str] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration class for RAGFusion."""
    
    # Component configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # Global settings
    verbose: bool = False
    log_level: str = "INFO"
    cache_embeddings: bool = True
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".ragfusion" / "cache")
    
    # Performance settings
    num_workers: int = 4
    enable_progress_bar: bool = True
    
    def save(self, path: Optional[Path] = None):
        """Save configuration to JSON file."""
        if path is None:
            path = Path.home() / ".ragfusion" / "config.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            "embedding": {
                "provider": self.embedding.provider,
                "model_name": self.embedding.model_name,
                "dimension": self.embedding.dimension,
                "batch_size": self.embedding.batch_size,
                "device": self.embedding.device,
            },
            "chunking": {
                "strategy": self.chunking.strategy,
                "chunk_size": self.chunking.chunk_size,
                "chunk_overlap": self.chunking.chunk_overlap,
                "min_chunk_size": self.chunking.min_chunk_size,
                "max_chunk_size": self.chunking.max_chunk_size,
                "separators": self.chunking.separators,
            },
            "retrieval": {
                "strategy": self.retrieval.strategy,
                "top_k": self.retrieval.top_k,
                "score_threshold": self.retrieval.score_threshold,
                "diversity_factor": self.retrieval.diversity_factor,
                "rerank_model": self.retrieval.rerank_model,
                "metadata_filters": self.retrieval.metadata_filters,
            },
            "verbose": self.verbose,
            "log_level": self.log_level,
            "cache_embeddings": self.cache_embeddings,
            "cache_dir": str(self.cache_dir),
            "num_workers": self.num_workers,
            "enable_progress_bar": self.enable_progress_bar,
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration from JSON file."""
        if path is None:
            path = Path.home() / ".ragfusion" / "config.json"
        
        if not path.exists():
            return cls()
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        embedding_config = EmbeddingConfig(**config_dict.get("embedding", {}))
        chunking_config = ChunkingConfig(**config_dict.get("chunking", {}))
        retrieval_config = RetrievalConfig(**config_dict.get("retrieval", {}))
        
        return cls(
            embedding=embedding_config,
            chunking=chunking_config,
            retrieval=retrieval_config,
            verbose=config_dict.get("verbose", False),
            log_level=config_dict.get("log_level", "INFO"),
            cache_embeddings=config_dict.get("cache_embeddings", True),
            cache_dir=Path(config_dict.get("cache_dir", Path.home() / ".ragfusion" / "cache")),
            num_workers=config_dict.get("num_workers", 4),
            enable_progress_bar=config_dict.get("enable_progress_bar", True),
        )


# Global configuration instance
_global_config = Config()


def get_config() -> Config:
    """Get the global configuration."""
    return _global_config


def set_config(config: Config):
    """Set the global configuration."""
    global _global_config
    _global_config = config


def configure(**kwargs):
    """
    Configure global settings using keyword arguments.
    
    Examples:
        configure(verbose=True)
        configure(embedding__model_name="text-embedding-ada-002")
    """
    config = get_config()
    
    for key, value in kwargs.items():
        if "__" in key:
            # Handle nested configuration
            parts = key.split("__")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            # Handle top-level configuration
            setattr(config, key, value)