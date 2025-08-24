"""
Text chunking strategies for RAGFusion.
"""

import re
from typing import List, Optional, Any, Dict
from abc import ABC, abstractmethod
import uuid

from ragfusion.common.base import BaseChunker, Document, DocumentChunk
from ragfusion.config import get_config


class RecursiveTextChunker(BaseChunker):
    """
    Recursively split text using multiple separators.
    This is the most versatile chunking strategy.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text into chunks."""
        chunks = self._split_text_recursive(text, self.separators)
        return self._merge_chunks(chunks)
    
    def chunk_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[DocumentChunk]:
        """Split documents into chunks."""
        all_chunks = []
        
        for doc in documents:
            text_chunks = self.chunk_text(doc.content)
            
            # Create DocumentChunk objects
            for i, chunk_text in enumerate(text_chunks):
                # Find position in original text
                start_pos = doc.content.find(chunk_text)
                end_pos = start_pos + len(chunk_text) if start_pos != -1 else -1
                
                chunk = DocumentChunk(
                    id=f"{doc.id}_chunk_{i}",
                    document_id=doc.id,
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                    },
                    start_char=start_pos,
                    end_char=end_pos,
                    chunk_index=i,
                )
                all_chunks.append(chunk)
        
        return all_chunks
    
    def _split_text_recursive(
        self,
        text: str,
        separators: List[str]
    ) -> List[str]:
        """Recursively split text using separators."""
        if not separators:
            return [text]
        
        separator = separators[0]
        
        if separator == "":
            # Split by character
            return list(text)
        
        splits = text.split(separator)
        
        # If we can't split or the splits are good sizes, return them
        if len(splits) == 1:
            # Try next separator
            return self._split_text_recursive(text, separators[1:])
        
        # Check if splits are reasonable sizes
        good_splits = []
        for split in splits:
            if len(split) > self.max_chunk_size:
                # Too big, need to split further
                sub_splits = self._split_text_recursive(split, separators[1:])
                good_splits.extend(sub_splits)
            elif split.strip():  # Not empty
                good_splits.append(split)
        
        return good_splits
    
    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small chunks and handle overlaps."""
        if not chunks:
            return []
        
        merged = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            # Check if we should merge with current chunk
            if len(current) + len(chunk) <= self.chunk_size:
                current += " " + chunk
            else:
                # Save current chunk
                if len(current) >= self.min_chunk_size:
                    merged.append(current.strip())
                
                # Handle overlap
                if self.chunk_overlap > 0 and merged:
                    # Take end of previous chunk as overlap
                    overlap_text = current[-self.chunk_overlap:] if len(current) > self.chunk_overlap else current
                    current = overlap_text + " " + chunk
                else:
                    current = chunk
        
        # Don't forget the last chunk
        if current and len(current) >= self.min_chunk_size:
            merged.append(current.strip())
        
        return merged


class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed-size chunks.
    Simple but effective for uniform text.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to end at a word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > self.chunk_size * 0.8:  # Within 20% of chunk size
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            
            # Move start position (accounting for overlap)
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
        
        return [c for c in chunks if c]  # Filter empty chunks
    
    def chunk_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[DocumentChunk]:
        """Split documents into fixed-size chunks."""
        all_chunks = []
        
        for doc in documents:
            text_chunks = self.chunk_text(doc.content)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    id=f"{doc.id}_chunk_{i}",
                    document_id=doc.id,
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                    },
                    chunk_index=i,
                )
                all_chunks.append(chunk)
        
        return all_chunks


class SentenceChunker(BaseChunker):
    """
    Split text by sentences.
    Good for maintaining semantic coherence.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 1,  # Number of sentences to overlap
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text by sentences."""
        # Simple sentence splitting (you might want to use spaCy or NLTK for better results)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Keep last n sentences for overlap
                    current_chunk = current_chunk[-self.chunk_overlap:]
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[DocumentChunk]:
        """Split documents by sentences."""
        all_chunks = []
        
        for doc in documents:
            text_chunks = self.chunk_text(doc.content)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    id=f"{doc.id}_chunk_{i}",
                    document_id=doc.id,
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                    },
                    chunk_index=i,
                )
                all_chunks.append(chunk)
        
        return all_chunks


class SemanticChunker(BaseChunker):
    """
    Split text based on semantic similarity.
    Uses embeddings to find natural breakpoints.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        similarity_threshold: float = 0.5,
        embedding_model: Optional[Any] = None,
    ):
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        
        if self.embedding_model is None:
            from ragfusion.common.embeddings import get_embedding_model
            self.embedding_model = get_embedding_model()
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text based on semantic similarity."""
        # First, split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return []
        
        # Embed all sentences
        embeddings = self.embedding_model.embed_documents(sentences)
        
        # Find semantic breakpoints
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = self._cosine_similarity(
                embeddings[i-1],
                embeddings[i]
            )
            
            sentence_size = len(sentences[i])
            
            # Decide whether to continue chunk or start new one
            if (similarity < self.similarity_threshold or 
                current_size + sentence_size > self.chunk_size):
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_size = sentence_size
            else:
                # Continue current chunk
                current_chunk.append(sentences[i])
                current_size += sentence_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[DocumentChunk]:
        """Split documents based on semantic similarity."""
        all_chunks = []
        
        for doc in documents:
            text_chunks = self.chunk_text(doc.content)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    id=f"{doc.id}_chunk_{i}",
                    document_id=doc.id,
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "chunking_method": "semantic",
                    },
                    chunk_index=i,
                )
                all_chunks.append(chunk)
        
        return all_chunks
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)


def get_chunker(
    strategy: Optional[str] = None,
    **kwargs
) -> BaseChunker:
    """
    Factory function to get a text chunker.
    
    Args:
        strategy: Chunking strategy (recursive, fixed, sentence, semantic)
        **kwargs: Additional arguments for the chunker
    
    Returns:
        Text chunker instance
    """
    config = get_config()
    strategy = strategy or config.chunking.strategy
    
    # Override with config values if not provided
    if "chunk_size" not in kwargs:
        kwargs["chunk_size"] = config.chunking.chunk_size
    if "chunk_overlap" not in kwargs:
        kwargs["chunk_overlap"] = config.chunking.chunk_overlap
    
    if strategy == "recursive":
        return RecursiveTextChunker(**kwargs)
    elif strategy == "fixed":
        return FixedSizeChunker(**kwargs)
    elif strategy == "sentence":
        return SentenceChunker(**kwargs)
    elif strategy == "semantic":
        return SemanticChunker(**kwargs)
    else:
        # Default to recursive
        return RecursiveTextChunker(**kwargs)