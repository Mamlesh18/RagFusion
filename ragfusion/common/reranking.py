"""
Result reranking strategies for RAGFusion.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from ragfusion.common.base import BaseReranker, SearchResult
from ragfusion.config import get_config


class CrossEncoderReranker(BaseReranker):
    """
    Rerank using a cross-encoder model.
    More accurate but slower than bi-encoders.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install ragfusion[embeddings]"
            )
        
        self.model_name = model_name
        self.device = device or get_config().embedding.device
        self.model = CrossEncoder(model_name, device=self.device)
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Query string
            results: Search results to rerank
            top_k: Number of results to return
            **kwargs: Additional parameters
        
        Returns:
            Reranked search results
        """
        if not results:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, result.content] for result in results]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Update result scores and sort
        for result, score in zip(results, scores):
            result.score = float(score)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k results
        if top_k:
            return results[:top_k]
        return results


class DiversityReranker(BaseReranker):
    """
    Rerank to increase diversity using MMR (Maximum Marginal Relevance).
    """
    
    def __init__(
        self,
        lambda_mult: float = 0.5,
        embedding_model: Optional[Any] = None,
    ):
        self.lambda_mult = lambda_mult
        self.embedding_model = embedding_model
        
        if self.embedding_model is None:
            from ragfusion.common.embeddings import get_embedding_model
            self.embedding_model = get_embedding_model()
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Rerank using Maximum Marginal Relevance for diversity.
        
        Args:
            query: Query string
            results: Search results to rerank
            top_k: Number of results to return
            **kwargs: Additional parameters
        
        Returns:
            Reranked search results with improved diversity
        """
        if not results:
            return []
        
        if top_k is None:
            top_k = len(results)
        
        # Get embeddings for all results
        result_texts = [r.content for r in results]
        result_embeddings = self.embedding_model.embed_documents(result_texts)
        
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Calculate similarity to query for all results
        query_similarities = [
            self._cosine_similarity(query_embedding, emb)
            for emb in result_embeddings
        ]
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(results)))
        
        # Select first document (highest similarity to query)
        first_idx = np.argmax(query_similarities)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select documents
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = query_similarities[idx]
                
                # Maximum similarity to already selected documents
                max_sim = max([
                    self._cosine_similarity(
                        result_embeddings[idx],
                        result_embeddings[sel_idx]
                    )
                    for sel_idx in selected_indices
                ])
                
                # MMR score
                mmr = self.lambda_mult * relevance - (1 - self.lambda_mult) * max_sim
                mmr_scores.append(mmr)
            
            # Select document with highest MMR score
            next_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        
        # Return reranked results
        return [results[idx] for idx in selected_indices]
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)


class ReciprocRankFusion(BaseReranker):
    """
    Combine multiple ranking lists using Reciprocal Rank Fusion.
    Useful for hybrid search.
    """
    
    def __init__(self, k: int = 60):
        self.k = k  # Constant for RRF formula
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
        ranking_lists: Optional[List[List[SearchResult]]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Rerank using Reciprocal Rank Fusion.
        
        Args:
            query: Query string (not used in RRF)
            results: Combined search results
            top_k: Number of results to return
            ranking_lists: Multiple ranking lists to fuse
            **kwargs: Additional parameters
        
        Returns:
            Fused and reranked results
        """
        if ranking_lists is None:
            # If no separate lists provided, just return original results
            return results[:top_k] if top_k else results
        
        # Calculate RRF scores
        rrf_scores = {}
        
        for ranking_list in ranking_lists:
            for rank, result in enumerate(ranking_list, 1):
                # Use document ID as key
                key = result.document_id
                if key not in rrf_scores:
                    rrf_scores[key] = 0
                rrf_scores[key] += 1 / (self.k + rank)
        
        # Update result scores
        result_dict = {r.document_id: r for r in results}
        for doc_id, score in rrf_scores.items():
            if doc_id in result_dict:
                result_dict[doc_id].score = score
        
        # Sort by RRF score
        sorted_results = sorted(
            result_dict.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        # Return top_k results
        if top_k:
            return sorted_results[:top_k]
        return sorted_results


class CombinedReranker(BaseReranker):
    """
    Combine multiple reranking strategies.
    """
    
    def __init__(
        self,
        rerankers: List[BaseReranker],
        weights: Optional[List[float]] = None,
    ):
        self.rerankers = rerankers
        self.weights = weights or [1.0] * len(rerankers)
        
        if len(self.weights) != len(self.rerankers):
            raise ValueError("Number of weights must match number of rerankers")
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Apply multiple reranking strategies and combine scores.
        
        Args:
            query: Query string
            results: Search results to rerank
            top_k: Number of results to return
            **kwargs: Additional parameters
        
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        # Store scores from each reranker
        all_scores = {}
        
        for reranker, weight in zip(self.rerankers, self.weights):
            reranked = reranker.rerank(query, results.copy(), top_k=None, **kwargs)
            
            # Normalize scores to [0, 1]
            max_score = max(r.score for r in reranked) if reranked else 1
            min_score = min(r.score for r in reranked) if reranked else 0
            score_range = max_score - min_score if max_score != min_score else 1
            
            for r in reranked:
                doc_id = r.document_id
                normalized_score = (r.score - min_score) / score_range
                
                if doc_id not in all_scores:
                    all_scores[doc_id] = 0
                all_scores[doc_id] += weight * normalized_score
        
        # Update result scores with combined scores
        result_dict = {r.document_id: r for r in results}
        for doc_id, score in all_scores.items():
            if doc_id in result_dict:
                result_dict[doc_id].score = score
        
        # Sort by combined score
        sorted_results = sorted(
            result_dict.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        # Return top_k results
        if top_k:
            return sorted_results[:top_k]
        return sorted_results


def get_reranker(
    strategy: Optional[str] = None,
    **kwargs
) -> BaseReranker:
    """
    Factory function to get a reranker.
    
    Args:
        strategy: Reranking strategy
        **kwargs: Additional arguments for the reranker
    
    Returns:
        Reranker instance
    """
    config = get_config()
    strategy = strategy or "cross-encoder"
    
    if strategy == "cross-encoder":
        return CrossEncoderReranker(**kwargs)
    elif strategy == "diversity":
        return DiversityReranker(**kwargs)
    elif strategy == "rrf":
        return ReciprocRankFusion(**kwargs)
    elif strategy == "combined":
        # Default combination: cross-encoder + diversity
        if "rerankers" not in kwargs:
            kwargs["rerankers"] = [
                CrossEncoderReranker(),
                DiversityReranker(lambda_mult=0.3),
            ]
            kwargs["weights"] = [0.7, 0.3]
        return CombinedReranker(**kwargs)
    else:
        # Default to cross-encoder
        return CrossEncoderReranker(**kwargs)