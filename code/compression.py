"""
VERITAS Semantic Compression
Implementation of semantic compression for reasoning traces.

This module implements the 4-step shape-embed-compress sequence described
in Section 2.3 of the paper:
1. Extract content
2. Embed semantically
3. Compress embedding
4. Commit to full content
"""

from typing import List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

from core import Node, NodeType


@dataclass
class CompressionConfig:
    """Configuration for semantic compression."""
    embedding_dim: int = 768  # Standard BERT/sentence-transformer dimension
    compressed_dim: int = 256  # Compressed dimension
    use_pca: bool = True  # Use PCA for compression
    normalize: bool = True  # Normalize embeddings


class SemanticEmbedder:
    """
    Semantic embedding interface using sentence-transformers.

    Uses sentence-transformers library for production-grade semantic embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 768):
        """
        Initialise embedder.

        Args:
            model_name: Name of embedding model (e.g., "all-MiniLM-L6-v2")
            embedding_dim: Dimension of output embeddings (ignored if using real model)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._real_model = None

        # Load sentence-transformers model
        try:
            from sentence_transformers import SentenceTransformer
            self._real_model = SentenceTransformer(model_name)
            self.embedding_dim = self._real_model.get_sentence_embedding_dimension()
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for semantic embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e

    def embed(self, text: str) -> List[float]:
        """
        Generate semantic embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        # Use sentence-transformers model
        embedding = self._real_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class EmbeddingCompressor:
    """
    Learned compression of semantic embeddings.

    Reduces dimensionality while preserving semantic structure.
    """

    def __init__(self, config: CompressionConfig):
        """
        Initialize compressor.

        Args:
            config: Compression configuration
        """
        self.config = config
        self._pca_components = None
        self._is_fitted = False

    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit compression model on embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        """
        if self.config.use_pca:
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=self.config.compressed_dim)
            self._pca.fit(embeddings)
            self._is_fitted = True
        else:
            # Use random projection as alternative
            from sklearn.random_projection import GaussianRandomProjection
            self._projector = GaussianRandomProjection(
                n_components=self.config.compressed_dim,
                random_state=42
            )
            self._projector.fit(embeddings)
            self._is_fitted = True

    def compress(self, embedding: List[float]) -> List[float]:
        """
        Compress an embedding.

        Args:
            embedding: Full embedding vector

        Returns:
            Compressed embedding vector
        """
        embedding_array = np.array(embedding).reshape(1, -1)

        if not self._is_fitted:
            # If not fitted, use simple truncation
            compressed = embedding_array[:, :self.config.compressed_dim]
        elif self.config.use_pca:
            compressed = self._pca.transform(embedding_array)
        else:
            compressed = self._projector.transform(embedding_array)

        if self.config.normalize:
            norm = np.linalg.norm(compressed)
            if norm > 0:
                compressed = compressed / norm

        return compressed[0].tolist()

    def decompress_approximate(self, compressed: List[float]) -> List[float]:
        """
        Approximate decompression (lossy).

        Args:
            compressed: Compressed embedding

        Returns:
            Approximation of original embedding
        """
        compressed_array = np.array(compressed).reshape(1, -1)

        if not self._is_fitted or not self.config.use_pca:
            # Cannot decompress without PCA
            # Pad with zeros
            padded = np.zeros((1, self.config.embedding_dim))
            padded[:, :self.config.compressed_dim] = compressed_array
            return padded[0].tolist()

        # PCA inverse transform
        decompressed = self._pca.inverse_transform(compressed_array)
        return decompressed[0].tolist()


class TraceCompressor:
    """
    High-level interface for compressing decision traces.

    Implements the complete compression pipeline from Section 2.3.
    """

    def __init__(
        self,
        embedder: Optional[SemanticEmbedder] = None,
        compressor: Optional[EmbeddingCompressor] = None
    ):
        """
        Initialize trace compressor.

        Args:
            embedder: Semantic embedder (default: all-MiniLM-L6-v2 model)
            compressor: Embedding compressor (default: simple truncation)
        """
        self.embedder = embedder or SemanticEmbedder()
        self.compressor = compressor or EmbeddingCompressor(CompressionConfig())

    def compress_node(self, node: Node) -> None:
        """
        Compress a node's content into semantic embedding.

        Modifies node in-place to add semantic_embedding.

        Args:
            node: Node to compress
        """
        if not node.full_content:
            return

        # Step 1: Extract content (already in node.full_content)

        # Step 2: Embed semantically
        full_embedding = self.embedder.embed(node.full_content)

        # Step 3: Compress embedding
        compressed_embedding = self.compressor.compress(full_embedding)

        # Step 4: Store compressed embedding
        node.semantic_embedding = compressed_embedding

        # Note: Commitment to full content is handled by crypto module

    def compress_trace(self, trace) -> None:
        """
        Compress all nodes in a trace.

        Args:
            trace: DecisionTrace to compress
        """
        for node in trace.trace_graph.nodes:
            self.compress_node(node)

    def get_compression_ratio(self, node: Node) -> float:
        """
        Calculate compression ratio for a node.

        Args:
            node: Node to analyze

        Returns:
            Compression ratio (original_size / compressed_size)
        """
        if not node.full_content or not node.semantic_embedding:
            return 1.0

        # Estimate sizes
        original_size = len(node.full_content.encode('utf-8'))
        # Embedding stored as float32: 4 bytes per dimension
        compressed_size = len(node.semantic_embedding) * 4

        return original_size / compressed_size


class SemanticSearch:
    """
    Semantic search over compressed trace embeddings.

    Demonstrates the utility of compressed embeddings for
    similarity comparison and pattern recognition.
    """

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity in range [-1, 1]
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

        if norm_product == 0:
            return 0.0

        return float(dot_product / norm_product)

    @staticmethod
    def find_similar_nodes(
        query_node: Node,
        candidate_nodes: List[Node],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[tuple]:
        """
        Find nodes semantically similar to query.

        Args:
            query_node: Query node
            candidate_nodes: Nodes to search
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (node, similarity) tuples
        """
        if not query_node.semantic_embedding:
            return []

        results = []
        for node in candidate_nodes:
            if not node.semantic_embedding or node.node_id == query_node.node_id:
                continue

            similarity = SemanticSearch.cosine_similarity(
                query_node.semantic_embedding,
                node.semantic_embedding
            )

            if similarity >= threshold:
                results.append((node, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    @staticmethod
    def cluster_reasoning_steps(nodes: List[Node], n_clusters: int = 5) -> Dict[int, List[Node]]:
        """
        Cluster reasoning steps by semantic similarity.

        Args:
            nodes: Nodes to cluster
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping cluster_id to list of nodes
        """
        # Extract embeddings
        embeddings = []
        valid_nodes = []

        for node in nodes:
            if node.semantic_embedding:
                embeddings.append(node.semantic_embedding)
                valid_nodes.append(node)

        if not embeddings:
            return {0: nodes}

        # Cluster
        try:
            from sklearn.cluster import KMeans
            embeddings_array = np.array(embeddings)
            kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
            labels = kmeans.fit_predict(embeddings_array)

            # Group by cluster
            clusters = {}
            for node, label in zip(valid_nodes, labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(node)

            return clusters

        except ImportError:
            print("Warning: scikit-learn not available for clustering")
            return {0: valid_nodes}


def analyze_compression_efficiency(trace) -> Dict[str, Any]:
    """
    Analyze compression efficiency for a trace.

    Args:
        trace: DecisionTrace to analyze

    Returns:
        Dictionary of compression statistics
    """
    total_original = 0
    total_compressed = 0
    node_count = 0

    compressor = TraceCompressor()

    for node in trace.trace_graph.nodes:
        if node.full_content and node.semantic_embedding:
            ratio = compressor.get_compression_ratio(node)
            if ratio > 0:
                original_size = len(node.full_content.encode('utf-8'))
                compressed_size = len(node.semantic_embedding) * 4
                total_original += original_size
                total_compressed += compressed_size
                node_count += 1

    if total_original == 0:
        return {
            "total_nodes": node_count,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "compression_ratio": 1.0,
            "space_savings_percent": 0.0
        }

    compression_ratio = total_original / total_compressed
    space_savings = (1 - total_compressed / total_original) * 100

    return {
        "total_nodes": node_count,
        "total_original_bytes": total_original,
        "total_compressed_bytes": total_compressed,
        "compression_ratio": compression_ratio,
        "space_savings_percent": space_savings
    }
