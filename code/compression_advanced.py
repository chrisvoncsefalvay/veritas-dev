"""
VERITAS Advanced Semantic Compression Methods
State-of-the-art compression techniques for decision traces.

This module implements several SOTA semantic compression methods:
1. Matryoshka Representation Learning (MRL) - Flexible nested embeddings
2. Product Quantization (PQ) - Classical but highly effective
3. Binary/Hamming Embeddings - Extreme compression with deep hashing
4. Scalar Quantization - Simple int8/float16 compression
5. Autoencoder-based Compression - Deep learning approach

References:
- Matryoshka Representation Learning (Kusupati et al., 2022)
  https://arxiv.org/abs/2205.13147
- Product Quantization for NN Search (Jegou et al., 2011)
- Binary and Scalar Embedding Quantization (HuggingFace, 2024)
  https://huggingface.co/blog/embedding-quantization
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import struct


@dataclass
class CompressionStats:
    """Statistics for compression performance."""
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    space_savings_pct: float
    method: str


# =============================================================================
# 1. Matryoshka Representation Learning (MRL)
# =============================================================================

class MatryoshkaCompressor:
    """
    Matryoshka Representation Learning compression.

    Creates nested embeddings where truncating to smaller dimensions
    preserves semantic information. Allows flexible compression at
    deployment time without retraining.

    Key insight: Train embeddings so that {d/8, d/4, d/2, d} all work well.

    References:
    - Kusupati et al. "Matryoshka Representation Learning" (NeurIPS 2022)
    - Achieves up to 14x compression with minimal accuracy loss
    """

    def __init__(
        self,
        full_dim: int = 768,
        target_dims: Optional[List[int]] = None
    ):
        """
        Initialize Matryoshka compressor.

        Args:
            full_dim: Full embedding dimension
            target_dims: Valid compression dimensions (e.g., [64, 128, 256, 384, 768])
        """
        self.full_dim = full_dim
        self.target_dims = target_dims or [64, 128, 256, 384, 512, 768]

        # Ensure target dims are sorted and valid
        self.target_dims = sorted([d for d in self.target_dims if d <= full_dim])

        # In a real MRL system, we'd have importance weights for each dimension
        # Here we simulate with a decay function
        self._importance_weights = self._compute_importance_weights()

    def _compute_importance_weights(self) -> np.ndarray:
        """
        Compute importance weights for each dimension.

        In MRL, early dimensions are more important. We simulate
        this with exponential decay.
        """
        weights = np.exp(-np.arange(self.full_dim) / (self.full_dim / 4))
        return weights / weights.sum()

    def compress(self, embedding: List[float], target_dim: int) -> List[float]:
        """
        Compress embedding to target dimension using Matryoshka approach.

        Args:
            embedding: Full embedding vector
            target_dim: Target dimension (must be in self.target_dims)

        Returns:
            Compressed embedding
        """
        if target_dim not in self.target_dims:
            # Find closest valid dimension
            target_dim = min(self.target_dims, key=lambda x: abs(x - target_dim))

        # Simple truncation (in real MRL, the model is trained for this)
        # Apply importance weighting for better results
        embedding_array = np.array(embedding)
        weighted = embedding_array * np.sqrt(self._importance_weights)

        return weighted[:target_dim].tolist()

    def decompress(self, compressed: List[float], original_dim: int) -> List[float]:
        """
        Pad compressed embedding back to original dimension.

        Note: This is lossy - information is not recovered.
        """
        compressed_array = np.array(compressed)
        result = np.zeros(original_dim)
        result[:len(compressed)] = compressed_array / np.sqrt(self._importance_weights[:len(compressed)])
        return result.tolist()

    def get_optimal_dim(self, quality_threshold: float = 0.95) -> int:
        """
        Get optimal dimension for a quality threshold.

        Args:
            quality_threshold: Minimum acceptable quality (0-1)

        Returns:
            Recommended dimension
        """
        # Estimate quality retention based on dimension
        for dim in self.target_dims:
            estimated_quality = dim / self.full_dim
            if estimated_quality >= quality_threshold:
                return dim
        return self.target_dims[-1]


# =============================================================================
# 2. Product Quantization (PQ)
# =============================================================================

class ProductQuantizer:
    """
    Product Quantization for extreme compression.

    Splits embedding into subvectors and quantizes each independently.
    Achieves 32-64x compression with acceptable accuracy loss.

    Used in FAISS and production vector databases.

    References:
    - Jegou et al. "Product Quantization for NN Search" (TPAMI 2011)
    - Achieves 64x compression in practice
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        n_subvectors: int = 96,  # 768 / 96 = 8 dims per subvector
        n_bits: int = 8  # 256 centroids per subvector
    ):
        """
        Initialize Product Quantizer.

        Args:
            embedding_dim: Dimension of embeddings
            n_subvectors: Number of subvectors (must divide embedding_dim)
            n_bits: Bits per subvector (2^n_bits centroids)
        """
        self.embedding_dim = embedding_dim
        self.n_subvectors = n_subvectors
        self.n_bits = n_bits
        self.n_centroids = 2 ** n_bits

        assert embedding_dim % n_subvectors == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by n_subvectors ({n_subvectors})"

        self.subvector_dim = embedding_dim // n_subvectors

        # Codebooks: one per subvector, each with n_centroids
        # In real PQ, these are learned via k-means on training data
        # Here we initialize randomly
        self.codebooks = self._initialize_codebooks()
        self._is_trained = False

    def _initialize_codebooks(self) -> List[np.ndarray]:
        """Initialize random codebooks."""
        np.random.seed(42)
        return [
            np.random.randn(self.n_centroids, self.subvector_dim).astype(np.float32)
            for _ in range(self.n_subvectors)
        ]

    def train(self, embeddings: np.ndarray) -> None:
        """
        Train codebooks using k-means on embedding data.

        Args:
            embeddings: Training embeddings (n_samples, embedding_dim)
        """
        from sklearn.cluster import MiniBatchKMeans

        # Split into subvectors
        subvectors = embeddings.reshape(-1, self.n_subvectors, self.subvector_dim)

        # Train a codebook for each subvector position
        for i in range(self.n_subvectors):
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_centroids,
                random_state=42,
                batch_size=min(1000, len(subvectors)),
                n_init=3
            )
            kmeans.fit(subvectors[:, i, :])
            self.codebooks[i] = kmeans.cluster_centers_.astype(np.float32)

        self._is_trained = True

    def compress(self, embedding: List[float]) -> bytes:
        """
        Compress embedding using product quantization.

        Args:
            embedding: Embedding vector

        Returns:
            Compressed codes as bytes (n_subvectors bytes)
        """
        embedding_array = np.array(embedding, dtype=np.float32)
        subvectors = embedding_array.reshape(self.n_subvectors, self.subvector_dim)

        codes = []
        for i, subvec in enumerate(subvectors):
            # Find nearest centroid in this codebook
            distances = np.linalg.norm(self.codebooks[i] - subvec, axis=1)
            code = np.argmin(distances)
            codes.append(code)

        # Pack codes into bytes
        return bytes(codes)

    def decompress(self, codes: bytes) -> List[float]:
        """
        Decompress codes back to embedding.

        Args:
            codes: Compressed codes

        Returns:
            Reconstructed embedding (lossy)
        """
        codes_list = list(codes)
        assert len(codes_list) == self.n_subvectors

        # Lookup centroids and concatenate
        subvectors = [
            self.codebooks[i][code]
            for i, code in enumerate(codes_list)
        ]

        reconstructed = np.concatenate(subvectors)
        return reconstructed.tolist()

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        original_bytes = self.embedding_dim * 4  # float32
        compressed_bytes = self.n_subvectors  # 1 byte per subvector

        return {
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': original_bytes / compressed_bytes,
            'space_savings_pct': (1 - compressed_bytes / original_bytes) * 100
        }


# =============================================================================
# 3. Binary Embeddings / Deep Hashing
# =============================================================================

class BinaryEmbedding:
    """
    Binary embeddings using sign-based hashing.

    Extreme compression: 768d float32 → 768 bits = 96 bytes
    32x compression with hamming distance for similarity.

    References:
    - Deep Hashing methods (various, 2015-2024)
    - HuggingFace: "Binary and Scalar Embedding Quantization" (2024)
    - Achieves 32x compression, 25x speedup
    """

    def __init__(self, embedding_dim: int = 768):
        """
        Initialize binary embedding.

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        # In real deep hashing, we'd learn a projection matrix
        # Here we use random projection (SimHash-style)
        np.random.seed(42)
        self.projection = np.random.randn(embedding_dim, embedding_dim).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0)

    def compress(self, embedding: List[float]) -> bytes:
        """
        Compress embedding to binary codes.

        Args:
            embedding: Embedding vector

        Returns:
            Binary codes as bytes
        """
        embedding_array = np.array(embedding, dtype=np.float32)

        # Project (optional, improves quality)
        projected = self.projection.T @ embedding_array

        # Binarize: sign function
        binary_codes = (projected >= 0).astype(np.uint8)

        # Pack bits into bytes
        return self._pack_bits(binary_codes)

    def decompress(self, codes: bytes) -> List[float]:
        """
        Decompress binary codes to approximate embedding.

        Note: Highly lossy, used mainly for similarity search.

        Args:
            codes: Binary codes

        Returns:
            Approximate embedding (-1 or +1 values)
        """
        # Unpack bits
        binary_codes = self._unpack_bits(codes, self.embedding_dim)

        # Convert to -1/+1
        pseudo_embedding = binary_codes.astype(np.float32) * 2 - 1

        return pseudo_embedding.tolist()

    def hamming_distance(self, codes1: bytes, codes2: bytes) -> int:
        """
        Compute hamming distance between two binary codes.

        Args:
            codes1: First binary codes
            codes2: Second binary codes

        Returns:
            Hamming distance (number of differing bits)
        """
        # XOR and count set bits
        xor_result = bytes(a ^ b for a, b in zip(codes1, codes2))
        return bin(int.from_bytes(xor_result, 'big')).count('1')

    def cosine_similarity_approx(self, codes1: bytes, codes2: bytes) -> float:
        """
        Approximate cosine similarity from hamming distance.

        Args:
            codes1: First binary codes
            codes2: Second binary codes

        Returns:
            Approximate cosine similarity
        """
        hamming_dist = self.hamming_distance(codes1, codes2)
        # cos(theta) ≈ 1 - 2 * hamming_dist / dim
        return 1.0 - 2.0 * hamming_dist / self.embedding_dim

    @staticmethod
    def _pack_bits(binary_array: np.ndarray) -> bytes:
        """Pack binary array into bytes."""
        # Pad to multiple of 8
        remainder = len(binary_array) % 8
        if remainder != 0:
            padding = np.zeros(8 - remainder, dtype=np.uint8)
            binary_array = np.concatenate([binary_array, padding])

        # Pack into bytes
        packed = np.packbits(binary_array)
        return packed.tobytes()

    @staticmethod
    def _unpack_bits(codes: bytes, target_len: int) -> np.ndarray:
        """Unpack bytes into binary array."""
        unpacked = np.unpackbits(np.frombuffer(codes, dtype=np.uint8))
        return unpacked[:target_len]

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        original_bytes = self.embedding_dim * 4  # float32
        compressed_bytes = (self.embedding_dim + 7) // 8  # Bits to bytes

        return {
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': original_bytes / compressed_bytes,
            'space_savings_pct': (1 - compressed_bytes / original_bytes) * 100
        }


# =============================================================================
# 4. Scalar Quantization
# =============================================================================

class ScalarQuantizer:
    """
    Scalar quantization to int8 or float16.

    Simple but effective: 4x compression with minimal accuracy loss.
    Widely used in production (OpenAI, Pinecone, etc.)

    References:
    - HuggingFace: Achieves 3.66x speedup with int8
    - AWS OpenSearch: Standard quantization technique
    """

    def __init__(
        self,
        mode: str = 'int8',  # 'int8' or 'float16'
        calibration_data: Optional[np.ndarray] = None
    ):
        """
        Initialize scalar quantizer.

        Args:
            mode: Quantization mode ('int8' or 'float16')
            calibration_data: Optional data for computing scale/offset
        """
        self.mode = mode
        self.scale = 1.0
        self.offset = 0.0

        if calibration_data is not None and mode == 'int8':
            self._calibrate(calibration_data)

    def _calibrate(self, data: np.ndarray) -> None:
        """
        Calibrate quantization parameters from data.

        Args:
            data: Calibration embeddings (n_samples, embedding_dim)
        """
        # Compute min/max for int8 range mapping
        min_val = data.min()
        max_val = data.max()

        # Map [min_val, max_val] to [-127, 127]
        self.scale = 254.0 / (max_val - min_val + 1e-8)
        self.offset = (max_val + min_val) / 2.0

    def compress_int8(self, embedding: List[float]) -> bytes:
        """
        Compress to int8.

        Args:
            embedding: Embedding vector

        Returns:
            Quantized int8 bytes
        """
        embedding_array = np.array(embedding, dtype=np.float32)

        # Quantize to [-127, 127]
        quantized = ((embedding_array - self.offset) * self.scale).clip(-127, 127)
        int8_array = quantized.astype(np.int8)

        return int8_array.tobytes()

    def decompress_int8(self, codes: bytes) -> List[float]:
        """
        Decompress int8 to float.

        Args:
            codes: Quantized int8 bytes

        Returns:
            Dequantized embedding
        """
        int8_array = np.frombuffer(codes, dtype=np.int8)
        float_array = int8_array.astype(np.float32) / self.scale + self.offset
        return float_array.tolist()

    def compress_float16(self, embedding: List[float]) -> bytes:
        """
        Compress to float16.

        Args:
            embedding: Embedding vector

        Returns:
            Float16 bytes
        """
        embedding_array = np.array(embedding, dtype=np.float32)
        float16_array = embedding_array.astype(np.float16)
        return float16_array.tobytes()

    def decompress_float16(self, codes: bytes) -> List[float]:
        """
        Decompress float16 to float32.

        Args:
            codes: Float16 bytes

        Returns:
            Float32 embedding
        """
        float16_array = np.frombuffer(codes, dtype=np.float16)
        float32_array = float16_array.astype(np.float32)
        return float32_array.tolist()

    def compress(self, embedding: List[float]) -> bytes:
        """Compress using configured mode."""
        if self.mode == 'int8':
            return self.compress_int8(embedding)
        else:
            return self.compress_float16(embedding)

    def decompress(self, codes: bytes) -> List[float]:
        """Decompress using configured mode."""
        if self.mode == 'int8':
            return self.decompress_int8(codes)
        else:
            return self.decompress_float16(codes)

    def get_compression_stats(self, embedding_dim: int) -> Dict[str, Any]:
        """Get compression statistics."""
        original_bytes = embedding_dim * 4  # float32

        if self.mode == 'int8':
            compressed_bytes = embedding_dim  # int8 = 1 byte
        else:  # float16
            compressed_bytes = embedding_dim * 2  # float16 = 2 bytes

        return {
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': original_bytes / compressed_bytes,
            'space_savings_pct': (1 - compressed_bytes / original_bytes) * 100
        }


# =============================================================================
# 5. Autoencoder-based Compression
# =============================================================================

class AutoencoderCompressor:
    """
    Autoencoder-based learned compression.

    Uses a neural network to learn optimal compression.
    More sophisticated than linear methods like PCA.

    Note: Requires training on domain-specific data for best results.
    """

    def __init__(
        self,
        input_dim: int = 768,
        compressed_dim: int = 128,
        hidden_dims: Optional[List[int]] = None
    ):
        """
        Initialize autoencoder compressor.

        Args:
            input_dim: Input embedding dimension
            compressed_dim: Compressed dimension
            hidden_dims: Hidden layer dimensions for encoder/decoder
        """
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dims = hidden_dims or [512, 256]

        # In a real implementation, we'd have actual neural network layers
        # For this reference implementation, we'll use a simple linear projection
        # initialized with SVD for better initial compression
        self._is_trained = False
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize encoder/decoder weights."""
        np.random.seed(42)
        # Random projection as initialization
        self.encoder_weight = np.random.randn(self.compressed_dim, self.input_dim).astype(np.float32)
        self.encoder_weight /= np.linalg.norm(self.encoder_weight, axis=1, keepdims=True)

        # Decoder is pseudo-inverse of encoder
        self.decoder_weight = np.linalg.pinv(self.encoder_weight)

    def train(self, embeddings: np.ndarray, epochs: int = 10) -> None:
        """
        Train autoencoder on embedding data.

        For this reference implementation, we use SVD as a proxy.

        Args:
            embeddings: Training embeddings (n_samples, input_dim)
            epochs: Number of training epochs (unused in SVD version)
        """
        # Use SVD for optimal linear compression
        U, S, Vt = np.linalg.svd(embeddings - embeddings.mean(axis=0), full_matrices=False)

        # Take top compressed_dim components
        self.encoder_weight = Vt[:self.compressed_dim].astype(np.float32)
        self.decoder_weight = Vt[:self.compressed_dim].T.astype(np.float32)

        self._is_trained = True

    def compress(self, embedding: List[float]) -> List[float]:
        """
        Compress embedding using encoder.

        Args:
            embedding: Input embedding

        Returns:
            Compressed embedding
        """
        embedding_array = np.array(embedding, dtype=np.float32)
        compressed = self.encoder_weight @ embedding_array
        return compressed.tolist()

    def decompress(self, compressed: List[float]) -> List[float]:
        """
        Decompress using decoder.

        Args:
            compressed: Compressed embedding

        Returns:
            Reconstructed embedding
        """
        compressed_array = np.array(compressed, dtype=np.float32)
        reconstructed = self.decoder_weight @ compressed_array
        return reconstructed.tolist()

    def get_reconstruction_error(self, embedding: List[float]) -> float:
        """
        Compute reconstruction error.

        Args:
            embedding: Original embedding

        Returns:
            Mean squared error
        """
        original = np.array(embedding)
        compressed = self.compress(embedding)
        reconstructed = np.array(self.decompress(compressed))

        mse = np.mean((original - reconstructed) ** 2)
        return float(mse)


# =============================================================================
# Utility Functions
# =============================================================================

def compare_compression_methods(
    embedding: List[float],
    methods: Optional[List[str]] = None
) -> Dict[str, CompressionStats]:
    """
    Compare different compression methods on the same embedding.

    Args:
        embedding: Test embedding
        methods: List of methods to test (default: all)

    Returns:
        Dictionary of compression statistics per method
    """
    if methods is None:
        methods = ['matryoshka', 'pq', 'binary', 'int8', 'float16', 'autoencoder']

    embedding_dim = len(embedding)
    results = {}

    # Matryoshka (256d)
    if 'matryoshka' in methods:
        mrl = MatryoshkaCompressor(full_dim=embedding_dim)
        compressed = mrl.compress(embedding, target_dim=256)
        original_bytes = embedding_dim * 4
        compressed_bytes = len(compressed) * 4
        results['Matryoshka (256d)'] = CompressionStats(
            original_size_bytes=original_bytes,
            compressed_size_bytes=compressed_bytes,
            compression_ratio=original_bytes / compressed_bytes,
            space_savings_pct=(1 - compressed_bytes / original_bytes) * 100,
            method='Matryoshka'
        )

    # Product Quantization
    if 'pq' in methods and embedding_dim % 96 == 0:
        pq = ProductQuantizer(embedding_dim=embedding_dim)
        compressed = pq.compress(embedding)
        stats = pq.get_compression_stats()
        results['Product Quantization'] = CompressionStats(
            original_size_bytes=stats['original_bytes'],
            compressed_size_bytes=stats['compressed_bytes'],
            compression_ratio=stats['compression_ratio'],
            space_savings_pct=stats['space_savings_pct'],
            method='Product Quantization'
        )

    # Binary
    if 'binary' in methods:
        binary = BinaryEmbedding(embedding_dim=embedding_dim)
        compressed = binary.compress(embedding)
        stats = binary.get_compression_stats()
        results['Binary Hashing'] = CompressionStats(
            original_size_bytes=stats['original_bytes'],
            compressed_size_bytes=stats['compressed_bytes'],
            compression_ratio=stats['compression_ratio'],
            space_savings_pct=stats['space_savings_pct'],
            method='Binary'
        )

    # Int8
    if 'int8' in methods:
        int8_quant = ScalarQuantizer(mode='int8')
        compressed = int8_quant.compress(embedding)
        stats = int8_quant.get_compression_stats(embedding_dim)
        results['Scalar Quantization (int8)'] = CompressionStats(
            original_size_bytes=stats['original_bytes'],
            compressed_size_bytes=stats['compressed_bytes'],
            compression_ratio=stats['compression_ratio'],
            space_savings_pct=stats['space_savings_pct'],
            method='Scalar Quantization (int8)'
        )

    # Float16
    if 'float16' in methods:
        float16_quant = ScalarQuantizer(mode='float16')
        compressed = float16_quant.compress(embedding)
        stats = float16_quant.get_compression_stats(embedding_dim)
        results['Scalar Quantization (float16)'] = CompressionStats(
            original_size_bytes=stats['original_bytes'],
            compressed_size_bytes=stats['compressed_bytes'],
            compression_ratio=stats['compression_ratio'],
            space_savings_pct=stats['space_savings_pct'],
            method='Scalar Quantization (float16)'
        )

    # Autoencoder
    if 'autoencoder' in methods:
        ae = AutoencoderCompressor(input_dim=embedding_dim, compressed_dim=128)
        compressed = ae.compress(embedding)
        original_bytes = embedding_dim * 4
        compressed_bytes = len(compressed) * 4
        results['Autoencoder (128d)'] = CompressionStats(
            original_size_bytes=original_bytes,
            compressed_size_bytes=compressed_bytes,
            compression_ratio=original_bytes / compressed_bytes,
            space_savings_pct=(1 - compressed_bytes / original_bytes) * 100,
            method='Autoencoder'
        )

    return results
