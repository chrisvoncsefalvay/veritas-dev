# VERITAS Semantic Compression Methods

This document describes the state-of-the-art semantic compression methods available in VERITAS for compressing decision traces.

## Overview

VERITAS supports multiple compression methods, each with different trade-offs between compression ratio, quality preservation, and computational cost. All methods are implemented in `compression_advanced.py`.

## Available Methods

### 1. Matryoshka Representation Learning (MRL)

**Status**: 🟢 Production-Ready (2024)

**Description**: Creates nested embeddings where truncating to smaller dimensions preserves semantic information. Allows flexible compression at deployment time without retraining.

**References**:
- Kusupati et al. "Matryoshka Representation Learning" (NeurIPS 2022)
- [arXiv:2205.13147](https://arxiv.org/abs/2205.13147)
- Used in OpenAI and Google Gemini embedding APIs (2024)

**Performance**:
- Compression Ratio: Up to 14x (768d → 64d)
- Quality: Minimal loss, maintains >95% similarity at 256d
- Speed: Fast (truncation-based)

**Use Cases**:
- Production deployments requiring flexibility
- When you need adjustable quality-size trade-offs
- Cross-platform embedding compatibility

**Example**:
```python
from compression_advanced import MatryoshkaCompressor

# Initialize
mrl = MatryoshkaCompressor(full_dim=768, target_dims=[64, 128, 256, 384, 768])

# Compress to 256 dimensions
compressed = mrl.compress(embedding, target_dim=256)

# Decompress (pads with zeros)
reconstructed = mrl.decompress(compressed, original_dim=768)

# Get optimal dimension for quality threshold
optimal_dim = mrl.get_optimal_dim(quality_threshold=0.95)  # Returns 256
```

**Pros**:
- Flexible compression levels
- Production-proven (OpenAI, Google)
- No training required
- Fast compression/decompression

**Cons**:
- Requires embeddings trained with MRL loss (or uses approximation)
- Moderate compression compared to PQ/Binary

---

### 2. Product Quantization (PQ)

**Status**: 🟢 Production-Ready (Classical)

**Description**: Splits embedding into subvectors and quantizes each independently using learned codebooks. Achieves extreme compression with acceptable accuracy loss.

**References**:
- Jegou et al. "Product Quantization for NN Search" (TPAMI 2011)
- Used in FAISS, Pinecone, Milvus

**Performance**:
- Compression Ratio: 32-64x (768d float32 → 96 bytes)
- Quality: Good for similarity search (~0.90-0.95 similarity)
- Speed: Fast decompression, moderate compression

**Use Cases**:
- Large-scale vector databases
- When storage is the primary concern
- Approximate nearest neighbor search

**Example**:
```python
from compression_advanced import ProductQuantizer
import numpy as np

# Initialize
pq = ProductQuantizer(
    embedding_dim=768,
    n_subvectors=96,  # 768 / 96 = 8 dims per subvector
    n_bits=8          # 256 centroids
)

# Train on embeddings (important!)
training_data = np.random.randn(10000, 768)  # Your training embeddings
pq.train(training_data)

# Compress
codes = pq.compress(embedding)  # Returns bytes (96 bytes)

# Decompress
reconstructed = pq.decompress(codes)

# Get stats
stats = pq.get_compression_stats()
print(f"Compression: {stats['compression_ratio']:.1f}x")
```

**Pros**:
- Extreme compression (32-64x)
- Fast similarity search in compressed space
- Industry-standard in vector databases

**Cons**:
- Requires training on representative data
- Higher reconstruction error than simpler methods
- Complexity in implementation

---

### 3. Binary Embeddings / Deep Hashing

**Status**: 🟢 Production-Ready

**Description**: Converts embeddings to binary codes using sign-based hashing. Enables ultra-fast similarity search using Hamming distance.

**References**:
- Various deep hashing methods (2015-2024)
- HuggingFace: "Binary and Scalar Embedding Quantization" (2024)
- [Blog Post](https://huggingface.co/blog/embedding-quantization)

**Performance**:
- Compression Ratio: 32x (768d float32 → 96 bytes)
- Quality: Moderate (~0.85-0.90 similarity)
- Speed: Extremely fast (Hamming distance via XOR)

**Use Cases**:
- Real-time similarity search
- Maximum compression with acceptable quality loss
- Large-scale retrieval systems

**Example**:
```python
from compression_advanced import BinaryEmbedding

# Initialize
binary = BinaryEmbedding(embedding_dim=768)

# Compress to binary codes
codes = binary.compress(embedding)  # Returns bytes (96 bytes)

# Decompress (approximate reconstruction)
reconstructed = binary.decompress(codes)

# Fast similarity computation
codes1 = binary.compress(embedding1)
codes2 = binary.compress(embedding2)
hamming_dist = binary.hamming_distance(codes1, codes2)
cosine_sim_approx = binary.cosine_similarity_approx(codes1, codes2)

# Stats
stats = binary.get_compression_stats()
```

**Pros**:
- Ultra-fast similarity search (25x speedup)
- Maximum compression
- Simple XOR operations

**Cons**:
- Significant quality loss
- Not suitable for exact reconstruction
- Best for retrieval, not semantic analysis

---

### 4. Scalar Quantization (int8 / float16)

**Status**: 🟢 Production-Ready (Industry Standard)

**Description**: Simple quantization of float32 to int8 or float16. Minimal quality loss with straightforward implementation.

**References**:
- HuggingFace: 3.66x speedup with int8 (2024)
- AWS OpenSearch: Standard quantization technique
- [Blog Post](https://huggingface.co/blog/embedding-quantization)

**Performance**:
- **int8**: 4x compression, >0.98 similarity
- **float16**: 2x compression, >0.99 similarity
- Speed: Very fast

**Use Cases**:
- Default compression for most applications
- When quality is paramount
- GPU-accelerated inference

**Example**:
```python
from compression_advanced import ScalarQuantizer

# Int8 quantization
int8_quant = ScalarQuantizer(mode='int8')
codes = int8_quant.compress(embedding)
reconstructed = int8_quant.decompress(codes)

# Float16 quantization
float16_quant = ScalarQuantizer(mode='float16')
codes = float16_quant.compress(embedding)
reconstructed = float16_quant.decompress(codes)

# With calibration for better int8 quality
import numpy as np
calibration_data = np.random.randn(1000, 768)
int8_calibrated = ScalarQuantizer(mode='int8', calibration_data=calibration_data)

# Stats
stats = int8_quant.get_compression_stats(embedding_dim=768)
```

**Pros**:
- Simple to implement
- Minimal quality loss
- Widely supported (hardware acceleration)
- No training required

**Cons**:
- Moderate compression only
- int8 may require calibration for best results

---

### 5. Autoencoder-based Compression

**Status**: 🟡 Research / Domain-Specific

**Description**: Uses neural networks to learn optimal nonlinear compression for specific data distributions.

**References**:
- Deep learning compression literature
- Domain-specific optimization

**Performance**:
- Compression Ratio: Flexible (typically 3-10x)
- Quality: Excellent when trained properly
- Speed: Moderate (neural network inference)

**Use Cases**:
- Domain-specific optimization
- When you have abundant training data
- Research applications

**Example**:
```python
from compression_advanced import AutoencoderCompressor
import numpy as np

# Initialize
ae = AutoencoderCompressor(
    input_dim=768,
    compressed_dim=128
)

# Train on your data
training_embeddings = np.random.randn(10000, 768)
ae.train(training_embeddings, epochs=10)

# Compress
compressed = ae.compress(embedding)

# Decompress
reconstructed = ae.decompress(compressed)

# Evaluate
error = ae.get_reconstruction_error(embedding)
print(f"Reconstruction MSE: {error:.6f}")
```

**Pros**:
- Can learn domain-specific patterns
- Nonlinear compression (better than PCA)
- Tunable compression ratio

**Cons**:
- Requires training data
- More complex than other methods
- Slower inference

---

## Comparison Matrix

| Method | Compression Ratio | Quality (Cosine Sim) | Speed | Training Required | Production Use |
|--------|------------------|----------------------|-------|-------------------|----------------|
| **Matryoshka** | 3-14x | >0.95 | Fast | No | ✅ OpenAI, Google |
| **Product Quantization** | 32-64x | 0.90-0.95 | Fast | Yes | ✅ FAISS, Pinecone |
| **Binary Hashing** | 32x | 0.85-0.90 | Very Fast | No | ✅ Large-scale retrieval |
| **int8 Quantization** | 4x | >0.98 | Very Fast | Optional | ✅ Industry standard |
| **float16 Quantization** | 2x | >0.99 | Very Fast | No | ✅ PyTorch, TensorFlow |
| **Autoencoder** | 3-10x | 0.92-0.98 | Moderate | Yes | 🟡 Research |

## Recommendations by Use Case

### For VERITAS Decision Traces

**Default Choice**: **Scalar Quantization (int8)**
- Reason: Best balance of simplicity, quality, and compression
- Compression: 4x (768d × 4 bytes → 768 bytes)
- Quality: >98% similarity preserved
- No training required

**Production Deployment**: **Matryoshka (256d)**
- Reason: Flexibility, production-proven, adjustable quality
- Compression: 3x (768d → 256d)
- Quality: >95% similarity
- Used by OpenAI, Google

**Storage-Critical**: **Product Quantization**
- Reason: Maximum compression for large-scale deployments
- Compression: 32-64x
- Quality: Good enough for similarity search
- Requires one-time training

**Real-Time Search**: **Binary Hashing**
- Reason: Ultra-fast Hamming distance search
- Compression: 32x
- Quality: Acceptable for retrieval
- 25x speedup in similarity search

### By Quality Requirements

**High Fidelity (>0.95 similarity)**:
1. float16 (2x compression)
2. int8 (4x compression)
3. Matryoshka 256d (3x compression)

**Moderate Fidelity (>0.90 similarity)**:
1. Matryoshka 128d (6x compression)
2. Product Quantization (32x compression)
3. Autoencoder (5-10x compression)

**Lower Fidelity Acceptable**:
1. Binary Hashing (32x compression)
2. Matryoshka 64d (12x compression)

### By Storage Constraints

**Tight Budget (<100 bytes per embedding)**:
- Binary Hashing: 96 bytes
- Product Quantization: 96 bytes

**Moderate (100-1000 bytes)**:
- Matryoshka 128d: 512 bytes
- int8: 768 bytes
- Matryoshka 256d: 1024 bytes

**Relaxed (>1000 bytes)**:
- float16: 1536 bytes
- Matryoshka 384d: 1536 bytes

## Implementation Guide

### Step 1: Choose Your Method

Consider your requirements:
- Quality vs. compression trade-off
- Training data availability
- Computational budget
- Production constraints

### Step 2: Integrate with VERITAS

```python
from core import DecisionTrace, NodeType
from compression_advanced import ScalarQuantizer  # or other method

# Create trace
trace = DecisionTrace()

# Add reasoning steps
node = trace.add_reasoning_step(
    content="Patient presents with fever...",
    node_type=NodeType.REASONING
)

# Compress node content
compressor = ScalarQuantizer(mode='int8')

for node in trace.trace_graph.nodes:
    if node.full_content:
        # Create embedding (in production, use sentence-transformers)
        embedding = create_embedding(node.full_content)

        # Compress
        compressed_codes = compressor.compress(embedding)

        # Store compressed version
        node.compressed_embedding = compressed_codes

# For similarity search
codes1 = trace.trace_graph.nodes[0].compressed_embedding
codes2 = trace.trace_graph.nodes[1].compressed_embedding

# Decompress for analysis
embedding1 = compressor.decompress(codes1)
embedding2 = compressor.decompress(codes2)
similarity = cosine_similarity(embedding1, embedding2)
```

### Step 3: Evaluate

Use the comparison notebook:
```bash
jupyter notebook notebooks/04_compression_methods_comparison.ipynb
```

This will help you:
- Benchmark all methods on your data
- Compare compression ratios
- Measure quality preservation
- Analyze speed trade-offs

## Future Directions

### Planned Enhancements

1. **Hybrid Methods**: Combine multiple techniques
   - Example: float16 + PCA for 8x with high quality
   - Example: Matryoshka + quantization

2. **Adaptive Compression**: Choose method based on content
   - High-importance nodes: float16
   - Regular nodes: int8
   - Archive nodes: Product Quantization

3. **Learned Quantization**: Train quantization for specific domains
   - Medical traces
   - Financial traces
   - Code generation traces

4. **Streaming Compression**: Compress as traces are generated

## References

### Academic Papers

1. Kusupati et al. "Matryoshka Representation Learning" (NeurIPS 2022)
   - https://arxiv.org/abs/2205.13147

2. Jegou et al. "Product Quantization for Nearest Neighbor Search" (TPAMI 2011)

3. Various deep hashing methods (2015-2024)

### Production Systems

1. FAISS - Facebook AI Similarity Search
   - https://github.com/facebookresearch/faiss

2. Pinecone - Vector Database
   - https://www.pinecone.io/learn/series/faiss/product-quantization/

3. HuggingFace - Binary and Scalar Quantization (2024)
   - https://huggingface.co/blog/embedding-quantization

4. AWS OpenSearch - Vector Database Quantization
   - https://aws.amazon.com/blogs/big-data/cost-optimized-vector-database-introduction-to-amazon-opensearch-service-quantization-techniques/

## Support

For questions or issues with compression methods:
- GitHub Issues: https://github.com/yourusername/veritas-dev/issues
- Email: kristof.csefalvay@hcltech.com

## License

Same as VERITAS main project.
