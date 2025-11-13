# VERITAS Reference Implementation

**Cryptographically Verifiable Semantic Traces for AI Agent Provenance**

This is a reference implementation of the VERITAS (Verifiable Reasoning Trace Attestation Standard) format as described in the paper "VERITAS: Cryptographically Verifiable Semantic Traces for AI Agent Provenance and Interoperability."

## Overview

VERITAS enables AI agents to create cryptographically verifiable traces of their decision-making processes. This reference implementation provides:

- **Cryptographic Integrity**: Merkle-DAG structure with digital signatures ensures tamper-evidence
- **Semantic Compression**: Reduce trace size by 60-80% while preserving verification capability
- **Selective Disclosure**: Zero-knowledge proof foundations (framework only; ZK proofs require additional libraries)
- **Cross-Framework Compatibility**: JSON-based interchange format for heterogeneous agent systems
- **Efficient Verification**: O(log n) proof checking with minimal runtime overhead

## Architecture

The implementation consists of five core modules:

### 1. `core.py` - Core Data Structures

Implements the fundamental VERITAS types as defined in Section 2 of the paper:

- `DecisionTrace`: Complete trace T = (V, E, M, Σ, τ)
- `Node`: Reasoning step with semantic embedding and cryptographic commitment
- `Edge`: Dependency relationships between nodes
- `TraceGraph`: DAG structure with topological operations
- `AgentManifest`: Agent identity and cryptographic keys
- `TemporalAttestation`: RFC 3161 timestamp integration

**Key Concepts:**
```python
from core import DecisionTrace, NodeType

# Create a trace
trace = DecisionTrace()

# Add reasoning steps
node = trace.add_reasoning_step(
    content="Patient presents with fever and cough...",
    node_type=NodeType.REASONING,
    parent_ids=["previous_node_id"],
    confidence=0.85
)
```

### 2. `crypto.py` - Cryptographic Primitives

Implements Section 3 (Cryptographic Approach):

- **Hash Functions**: SHA-256 and BLAKE3 support
- **Merkle-DAG Construction**: Recursive hash binding per Equation 1
- **Commitment Scheme**: Hash-based commitments with hiding and binding properties
- **Digital Signatures**: EdDSA (Ed25519) for performance
- **Trace Verification**: Complete integrity checking

**Merkle-DAG Binding (Equation 1):**
```
bind(v) = H(commit(v) || bind(p1) || ... || bind(pk) || metadata(v))
```

**Usage:**
```python
from crypto import TraceVerifier, SignatureScheme

# Generate keypair
private_key, public_key = SignatureScheme.generate_keypair()

# Finalize trace with cryptographic proofs
TraceVerifier.finalize_trace(trace, private_key)

# Verify integrity
is_valid = TraceVerifier.verify_trace_integrity(trace, public_key)
```

### 3. `compression.py` - Semantic Compression

Implements Section 2.3 (Semantic Compression):

The 4-step compression pipeline:
1. **Extract content**: Gather reasoning text, tool calls, outputs
2. **Embed semantically**: Generate 768-d dense embedding
3. **Compress embedding**: Reduce to 256-512 dimensions
4. **Commit to full content**: Hash-based commitment for verification

**Features:**
- Pluggable embedding models (sentence-transformers support)
- PCA or random projection compression
- Semantic search over compressed embeddings
- Clustering of reasoning steps

**Usage:**
```python
from compression import TraceCompressor, SemanticEmbedder

# Create compressor with real embeddings (requires sentence-transformers)
embedder = SemanticEmbedder("all-MiniLM-L6-v2")
compressor = TraceCompressor(embedder=embedder)

# Compress all nodes in trace
compressor.compress_trace(trace)
```

### 4. `serialization.py` - JSON Schema

Implements Section 2.4 (JSON Schema):

- Bidirectional JSON serialization
- Schema validation
- Compact representation (excludes full content by default)
- Optional full content storage for debugging

**JSON Structure:**
```json
{
  "veritas_version": "1.0",
  "trace_id": "uuid-v4",
  "created_at": "2025-10-19T14:23:17Z",
  "agent_manifest": {
    "agent_did": "did:agent:anthropic:claude-sonnet-4",
    "model_version": "claude-sonnet-4-20250514",
    "public_key": "base64-encoded-key",
    "signature": "base64-signature"
  },
  "trace_graph": {
    "nodes": [...],
    "edges": [...]
  },
  "root_hash": "sha256:complete-trace-hash"
}
```

**Usage:**
```python
from serialization import TraceSerializer

# Save trace to JSON
TraceSerializer.save_trace(trace, "trace.json")

# Load from JSON
loaded_trace = TraceSerializer.load_trace("trace.json")
```

### 5. `example.py` - Complete Demonstration

A comprehensive example showing:
- Creating a multi-step medical diagnosis trace
- Computing cryptographic proofs
- Semantic compression
- Serialization and deserialization
- Tampering detection

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/veritas-dev.git
cd veritas-dev/code

# Install core dependencies
pip install -r requirements.txt
```

### Optional Dependencies

For full functionality, install optional dependencies:

```bash
# For real semantic embeddings (recommended for production)
pip install sentence-transformers torch

# For compression (PCA, clustering)
pip install scikit-learn

# For BLAKE3 hashing (faster alternative to SHA-256)
pip install blake3
```

## Quick Start

### Running the Example

```bash
python example.py
```

This will:
1. Create a multi-step decision trace
2. Compute semantic embeddings
3. Generate cryptographic proofs (Merkle-DAG, signatures)
4. Serialize to JSON
5. Verify trace integrity
6. Demonstrate tampering detection

### Creating Your Own Trace

```python
from core import DecisionTrace, NodeType, AgentManifest
from crypto import SignatureScheme, TraceVerifier
from compression import TraceCompressor
from serialization import TraceSerializer

# 1. Setup
private_key, public_key = SignatureScheme.generate_keypair()
trace = DecisionTrace()
trace.agent_manifest = AgentManifest(
    agent_did="did:agent:myorg:my-agent-v1",
    model_version="gpt-4-turbo",
    framework="custom"
)

# 2. Add reasoning steps
n1 = trace.add_reasoning_step(
    content="Initial observation or input...",
    node_type=NodeType.OBSERVATION,
    confidence=0.95
)

n2 = trace.add_reasoning_step(
    content="Reasoning about the observation...",
    node_type=NodeType.REASONING,
    parent_ids=[n1.node_id],
    confidence=0.88
)

n3 = trace.add_reasoning_step(
    content="Final decision: ...",
    node_type=NodeType.DECISION,
    parent_ids=[n2.node_id],
    confidence=0.92
)

# 3. Compress (optional but recommended)
compressor = TraceCompressor()
compressor.compress_trace(trace)

# 4. Finalize with cryptographic proofs
TraceVerifier.finalize_trace(trace, private_key)

# 5. Verify
is_valid = TraceVerifier.verify_trace_integrity(trace, public_key)
print(f"Trace valid: {is_valid}")

# 6. Save
TraceSerializer.save_trace(trace, "my_trace.json")
```

## Key Concepts

### Decision Trace Structure

A VERITAS trace is formally defined as:

**T = (V, E, M, Σ, τ)**

where:
- **V**: Set of reasoning step nodes
- **E**: Set of directed edges (dependencies)
- **M**: Agent manifest (identity, keys)
- **Σ**: Cryptographic signatures
- **τ**: Temporal attestation

### Node Types

The implementation supports five node types from the paper:

- `REASONING`: Internal reasoning and analysis
- `TOOL_CALL`: External tool/API invocation
- `OBSERVATION`: External information or sensor data
- `DECISION`: Final or intermediate decisions
- `MEMORY_ACCESS`: Retrieval from agent memory

### Cryptographic Binding

Each node has a cryptographic binding computed as:

```
bind(v) = H(commit(v) || bind(parent_1) || ... || bind(parent_k) || metadata(v))
```

This creates a Merkle-DAG where:
- Tampering with any node invalidates its commitment
- Reordering nodes invalidates bindings
- The root hash serves as a compact fingerprint

### Root Hash Computation

The root hash is computed as (Equation 2 in paper):

```
root = H(bind(terminal_nodes) || agent_manifest || temporal_attestation)
```

The agent signs this root hash, creating a non-repudiable cryptographic proof.

## Performance

Based on Section 4.1 of the paper:

**Trace Generation Overhead:**
- Per-step overhead: ~5-10ms
- Components:
  - Semantic embedding: 10-50ms
  - Commitment: ~1ms
  - Binding: ~2ms
  - Signature (terminal): ~3ms
- For 100-step workflow: +0.5-1s overhead (<2% increase)

**Storage:**
- Raw trace: ~50KB per step
- Compressed: ~4KB per step
- **Reduction: 92% space savings**

**Verification:**
- Single node: ~5ms
- Full trace: ~10ms + 2ms per node
- With ZK proof: +5ms

## Design Principles

The implementation follows the five core principles from Section 2.1:

1. **P1: Tamper-evidence** - Any modification invalidates cryptographic proofs
2. **P2: Selective disclosure** - ZK proof foundations (requires additional libraries)
3. **P3: Cross-framework compatibility** - JSON interchange format
4. **P4: Semantic richness** - Preserves causal structure and metadata
5. **P5: Practical efficiency** - <10ms overhead per reasoning step

## Security

### Threat Model

The implementation defends against (Section 4.2):

1. **Forgery**: Attackers claiming false provenance
2. **Tampering**: Post-hoc modification of reasoning
3. **Replay attacks**: Reusing old traces in new contexts
4. **Selective disclosure abuse**: Misleading cherry-picked samples

### Cryptographic Security

- **Signatures**: EdDSA (Ed25519) - 128-bit security
- **Hashing**: SHA-256 or BLAKE3 - collision resistance
- **Commitments**: Hash-based with hiding and binding properties

### Verification

Always verify traces before trusting:

```python
from crypto import TraceVerifier

# Verify complete trace
is_valid = TraceVerifier.verify_trace_integrity(trace, agent_public_key)

if not is_valid:
    raise SecurityError("Trace verification failed - possible tampering")
```

## Limitations

1. **ZK Proofs**: This reference implementation provides the foundation but does not include full zero-knowledge proof generation (Groth16, Bulletproofs). For production ZK support, integrate libraries like libsnark or bellman.

2. **Temporal Attestation**: RFC 3161 timestamp integration is defined but requires connection to a timestamp authority. Production deployments should use services like FreeTSA or DigiCert.

3. **Mock Embeddings**: Default configuration uses hash-based mock embeddings. For production, install sentence-transformers for real semantic embeddings.

4. **Performance**: ZK proof generation can take 100ms-2s. Consider proof caching for real-time applications.

## Integration with Agent Frameworks

### LangChain Integration

```python
from langchain.callbacks.base import BaseCallbackHandler
from core import DecisionTrace, NodeType

class VeritasCallbackHandler(BaseCallbackHandler):
    def __init__(self, trace: DecisionTrace):
        self.trace = trace
        self.node_map = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        # Add reasoning node
        node = self.trace.add_reasoning_step(
            content=prompts[0],
            node_type=NodeType.REASONING
        )
        self.node_map['current'] = node.node_id

    def on_tool_start(self, serialized, input_str, **kwargs):
        # Add tool call node
        parent_id = self.node_map.get('current')
        node = self.trace.add_reasoning_step(
            content=f"Tool: {serialized['name']} - Input: {input_str}",
            node_type=NodeType.TOOL_CALL,
            parent_ids=[parent_id] if parent_id else []
        )
```

### Custom Agent Integration

Wrap your agent's reasoning loop:

```python
def agent_step(observation, trace, parent_id=None):
    # Record observation
    obs_node = trace.add_reasoning_step(
        content=observation,
        node_type=NodeType.OBSERVATION,
        parent_ids=[parent_id] if parent_id else []
    )

    # Perform reasoning
    reasoning_result = your_agent.reason(observation)

    # Record reasoning
    reasoning_node = trace.add_reasoning_step(
        content=reasoning_result,
        node_type=NodeType.REASONING,
        parent_ids=[obs_node.node_id]
    )

    return reasoning_node.node_id
```

## Use Cases

### 1. Regulatory Compliance (Section 1.2)

```python
# Healthcare agent making treatment recommendations
trace = create_trace_for_patient(patient_id)

# Agent performs reasoning...
# ...

# Generate compliance proof
TraceVerifier.finalize_trace(trace, agent_private_key)

# Submit to regulatory authority
TraceSerializer.save_trace(trace, f"audit_trail_{patient_id}.json")
```

### 2. Adversarial Robustness

```python
# Verify agent followed safety protocols
if not TraceVerifier.verify_trace_integrity(trace, agent_public_key):
    alert("Possible jailbreak - trace tampered!")

# Check that all tool calls were from approved list
for node in trace.trace_graph.nodes:
    if node.node_type == NodeType.TOOL_CALL:
        verify_tool_approved(node)
```

### 3. Knowledge Distillation

```python
# Collect verified traces for training
verified_traces = []
for trace_file in training_traces:
    trace = TraceSerializer.load_trace(trace_file)
    if TraceVerifier.verify_trace_integrity(trace, teacher_public_key):
        verified_traces.append(trace)

# Train student model only on verified reasoning
train_student_model(verified_traces)
```

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=.
```

(Note: Test suite not included in this reference implementation but should cover all modules)

## Contributing

This is a reference implementation of the VERITAS standard. Contributions welcome:

- Performance optimizations
- Additional embedding models
- ZK proof implementations (Groth16, Bulletproofs)
- Framework integrations (AutoGPT, Semantic Kernel, etc.)
- Production-grade temporal attestation

## Citation

If you use VERITAS in your research, please cite:

```bibtex
@article{veritas2025,
  title={VERITAS: Cryptographically Verifiable Semantic Traces for AI Agent Provenance and Interoperability},
  author={von Csefalvay, Chris and Amiribesheli, Mohsen},
  journal={arXiv preprint},
  year={2025}
}
```

## License

[Your chosen license - e.g., MIT, Apache 2.0]

## References

See `../paper/veritas.bib` for complete bibliography.

Key references:
- **zkML**: EZKL, zkTorch for zero-knowledge machine learning
- **Merkle-DAG**: IPFS, Git for content-addressed storage
- **RFC 3161**: Time-Stamp Protocol for temporal attestation
- **EdDSA**: Ed25519 for digital signatures
- **Groth16**: Efficient zkSNARKs for zero-knowledge proofs

## Contact

For questions about the VERITAS standard or this implementation:

- Chris von Csefalvay - kristof.csefalvay@hcltech.com
- GitHub Issues: https://github.com/yourusername/veritas-dev/issues

## Roadmap

Future enhancements:

- [ ] Full Groth16 ZK proof implementation
- [ ] Production RFC 3161 timestamp integration
- [ ] Hardware security module (HSM) support
- [ ] Blockchain commitment (Ethereum, Hyperledger)
- [ ] Post-quantum cryptography (Dilithium, SPHINCS+)
- [ ] Federated verification across organizations
- [ ] GraphQL API for trace querying
- [ ] Web interface for trace visualization
