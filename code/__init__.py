"""
VERITAS - Cryptographically Verifiable Semantic Traces for AI Agent Provenance

A reference implementation of the VERITAS standard for creating cryptographically
verifiable decision traces for AI agents.

Main modules:
- core: Core data structures (DecisionTrace, Node, Edge, etc.)
- crypto: Cryptographic primitives (signatures, commitments, Merkle-DAG)
- compression: Semantic compression and embedding
- serialization: JSON serialization and schema validation

Quick start:
    >>> from veritas import DecisionTrace, NodeType
    >>> from veritas.crypto import SignatureScheme, TraceVerifier
    >>>
    >>> # Create a trace
    >>> trace = DecisionTrace()
    >>> node = trace.add_reasoning_step(
    ...     content="Some reasoning...",
    ...     node_type=NodeType.REASONING
    ... )
    >>>
    >>> # Finalize and sign
    >>> private_key, public_key = SignatureScheme.generate_keypair()
    >>> TraceVerifier.finalize_trace(trace, private_key)
    >>>
    >>> # Verify
    >>> is_valid = TraceVerifier.verify_trace_integrity(trace, public_key)

For more details, see README.md or run example.py.
"""

__version__ = "1.0.0"
__author__ = "Chris von Csefalvay, Mohsen Amiribesheli"

# Import main classes for convenience
from core import (
    DecisionTrace,
    Node,
    Edge,
    TraceGraph,
    NodeType,
    RelationType,
    AgentManifest,
    NodeMetadata,
    TemporalAttestation,
)

from crypto import (
    SignatureScheme,
    TraceVerifier,
    MerkleDAG,
    CommitmentScheme,
    HashFunction,
)

from compression import (
    TraceCompressor,
    SemanticEmbedder,
    EmbeddingCompressor,
    SemanticSearch,
)

from serialization import (
    TraceSerializer,
    SchemaValidator,
)

__all__ = [
    # Core
    "DecisionTrace",
    "Node",
    "Edge",
    "TraceGraph",
    "NodeType",
    "RelationType",
    "AgentManifest",
    "NodeMetadata",
    "TemporalAttestation",
    # Crypto
    "SignatureScheme",
    "TraceVerifier",
    "MerkleDAG",
    "CommitmentScheme",
    "HashFunction",
    # Compression
    "TraceCompressor",
    "SemanticEmbedder",
    "EmbeddingCompressor",
    "SemanticSearch",
    # Serialization
    "TraceSerializer",
    "SchemaValidator",
]
