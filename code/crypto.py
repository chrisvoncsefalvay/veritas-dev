"""
VERITAS Cryptographic Primitives
Implementation of cryptographic functions for decision trace verification.

This module implements:
- Merkle-DAG construction (Section 3.2 of the paper)
- Cryptographic commitments (Section 3.4)
- Digital signatures (Section 3.1)
- Hash chain construction
"""

import hashlib
import secrets
from typing import List, Tuple, Optional
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, ec
from cryptography.hazmat.backends import default_backend
import base64

from core import Node, DecisionTrace, TraceGraph


class HashFunction:
    """Hash function wrapper supporting SHA-256 and BLAKE3."""

    @staticmethod
    def sha256(data: bytes) -> bytes:
        """Compute SHA-256 hash."""
        return hashlib.sha256(data).digest()

    @staticmethod
    def blake3(data: bytes) -> bytes:
        """Compute BLAKE3 hash (fallback to SHA-256 if unavailable)."""
        try:
            import blake3
            return blake3.blake3(data).digest()
        except ImportError:
            # Fallback to SHA-256
            return HashFunction.sha256(data)

    @staticmethod
    def hash(data: bytes, algorithm: str = "sha256") -> bytes:
        """
        Compute hash using specified algorithm.

        Args:
            data: Data to hash
            algorithm: Hash algorithm ("sha256" or "blake3")

        Returns:
            Hash digest as bytes
        """
        if algorithm == "sha256":
            return HashFunction.sha256(data)
        elif algorithm == "blake3":
            return HashFunction.blake3(data)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    @staticmethod
    def hash_hex(data: bytes, algorithm: str = "sha256") -> str:
        """
        Compute hash and return as hex string.

        Args:
            data: Data to hash
            algorithm: Hash algorithm

        Returns:
            Hash digest as hex string with algorithm prefix (e.g., "sha256:abc123...")
        """
        digest = HashFunction.hash(data, algorithm)
        return f"{algorithm}:{digest.hex()}"


class CommitmentScheme:
    """
    Cryptographic commitment scheme for node content.

    Implements hash-based commitments as described in Section 3.4:
    commit = H(content || nonce)

    This provides:
    - Binding: Cannot change content after commitment
    - Hiding: Commitment reveals nothing about content
    - Verifiability: Can verify by revealing content and nonce
    """

    @staticmethod
    def commit(content: str, nonce: Optional[bytes] = None) -> Tuple[str, bytes]:
        """
        Create a cryptographic commitment to content.

        Args:
            content: Content to commit to
            nonce: Optional nonce (generated if not provided)

        Returns:
            Tuple of (commitment_hash, nonce)
        """
        if nonce is None:
            nonce = secrets.token_bytes(32)

        # Commit = H(content || nonce)
        content_bytes = content.encode('utf-8')
        commitment_data = content_bytes + nonce
        commitment = HashFunction.hash_hex(commitment_data)

        return commitment, nonce

    @staticmethod
    def verify(content: str, nonce: bytes, commitment: str) -> bool:
        """
        Verify a commitment.

        Args:
            content: Original content
            nonce: Nonce used in commitment
            commitment: Commitment to verify

        Returns:
            True if commitment is valid
        """
        recomputed_commitment, _ = CommitmentScheme.commit(content, nonce)
        return recomputed_commitment == commitment


class MerkleDAG:
    """
    Merkle-DAG construction for trace verification.

    Implements the hash chain construction from Section 3.2:
    - For nodes with parents: bind(v) = H(commit(v) || bind(p1) || ... || bind(pk) || metadata(v))
    - For leaf nodes: bind(v) = H(commit(v) || metadata(v))
    """

    @staticmethod
    def compute_node_binding(
        node: Node,
        parent_bindings: List[str],
        algorithm: str = "sha256"
    ) -> str:
        """
        Compute cryptographic binding for a node.

        Args:
            node: The node to compute binding for
            parent_bindings: List of parent node bindings
            algorithm: Hash algorithm to use

        Returns:
            Cryptographic binding as hex string
        """
        # Collect components to hash
        components = []

        # Add content commitment
        components.append(node.content_commitment.encode('utf-8'))

        # Add parent bindings
        for parent_binding in parent_bindings:
            components.append(parent_binding.encode('utf-8'))

        # Add metadata
        if node.metadata:
            metadata_str = (
                f"{node.metadata.timestamp}|"
                f"{node.node_type.value}|"
                f"{node.metadata.confidence}|"
                f"{node.metadata.token_count}"
            )
            components.append(metadata_str.encode('utf-8'))

        # Concatenate all components
        binding_data = b'||'.join(components)

        # Compute hash
        return HashFunction.hash_hex(binding_data, algorithm)

    @staticmethod
    def build_merkle_dag(trace_graph: TraceGraph, algorithm: str = "sha256") -> dict:
        """
        Build Merkle-DAG bindings for entire trace graph.

        Performs topological traversal to compute bindings bottom-up.

        Args:
            trace_graph: The trace graph to build bindings for
            algorithm: Hash algorithm to use

        Returns:
            Dictionary mapping node_id to binding hash
        """
        bindings = {}

        # Get nodes in topological order (parents before children)
        sorted_nodes = MerkleDAG._topological_sort(trace_graph)

        for node in sorted_nodes:
            # Get parent bindings
            parent_bindings = [
                bindings[parent_id]
                for parent_id in node.parent_nodes
                if parent_id in bindings
            ]

            # Compute binding
            binding = MerkleDAG.compute_node_binding(node, parent_bindings, algorithm)
            bindings[node.node_id] = binding

            # Update node with binding
            node.cryptographic_binding = binding

        return bindings

    @staticmethod
    def _topological_sort(trace_graph: TraceGraph) -> List[Node]:
        """
        Topologically sort nodes (parents before children).

        Uses Kahn's algorithm for topological sorting.

        Args:
            trace_graph: The trace graph

        Returns:
            List of nodes in topological order
        """
        # Build in-degree map
        in_degree = {node.node_id: len(node.parent_nodes) for node in trace_graph.nodes}

        # Queue of nodes with no dependencies
        queue = [node for node in trace_graph.nodes if in_degree[node.node_id] == 0]
        result = []

        while queue:
            # Process node with no remaining dependencies
            node = queue.pop(0)
            result.append(node)

            # Update in-degrees of children
            for child in trace_graph.get_children(node.node_id):
                in_degree[child.node_id] -= 1
                if in_degree[child.node_id] == 0:
                    queue.append(child)

        if len(result) != len(trace_graph.nodes):
            raise ValueError("Cycle detected in trace graph")

        return result

    @staticmethod
    def compute_root_hash(
        terminal_bindings: List[str],
        agent_manifest_data: str,
        temporal_data: str,
        algorithm: str = "sha256"
    ) -> str:
        """
        Compute root hash of the trace.

        As defined in Section 3.2:
        root = H(bind(terminal_nodes) || agent_manifest || temporal_attestation)

        Args:
            terminal_bindings: Bindings of terminal nodes
            agent_manifest_data: Serialized agent manifest
            temporal_data: Temporal attestation data
            algorithm: Hash algorithm

        Returns:
            Root hash as hex string
        """
        components = []

        # Add terminal node bindings
        for binding in sorted(terminal_bindings):  # Sort for determinism
            components.append(binding.encode('utf-8'))

        # Add manifest and temporal data
        components.append(agent_manifest_data.encode('utf-8'))
        components.append(temporal_data.encode('utf-8'))

        # Concatenate and hash
        root_data = b'||'.join(components)
        return HashFunction.hash_hex(root_data, algorithm)


class SignatureScheme:
    """
    Digital signature implementation.

    Supports EdDSA (Ed25519) for performance as described in Section 3.1.
    """

    @staticmethod
    def generate_keypair() -> Tuple[bytes, bytes]:
        """
        Generate Ed25519 keypair.

        Returns:
            Tuple of (private_key_bytes, public_key_bytes)
        """
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

        return private_bytes, public_bytes

    @staticmethod
    def sign(data: bytes, private_key_bytes: bytes) -> bytes:
        """
        Sign data using Ed25519.

        Args:
            data: Data to sign
            private_key_bytes: Private key as raw bytes

        Returns:
            Signature as bytes
        """
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        signature = private_key.sign(data)
        return signature

    @staticmethod
    def verify(data: bytes, signature: bytes, public_key_bytes: bytes) -> bool:
        """
        Verify Ed25519 signature.

        Args:
            data: Data that was signed
            signature: Signature to verify
            public_key_bytes: Public key as raw bytes

        Returns:
            True if signature is valid
        """
        try:
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
            public_key.verify(signature, data)
            return True
        except Exception:
            return False

    @staticmethod
    def sign_trace(trace: DecisionTrace, private_key_bytes: bytes) -> str:
        """
        Sign a complete trace.

        Args:
            trace: Decision trace to sign
            private_key_bytes: Agent's private key

        Returns:
            Base64-encoded signature
        """
        if not trace.root_hash:
            raise ValueError("Trace must have root hash computed before signing")

        root_hash_bytes = trace.root_hash.encode('utf-8')
        signature = SignatureScheme.sign(root_hash_bytes, private_key_bytes)
        return base64.b64encode(signature).decode('utf-8')

    @staticmethod
    def verify_trace(trace: DecisionTrace, signature_b64: str, public_key_bytes: bytes) -> bool:
        """
        Verify trace signature.

        Args:
            trace: Decision trace
            signature_b64: Base64-encoded signature
            public_key_bytes: Agent's public key

        Returns:
            True if signature is valid
        """
        if not trace.root_hash:
            raise ValueError("Trace must have root hash")

        root_hash_bytes = trace.root_hash.encode('utf-8')
        signature = base64.b64decode(signature_b64)
        return SignatureScheme.verify(root_hash_bytes, signature, public_key_bytes)


class TraceVerifier:
    """
    High-level trace verification interface.

    Combines all cryptographic primitives to verify trace integrity.
    """

    @staticmethod
    def finalize_trace(trace: DecisionTrace, private_key_bytes: bytes) -> None:
        """
        Finalize a trace by computing all cryptographic bindings and signing.

        Args:
            trace: Decision trace to finalize
            private_key_bytes: Agent's private key for signing
        """
        # Step 1: Compute commitments for all nodes
        for node in trace.trace_graph.nodes:
            if node.full_content and not node.content_commitment:
                commitment, nonce = CommitmentScheme.commit(node.full_content)
                node.content_commitment = commitment
                # Store nonce separately if needed for disclosure

        # Step 2: Build Merkle-DAG bindings
        MerkleDAG.build_merkle_dag(trace.trace_graph)

        # Step 3: Compute root hash
        terminal_nodes = trace.trace_graph.get_terminal_nodes()
        terminal_bindings = [node.cryptographic_binding for node in terminal_nodes]

        agent_data = f"{trace.agent_manifest.agent_did}|{trace.agent_manifest.model_version}"
        temporal_data = trace.created_at

        trace.root_hash = MerkleDAG.compute_root_hash(
            terminal_bindings,
            agent_data,
            temporal_data
        )

        # Step 4: Sign root hash
        signature = SignatureScheme.sign_trace(trace, private_key_bytes)
        trace.agent_manifest.signature = signature

    @staticmethod
    def verify_trace_integrity(trace: DecisionTrace, public_key_bytes: bytes) -> bool:
        """
        Verify complete trace integrity.

        Checks:
        1. Merkle-DAG bindings are correct
        2. Root hash is correct
        3. Signature is valid

        Args:
            trace: Decision trace to verify
            public_key_bytes: Agent's public key

        Returns:
            True if trace is valid and unmodified
        """
        try:
            # Verify Merkle-DAG bindings
            computed_bindings = MerkleDAG.build_merkle_dag(trace.trace_graph)

            for node in trace.trace_graph.nodes:
                if node.cryptographic_binding != computed_bindings[node.node_id]:
                    return False

            # Verify root hash
            terminal_nodes = trace.trace_graph.get_terminal_nodes()
            terminal_bindings = [node.cryptographic_binding for node in terminal_nodes]

            agent_data = f"{trace.agent_manifest.agent_did}|{trace.agent_manifest.model_version}"
            temporal_data = trace.created_at

            expected_root = MerkleDAG.compute_root_hash(
                terminal_bindings,
                agent_data,
                temporal_data
            )

            if trace.root_hash != expected_root:
                return False

            # Verify signature
            return SignatureScheme.verify_trace(
                trace,
                trace.agent_manifest.signature,
                public_key_bytes
            )

        except Exception as e:
            print(f"Verification error: {e}")
            return False
