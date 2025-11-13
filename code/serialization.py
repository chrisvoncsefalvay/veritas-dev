"""
VERITAS JSON Serialization
Implementation of JSON schema and serialization for VERITAS traces.

Implements the JSON schema from Section 2.4 of the paper.
"""

import json
from typing import Dict, Any
from datetime import datetime

from core import (
    DecisionTrace, Node, Edge, TraceGraph,
    AgentManifest, TemporalAttestation, NodeMetadata,
    NodeType, RelationType
)


class VeritasEncoder(json.JSONEncoder):
    """Custom JSON encoder for VERITAS objects."""

    def default(self, obj):
        """Convert VERITAS objects to JSON-serializable format."""
        if isinstance(obj, NodeType) or isinstance(obj, RelationType):
            return obj.value
        return super().default(obj)


class TraceSerializer:
    """
    Serialization and deserialization for VERITAS traces.

    Handles conversion between Python objects and JSON format.
    """

    @staticmethod
    def node_to_dict(node: Node) -> Dict[str, Any]:
        """
        Convert Node to dictionary.

        Args:
            node: Node to serialize

        Returns:
            Dictionary representation
        """
        return {
            "node_id": node.node_id,
            "node_type": node.node_type.value,
            "semantic_embedding": node.semantic_embedding,
            "metadata": {
                "timestamp": node.metadata.timestamp,
                "confidence": node.metadata.confidence,
                "token_count": node.metadata.token_count,
                "temperature": node.metadata.temperature,
                "model_id": node.metadata.model_id
            } if node.metadata else None,
            "content_commitment": node.content_commitment,
            "parent_nodes": node.parent_nodes,
            "cryptographic_binding": node.cryptographic_binding
        }

    @staticmethod
    def edge_to_dict(edge: Edge) -> Dict[str, Any]:
        """
        Convert Edge to dictionary.

        Args:
            edge: Edge to serialize

        Returns:
            Dictionary representation
        """
        return {
            "from": edge.from_node,
            "to": edge.to_node,
            "relation_type": edge.relation_type.value,
            "weight": edge.weight
        }

    @staticmethod
    def trace_to_dict(trace: DecisionTrace, include_full_content: bool = False) -> Dict[str, Any]:
        """
        Convert DecisionTrace to dictionary.

        Args:
            trace: Trace to serialize
            include_full_content: Whether to include full node content (for storage/debugging)

        Returns:
            Dictionary representation following VERITAS JSON schema
        """
        nodes_data = []
        for node in trace.trace_graph.nodes:
            node_dict = TraceSerializer.node_to_dict(node)
            if include_full_content and node.full_content:
                node_dict["full_content"] = node.full_content
            nodes_data.append(node_dict)

        result = {
            "veritas_version": trace.veritas_version,
            "trace_id": trace.trace_id,
            "created_at": trace.created_at,
        }

        # Agent manifest
        if trace.agent_manifest:
            result["agent_manifest"] = {
                "agent_did": trace.agent_manifest.agent_did,
                "model_version": trace.agent_manifest.model_version,
                "framework": trace.agent_manifest.framework,
                "public_key": trace.agent_manifest.public_key,
                "signature": trace.agent_manifest.signature
            }

        # Trace graph
        result["trace_graph"] = {
            "nodes": nodes_data,
            "edges": [TraceSerializer.edge_to_dict(e) for e in trace.trace_graph.edges]
        }

        # Root hash
        result["root_hash"] = trace.root_hash

        # Temporal attestation
        if trace.temporal_attestation:
            result["temporal_attestation"] = {
                "rfc3161_token": trace.temporal_attestation.rfc3161_token,
                "tsa_url": trace.temporal_attestation.tsa_url,
                "timestamp": trace.temporal_attestation.timestamp
            }

        return result

    @staticmethod
    def trace_to_json(trace: DecisionTrace, include_full_content: bool = False, indent: int = 2) -> str:
        """
        Convert DecisionTrace to JSON string.

        Args:
            trace: Trace to serialize
            include_full_content: Whether to include full node content
            indent: JSON indentation level

        Returns:
            JSON string
        """
        trace_dict = TraceSerializer.trace_to_dict(trace, include_full_content)
        return json.dumps(trace_dict, indent=indent, cls=VeritasEncoder)

    @staticmethod
    def dict_to_node(data: Dict[str, Any]) -> Node:
        """
        Convert dictionary to Node.

        Args:
            data: Dictionary representation

        Returns:
            Node object
        """
        metadata = None
        if data.get("metadata"):
            meta_dict = data["metadata"]
            metadata = NodeMetadata(
                timestamp=meta_dict["timestamp"],
                confidence=meta_dict.get("confidence", 1.0),
                token_count=meta_dict.get("token_count", 0),
                temperature=meta_dict.get("temperature", 0.0),
                model_id=meta_dict.get("model_id", "")
            )

        return Node(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            semantic_embedding=data.get("semantic_embedding"),
            metadata=metadata,
            content_commitment=data.get("content_commitment", ""),
            parent_nodes=data.get("parent_nodes", []),
            cryptographic_binding=data.get("cryptographic_binding", ""),
            full_content=data.get("full_content")
        )

    @staticmethod
    def dict_to_edge(data: Dict[str, Any]) -> Edge:
        """
        Convert dictionary to Edge.

        Args:
            data: Dictionary representation

        Returns:
            Edge object
        """
        return Edge(
            from_node=data["from"],
            to_node=data["to"],
            relation_type=RelationType(data.get("relation_type", "CAUSAL")),
            weight=data.get("weight", 1.0)
        )

    @staticmethod
    def dict_to_trace(data: Dict[str, Any]) -> DecisionTrace:
        """
        Convert dictionary to DecisionTrace.

        Args:
            data: Dictionary representation

        Returns:
            DecisionTrace object
        """
        trace = DecisionTrace(
            veritas_version=data.get("veritas_version", "1.0"),
            trace_id=data.get("trace_id", ""),
            created_at=data.get("created_at", "")
        )

        # Parse agent manifest
        if "agent_manifest" in data:
            manifest_data = data["agent_manifest"]
            trace.agent_manifest = AgentManifest(
                agent_did=manifest_data["agent_did"],
                model_version=manifest_data["model_version"],
                framework=manifest_data.get("framework", "custom"),
                public_key=manifest_data.get("public_key", ""),
                signature=manifest_data.get("signature", "")
            )

        # Parse trace graph
        if "trace_graph" in data:
            graph_data = data["trace_graph"]

            # Parse nodes
            for node_data in graph_data.get("nodes", []):
                node = TraceSerializer.dict_to_node(node_data)
                trace.trace_graph.add_node(node)

            # Parse edges
            for edge_data in graph_data.get("edges", []):
                edge = TraceSerializer.dict_to_edge(edge_data)
                trace.trace_graph.add_edge(edge)

        # Parse root hash
        trace.root_hash = data.get("root_hash", "")

        # Parse temporal attestation
        if "temporal_attestation" in data:
            attest_data = data["temporal_attestation"]
            trace.temporal_attestation = TemporalAttestation(
                rfc3161_token=attest_data.get("rfc3161_token", ""),
                tsa_url=attest_data.get("tsa_url", ""),
                timestamp=attest_data.get("timestamp", "")
            )

        return trace

    @staticmethod
    def json_to_trace(json_str: str) -> DecisionTrace:
        """
        Convert JSON string to DecisionTrace.

        Args:
            json_str: JSON string

        Returns:
            DecisionTrace object
        """
        data = json.loads(json_str)
        return TraceSerializer.dict_to_trace(data)

    @staticmethod
    def save_trace(trace: DecisionTrace, filepath: str, include_full_content: bool = False) -> None:
        """
        Save trace to JSON file.

        Args:
            trace: Trace to save
            filepath: Output file path
            include_full_content: Whether to include full node content
        """
        json_str = TraceSerializer.trace_to_json(trace, include_full_content, indent=2)
        with open(filepath, 'w') as f:
            f.write(json_str)

    @staticmethod
    def load_trace(filepath: str) -> DecisionTrace:
        """
        Load trace from JSON file.

        Args:
            filepath: Input file path

        Returns:
            DecisionTrace object
        """
        with open(filepath, 'r') as f:
            json_str = f.read()
        return TraceSerializer.json_to_trace(json_str)


class SchemaValidator:
    """
    Validates VERITAS traces against the JSON schema.

    Provides basic validation - for production use, consider using
    a JSON Schema validator library.
    """

    @staticmethod
    def validate_trace(trace: DecisionTrace) -> tuple[bool, list[str]]:
        """
        Validate a trace object.

        Args:
            trace: Trace to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check version
        if not trace.veritas_version:
            errors.append("Missing veritas_version")

        # Check trace ID
        if not trace.trace_id:
            errors.append("Missing trace_id")

        # Check created_at
        if not trace.created_at:
            errors.append("Missing created_at")

        # Check agent manifest
        if not trace.agent_manifest:
            errors.append("Missing agent_manifest")
        else:
            if not trace.agent_manifest.agent_did:
                errors.append("Missing agent_manifest.agent_did")
            if not trace.agent_manifest.model_version:
                errors.append("Missing agent_manifest.model_version")

        # Check nodes
        if not trace.trace_graph.nodes:
            errors.append("Trace has no nodes")

        for node in trace.trace_graph.nodes:
            if not node.node_id:
                errors.append(f"Node missing node_id")
            if not node.node_type:
                errors.append(f"Node {node.node_id} missing node_type")
            if node.metadata and not (0 <= node.metadata.confidence <= 1):
                errors.append(f"Node {node.node_id} has invalid confidence")

        # Check edges reference valid nodes
        node_ids = {n.node_id for n in trace.trace_graph.nodes}
        for edge in trace.trace_graph.edges:
            if edge.from_node not in node_ids:
                errors.append(f"Edge references non-existent from_node: {edge.from_node}")
            if edge.to_node not in node_ids:
                errors.append(f"Edge references non-existent to_node: {edge.to_node}")

        # Check for cycles (DAG property)
        try:
            from crypto import MerkleDAG
            MerkleDAG._topological_sort(trace.trace_graph)
        except ValueError as e:
            errors.append(f"Graph validation error: {str(e)}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_json(json_str: str) -> tuple[bool, list[str]]:
        """
        Validate JSON string against VERITAS schema.

        Args:
            json_str: JSON string to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            trace = TraceSerializer.json_to_trace(json_str)
            return SchemaValidator.validate_trace(trace)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {str(e)}"]
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
