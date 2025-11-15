"""
VERITAS Core Data Structures
Implementation of the core decision trace format as described in the VERITAS paper.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class NodeType(Enum):
    """Classification of reasoning step nodes."""
    REASONING = "REASONING"
    TOOL_CALL = "TOOL_CALL"
    OBSERVATION = "OBSERVATION"
    DECISION = "DECISION"
    MEMORY_ACCESS = "MEMORY_ACCESS"


class RelationType(Enum):
    """Types of edges between nodes in the trace DAG."""
    CAUSAL = "CAUSAL"
    INFORMATIONAL = "INFORMATIONAL"
    TEMPORAL = "TEMPORAL"


@dataclass
class NodeMetadata:
    """
    Structured metadata for a reasoning step node.

    Attributes:
        timestamp: ISO 8601 timestamp of when the node was created
        confidence: Confidence score in range [0, 1]
        token_count: Number of tokens in the reasoning step
        temperature: Model temperature parameter used
        model_id: Identifier of the model that generated this step
    """
    timestamp: str
    confidence: float = 1.0
    token_count: int = 0
    temperature: float = 0.0
    model_id: str = ""

    def __post_init__(self):
        """Validate metadata values."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be in range [0, 1]")
        if self.token_count < 0:
            raise ValueError("Token count must be non-negative")


@dataclass
class Node:
    """
    A reasoning step node in the VERITAS trace DAG.

    As defined in the paper (Section 2.2), each node has:
    - id: Globally unique identifier (UUIDv4)
    - type: Node classification
    - sem: Semantic embedding vector
    - meta: Structured metadata
    - commit: Cryptographic commitment to node content
    - parents: List of parent node IDs
    - bind: Cryptographic binding computed from commit and parent bindings
    """
    node_id: str
    node_type: NodeType
    semantic_embedding: Optional[List[float]] = None
    metadata: Optional[NodeMetadata] = None
    content_commitment: str = ""
    parent_nodes: List[str] = field(default_factory=list)
    cryptographic_binding: str = ""

    # Optional: full content (not stored in compressed traces)
    full_content: Optional[str] = None

    def __post_init__(self):
        """Initialize defaults."""
        if self.metadata is None:
            self.metadata = NodeMetadata(
                timestamp=datetime.utcnow().isoformat() + "Z"
            )


@dataclass
class Edge:
    """
    A directed edge in the trace DAG encoding dependencies.

    Attributes:
        from_node: Source node ID
        to_node: Destination node ID
        relation_type: Type of relationship
        weight: Edge weight/strength in range [0, 1]
    """
    from_node: str
    to_node: str
    relation_type: RelationType = RelationType.CAUSAL
    weight: float = 1.0

    def __post_init__(self):
        """Validate edge values."""
        if not 0 <= self.weight <= 1:
            raise ValueError("Edge weight must be in range [0, 1]")


@dataclass
class AgentManifest:
    """
    Agent identity and metadata for the trace.

    Attributes:
        agent_did: Decentralized identifier for the agent
        model_version: Version of the model
        framework: Framework used (e.g., "custom", "langchain", "autogpt")
        public_key: Base64-encoded public key for signature verification
        signature: Base64-encoded signature of root hash
    """
    agent_did: str
    model_version: str
    framework: str = "custom"
    public_key: str = ""
    signature: str = ""


@dataclass
class TemporalAttestation:
    """
    RFC 3161 timestamp token for temporal binding.

    Attributes:
        rfc3161_token: Base64-encoded RFC 3161 timestamp token
        tsa_url: URL of the timestamp authority used
        timestamp: Human-readable timestamp
    """
    rfc3161_token: str = ""
    tsa_url: str = ""
    timestamp: str = ""


@dataclass
class TraceGraph:
    """
    The DAG structure of the reasoning trace.

    Attributes:
        nodes: List of reasoning step nodes
        edges: List of directed edges between nodes
    """
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        """Add a node to the trace graph."""
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the trace graph."""
        # Validate that nodes exist
        node_ids = {n.node_id for n in self.nodes}
        if edge.from_node not in node_ids:
            raise ValueError(f"Source node {edge.from_node} not found in graph")
        if edge.to_node not in node_ids:
            raise ValueError(f"Destination node {edge.to_node} not found in graph")
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_parents(self, node_id: str) -> List[Node]:
        """Get all parent nodes of a given node."""
        node = self.get_node(node_id)
        if node is None:
            return []
        return [self.get_node(pid) for pid in node.parent_nodes if self.get_node(pid)]

    def get_children(self, node_id: str) -> List[Node]:
        """Get all child nodes of a given node."""
        children = []
        for edge in self.edges:
            if edge.from_node == node_id:
                child = self.get_node(edge.to_node)
                if child:
                    children.append(child)
        return children

    def get_root_nodes(self) -> List[Node]:
        """Get all root nodes (nodes with no parents)."""
        return [node for node in self.nodes if not node.parent_nodes]

    def get_terminal_nodes(self) -> List[Node]:
        """Get all terminal/leaf nodes (nodes with no children)."""
        nodes_with_children = {edge.from_node for edge in self.edges}
        return [node for node in self.nodes if node.node_id not in nodes_with_children]


@dataclass
class DecisionTrace:
    """
    A complete VERITAS decision trace.

    As defined in the paper (Section 2), a trace T = (V, E, M, Σ, τ) where:
    - V: Set of reasoning step nodes
    - E: Set of directed edges encoding dependencies
    - M: Agent manifest
    - Σ: Set of cryptographic signatures
    - τ: Timestamp attestation

    Attributes:
        veritas_version: Version of the VERITAS format
        trace_id: Unique identifier for this trace
        created_at: ISO 8601 timestamp of trace creation
        agent_manifest: Agent identity and metadata
        trace_graph: The DAG structure of reasoning steps
        root_hash: SHA-256 hash of the complete trace
        temporal_attestation: RFC 3161 timestamp token
    """
    veritas_version: str = "1.0"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    agent_manifest: Optional[AgentManifest] = None
    trace_graph: TraceGraph = field(default_factory=TraceGraph)
    root_hash: str = ""
    temporal_attestation: Optional[TemporalAttestation] = None

    def add_reasoning_step(
        self,
        content: str,
        node_type: NodeType = NodeType.REASONING,
        parent_ids: Optional[List[str]] = None,
        confidence: float = 1.0,
        **kwargs
    ) -> Node:
        """
        Add a new reasoning step to the trace.

        Args:
            content: The full content of the reasoning step
            node_type: Type of the node
            parent_ids: List of parent node IDs
            confidence: Confidence score for this step
            **kwargs: Additional metadata fields

        Returns:
            The created Node
        """
        node_id = str(uuid.uuid4())
        metadata = NodeMetadata(
            timestamp=datetime.utcnow().isoformat() + "Z",
            confidence=confidence,
            **kwargs
        )

        node = Node(
            node_id=node_id,
            node_type=node_type,
            metadata=metadata,
            parent_nodes=parent_ids or [],
            full_content=content
        )

        self.trace_graph.add_node(node)

        # Create edges from parents
        if parent_ids:
            for parent_id in parent_ids:
                edge = Edge(
                    from_node=parent_id,
                    to_node=node_id,
                    relation_type=RelationType.CAUSAL
                )
                self.trace_graph.add_edge(edge)

        return node

    def get_trace_depth(self) -> int:
        """Calculate the maximum depth of the trace DAG."""
        def get_depth(node_id: str, visited: set) -> int:
            if node_id in visited:
                return 0
            visited.add(node_id)

            children = self.trace_graph.get_children(node_id)
            if not children:
                return 1

            return 1 + max(get_depth(child.node_id, visited.copy()) for child in children)

        roots = self.trace_graph.get_root_nodes()
        if not roots:
            return 0

        return max(get_depth(root.node_id, set()) for root in roots)

    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get statistics about the trace."""
        return {
            "total_nodes": len(self.trace_graph.nodes),
            "total_edges": len(self.trace_graph.edges),
            "root_nodes": len(self.trace_graph.get_root_nodes()),
            "terminal_nodes": len(self.trace_graph.get_terminal_nodes()),
            "max_depth": self.get_trace_depth(),
            "node_types": {
                node_type.value: sum(1 for n in self.trace_graph.nodes if n.node_type == node_type)
                for node_type in NodeType
            }
        }
