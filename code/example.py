"""
VERITAS Reference Implementation - Example Usage

This example demonstrates the complete VERITAS workflow:
1. Creating a decision trace
2. Adding reasoning steps
3. Computing semantic embeddings
4. Generating cryptographic proofs
5. Serializing to JSON
6. Verifying trace integrity
"""

import base64
from core import DecisionTrace, NodeType, AgentManifest
from crypto import SignatureScheme, TraceVerifier
from compression import TraceCompressor, SemanticEmbedder, analyze_compression_efficiency
from serialization import TraceSerializer, SchemaValidator


def create_example_trace():
    """
    Create an example decision trace for a medical diagnosis scenario.

    This simulates an AI agent making a medical recommendation through
    multiple reasoning steps.
    """
    print("=" * 60)
    print("VERITAS Decision Trace - Example")
    print("=" * 60)
    print()

    # Step 1: Generate agent keypair
    print("Step 1: Generating agent keypair...")
    private_key, public_key = SignatureScheme.generate_keypair()
    public_key_b64 = base64.b64encode(public_key).decode('utf-8')
    print(f"  ✓ Public key: {public_key_b64[:32]}...")
    print()

    # Step 2: Create trace with agent manifest
    print("Step 2: Creating decision trace...")
    trace = DecisionTrace()
    trace.agent_manifest = AgentManifest(
        agent_did="did:agent:example:medical-advisor-v1",
        model_version="gpt-4-2024-01-01",
        framework="custom",
        public_key=public_key_b64
    )
    print(f"  ✓ Trace ID: {trace.trace_id}")
    print(f"  ✓ Agent: {trace.agent_manifest.agent_did}")
    print()

    # Step 3: Add reasoning steps
    print("Step 3: Adding reasoning steps...")

    # Initial observation
    n1 = trace.add_reasoning_step(
        content="Patient presents with persistent cough for 2 weeks, fever (101°F), and fatigue.",
        node_type=NodeType.OBSERVATION,
        confidence=0.95,
        token_count=25
    )
    print(f"  ✓ Added OBSERVATION node: {n1.node_id[:8]}")

    # Memory access - retrieve similar cases
    n2 = trace.add_reasoning_step(
        content="Accessing patient history database for similar symptom patterns in past 6 months.",
        node_type=NodeType.MEMORY_ACCESS,
        parent_ids=[n1.node_id],
        confidence=0.90,
        token_count=18
    )
    print(f"  ✓ Added MEMORY_ACCESS node: {n2.node_id[:8]}")

    # Reasoning step 1
    n3 = trace.add_reasoning_step(
        content="Symptoms align with respiratory infection. Differential diagnosis includes: "
                "1) Bacterial pneumonia (40% likelihood), 2) Viral bronchitis (35% likelihood), "
                "3) Mycoplasma pneumonia (15% likelihood), 4) Other (10%).",
        node_type=NodeType.REASONING,
        parent_ids=[n1.node_id, n2.node_id],
        confidence=0.85,
        token_count=67
    )
    print(f"  ✓ Added REASONING node: {n3.node_id[:8]}")

    # Tool call - order lab test
    n4 = trace.add_reasoning_step(
        content="Tool invocation: order_lab_test(test='chest_xray', test='complete_blood_count', "
                "priority='routine')",
        node_type=NodeType.TOOL_CALL,
        parent_ids=[n3.node_id],
        confidence=0.92,
        token_count=22
    )
    print(f"  ✓ Added TOOL_CALL node: {n4.node_id[:8]}")

    # Observation from tool
    n5 = trace.add_reasoning_step(
        content="Lab results received: Chest X-ray shows infiltrate in right lower lobe. "
                "CBC shows elevated WBC count (15,000/μL) with left shift.",
        node_type=NodeType.OBSERVATION,
        parent_ids=[n4.node_id],
        confidence=0.98,
        token_count=35
    )
    print(f"  ✓ Added OBSERVATION node: {n5.node_id[:8]}")

    # Reasoning step 2
    n6 = trace.add_reasoning_step(
        content="X-ray findings and lab results strongly suggest bacterial pneumonia. "
                "Elevated WBC with left shift indicates bacterial infection. "
                "Likelihood of bacterial pneumonia increased to 85%.",
        node_type=NodeType.REASONING,
        parent_ids=[n3.node_id, n5.node_id],
        confidence=0.90,
        token_count=52
    )
    print(f"  ✓ Added REASONING node: {n6.node_id[:8]}")

    # Final decision
    n7 = trace.add_reasoning_step(
        content="DECISION: Recommend antibiotic therapy with amoxicillin-clavulanate 875mg twice daily "
                "for 7 days. Schedule follow-up in 3 days. Patient should seek immediate care if "
                "symptoms worsen or breathing difficulties develop.",
        node_type=NodeType.DECISION,
        parent_ids=[n6.node_id],
        confidence=0.88,
        token_count=58
    )
    print(f"  ✓ Added DECISION node: {n7.node_id[:8]}")
    print()

    # Step 4: Compute semantic embeddings
    print("Step 4: Computing semantic embeddings...")
    compressor = TraceCompressor()
    compressor.compress_trace(trace)
    print(f"  ✓ Compressed {len(trace.trace_graph.nodes)} nodes")

    # Analyze compression
    compression_stats = analyze_compression_efficiency(trace)
    print(f"  ✓ Compression ratio: {compression_stats['compression_ratio']:.1f}x")
    print(f"  ✓ Space savings: {compression_stats['space_savings_percent']:.1f}%")
    print()

    # Step 5: Finalize trace (compute cryptographic proofs)
    print("Step 5: Computing cryptographic proofs...")
    TraceVerifier.finalize_trace(trace, private_key)
    print(f"  ✓ Computed Merkle-DAG bindings for {len(trace.trace_graph.nodes)} nodes")
    print(f"  ✓ Root hash: {trace.root_hash[:32]}...")
    print(f"  ✓ Signature: {trace.agent_manifest.signature[:32]}...")
    print()

    # Step 6: Validate trace
    print("Step 6: Validating trace structure...")
    is_valid, errors = SchemaValidator.validate_trace(trace)
    if is_valid:
        print("  ✓ Trace is valid")
    else:
        print("  ✗ Trace validation failed:")
        for error in errors:
            print(f"    - {error}")
    print()

    # Step 7: Verify cryptographic integrity
    print("Step 7: Verifying cryptographic integrity...")
    is_verified = TraceVerifier.verify_trace_integrity(trace, public_key)
    if is_verified:
        print("  ✓ Trace integrity verified")
    else:
        print("  ✗ Trace integrity verification failed")
    print()

    # Step 8: Display trace statistics
    print("Step 8: Trace statistics...")
    stats = trace.get_trace_statistics()
    print(f"  • Total nodes: {stats['total_nodes']}")
    print(f"  • Total edges: {stats['total_edges']}")
    print(f"  • Max depth: {stats['max_depth']}")
    print(f"  • Root nodes: {stats['root_nodes']}")
    print(f"  • Terminal nodes: {stats['terminal_nodes']}")
    print(f"  • Node types:")
    for node_type, count in stats['node_types'].items():
        if count > 0:
            print(f"    - {node_type}: {count}")
    print()

    return trace, public_key


def demonstrate_serialization(trace):
    """Demonstrate JSON serialization."""
    print("=" * 60)
    print("JSON Serialization")
    print("=" * 60)
    print()

    # Serialize to JSON
    print("Serializing trace to JSON...")
    json_str = TraceSerializer.trace_to_json(trace, include_full_content=False)
    print(f"  ✓ JSON size: {len(json_str)} bytes")
    print()

    # Save to file
    output_file = "/tmp/veritas_example_trace.json"
    TraceSerializer.save_trace(trace, output_file, include_full_content=True)
    print(f"  ✓ Saved to: {output_file}")
    print()

    # Show excerpt
    print("JSON excerpt (first 500 characters):")
    print("-" * 60)
    print(json_str[:500] + "...")
    print("-" * 60)
    print()

    # Deserialize
    print("Deserializing trace from JSON...")
    loaded_trace = TraceSerializer.load_trace(output_file)
    print(f"  ✓ Loaded trace: {loaded_trace.trace_id}")
    print(f"  ✓ Nodes: {len(loaded_trace.trace_graph.nodes)}")
    print()

    return loaded_trace


def demonstrate_tampering_detection(trace, public_key):
    """Demonstrate that tampering is detected."""
    print("=" * 60)
    print("Tampering Detection")
    print("=" * 60)
    print()

    # Verify original trace
    print("Verifying original trace...")
    is_valid = TraceVerifier.verify_trace_integrity(trace, public_key)
    print(f"  Original trace: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print()

    # Tamper with a node
    print("Tampering with a node (changing content)...")
    original_content = trace.trace_graph.nodes[2].full_content
    trace.trace_graph.nodes[2].full_content = "TAMPERED CONTENT"

    # Update commitment to reflect tampering
    from crypto import CommitmentScheme
    new_commit, _ = CommitmentScheme.commit(trace.trace_graph.nodes[2].full_content)
    trace.trace_graph.nodes[2].content_commitment = new_commit

    # Try to verify tampered trace
    print("Verifying tampered trace...")
    is_valid = TraceVerifier.verify_trace_integrity(trace, public_key)
    print(f"  Tampered trace: {'✓ VALID' if is_valid else '✗ INVALID (detected!)'}")
    print()

    # Restore original
    trace.trace_graph.nodes[2].full_content = original_content


def main():
    """Run the complete example."""
    # Create and verify trace
    trace, public_key = create_example_trace()

    # Demonstrate serialization
    loaded_trace = demonstrate_serialization(trace)

    # Demonstrate tampering detection
    demonstrate_tampering_detection(trace, public_key)

    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Examine the generated JSON file: /tmp/veritas_example_trace.json")
    print("  2. Explore the code in core.py, crypto.py, and compression.py")
    print("  3. Integrate VERITAS into your own agent framework")
    print()


if __name__ == "__main__":
    main()
