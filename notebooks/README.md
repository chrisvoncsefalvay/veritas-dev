# VERITAS Analysis Notebooks

This directory contains Jupyter notebooks for analyzing, measuring, and visualizing the VERITAS implementation. These notebooks generate the measurements, timings, and visualizations referenced in the paper.

## Notebooks

### 01_performance_benchmarks.ipynb

**Purpose**: Measure and validate the performance claims from Section 4.1 of the paper.

**Measurements**:
- Per-step overhead (target: ~5-10ms)
- Component-level timings (embedding, commitment, binding, signing)
- End-to-end trace generation for various workflow sizes
- Verification performance (O(log n) complexity)
- Scalability analysis

**Outputs**:
- `performance_component_times.png` - Breakdown of per-step overhead
- `performance_scaling.png` - Scaling behavior with workflow size
- `performance_verification.png` - Verification time analysis
- `performance_summary.txt` - Comparison to paper claims

**Key Results**:
- Validates 5-10ms per-step overhead claim
- Demonstrates linear scaling O(n)
- Confirms negligible impact on agent runtime (<2%)

---

### 02_compression_analysis.ipynb

**Purpose**: Analyze semantic compression effectiveness as described in Section 2.3.

**Measurements**:
- Storage size comparison (raw vs. compressed)
- Compression ratios and space savings
- Trade-offs between compression levels
- Semantic similarity preservation after compression

**Outputs**:
- `compression_storage_analysis.png` - Storage size comparisons
- `compression_tradeoffs.png` - Compression level analysis
- `compression_similarity.png` - Semantic preservation metrics
- `compression_summary.txt` - Compression statistics

**Key Results**:
- Validates ~92% space savings claim
- Raw: ~50KB per step → Compressed: ~4KB per step
- Demonstrates semantic structure preservation
- Analyzes 128d to 768d embedding dimensions

---

### 03_trace_visualization.ipynb

**Purpose**: Create visualizations of VERITAS decision traces.

**Visualizations**:
- DAG structure showing reasoning flow
- Merkle-DAG with cryptographic bindings
- Node type distribution analysis
- Detailed flow diagrams with content
- Confidence score distributions

**Outputs**:
- `trace_dag_visualization.png` - Basic DAG structure
- `trace_merkle_dag.png` - Cryptographic binding tree
- `trace_statistics.png` - Statistical analysis
- `trace_detailed_flow.png` - Detailed reasoning flow
- `visualization_summary.txt` - Trace summary

**Use Cases**:
- Paper figures and illustrations
- Understanding trace structure
- Debugging complex reasoning chains
- Presentation materials

---

## Installation

### Basic Installation

```bash
# Install core dependencies
cd ../code
pip install -r requirements.txt

# Install notebook dependencies
cd ../notebooks
pip install -r requirements.txt
```

### Full Installation (with optional features)

```bash
# Install all dependencies including optional visualization tools
pip install -r requirements.txt plotly ipywidgets tqdm

# Enable Jupyter widgets
jupyter nbextension enable --py widgetsnbextension
```

## Usage

### Running All Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Then open and run each notebook in sequence:
1. `01_performance_benchmarks.ipynb`
2. `02_compression_analysis.ipynb`
3. `03_trace_visualization.ipynb`

### Running from Command Line

```bash
# Execute a notebook and save outputs
jupyter nbconvert --to notebook --execute 01_performance_benchmarks.ipynb

# Convert to HTML with outputs
jupyter nbconvert --to html --execute 01_performance_benchmarks.ipynb
```

### Automated Execution

```bash
# Run all notebooks
for notebook in *.ipynb; do
    jupyter nbconvert --to notebook --execute "$notebook" --inplace
done
```

## Generated Outputs

After running all notebooks, you'll have:

### Performance Data
- Component timing breakdowns
- Scaling analysis charts
- Verification performance metrics
- Summary comparisons to paper claims

### Compression Analysis
- Storage size comparisons
- Compression ratio visualizations
- Semantic similarity heatmaps
- Trade-off analysis

### Visualizations
- Publication-quality DAG figures
- Cryptographic structure diagrams
- Statistical distributions
- Interactive trace explorations

## Customization

### Modify Benchmark Parameters

In `01_performance_benchmarks.ipynb`:

```python
# Change workflow sizes
workflow_sizes = [10, 25, 50, 100, 200, 500]  # Add more sizes

# Increase iterations for more accurate measurements
n_iterations = 1000  # Default is 100
```

### Customize Compression Analysis

In `02_compression_analysis.ipynb`:

```python
# Test different compression dimensions
dimensions = [64, 128, 256, 384, 512, 768, 1024]

# Use real embeddings instead of mock
embedder = SemanticEmbedder("all-MiniLM-L6-v2")  # Requires sentence-transformers
```

### Adjust Visualizations

In `03_trace_visualization.ipynb`:

```python
# Change figure size
plt.rcParams['figure.figsize'] = (20, 15)

# Modify color schemes
type_colors = {
    'REASONING': '#your_color_here',
    ...
}

# Adjust DPI for higher quality
plt.savefig('output.png', dpi=600)  # Default is 300
```

## Reproducing Paper Results

To reproduce the exact figures and measurements from the paper:

### 1. Performance Benchmarks (Section 4.1)

```bash
jupyter nbconvert --to notebook --execute 01_performance_benchmarks.ipynb
```

Expected outputs match paper claims:
- Per-step overhead: 5-10ms ✓
- 100-step workflow: 0.5-1s overhead ✓
- Verification: ~10ms + 2ms per node ✓

### 2. Compression Analysis (Section 2.3, 4.1)

```bash
jupyter nbconvert --to notebook --execute 02_compression_analysis.ipynb
```

Expected outputs:
- Space savings: ~92% ✓
- Raw: ~50KB/step → Compressed: ~4KB/step ✓
- Compression ratio: ~12:1 ✓

### 3. Trace Visualizations (Figures in paper)

```bash
jupyter nbconvert --to notebook --execute 03_trace_visualization.ipynb
```

Generates publication-ready figures for:
- Figure 2: Trace DAG structure
- Figure 3: Merkle-DAG bindings
- Figure 4: Node type distribution

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Make sure code directory is in Python path
export PYTHONPATH="${PYTHONPATH}:../code"

# Or install code as package
cd ../code
pip install -e .
```

### Visualization Issues

If graphs don't display:

```bash
# Install backend
pip install ipympl

# Enable in notebook
%matplotlib widget
```

### Memory Issues

For large traces:

```python
# Reduce workflow sizes
workflow_sizes = [10, 25, 50]  # Instead of [10, 25, 50, 100, 200]

# Process in batches
# Or increase available memory
```

## Exporting Results

### Export to PDF

```bash
# Install dependencies
pip install nbconvert[webpdf]

# Convert to PDF
jupyter nbconvert --to pdf 01_performance_benchmarks.ipynb
```

### Export All Figures

```bash
# All PNG files are automatically saved to notebooks directory
ls -lh *.png

# Copy to paper directory
cp *.png ../paper/figures/
```

### Export Data Tables

Results are also saved as text files:
- `performance_summary.txt`
- `compression_summary.txt`
- `visualization_summary.txt`

These can be copied directly into the paper.

## Performance Tips

### 1. Use Mock Embeddings for Speed

Mock embeddings are much faster for benchmarking:

```python
embedder = SemanticEmbedder('mock')  # Fast, deterministic
# vs
embedder = SemanticEmbedder('all-MiniLM-L6-v2')  # Slow, realistic
```

### 2. Reduce Iterations

For quick testing:

```python
n_iterations = 10  # Fast testing
# vs
n_iterations = 1000  # Accurate measurements
```

### 3. Parallel Execution

Some cells can run in parallel:

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(benchmark_function)(size) for size in sizes
)
```

## Contributing

To add new notebooks:

1. Follow naming convention: `XX_descriptive_name.ipynb`
2. Include markdown documentation
3. Save outputs in notebook
4. Update this README
5. Add to automated testing if applicable

## Citation

If you use these notebooks in your research:

```bibtex
@article{veritas2025,
  title={VERITAS: Cryptographically Verifiable Semantic Traces for AI Agent Provenance},
  author={von Csefalvay, Chris and Amiribesheli, Mohsen},
  year={2025}
}
```

## License

Same license as the main VERITAS project.

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/veritas-dev/issues
- Email: kristof.csefalvay@hcltech.com
