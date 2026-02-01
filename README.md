# HDF5 Tools

![Claude Code Opus 4.5 coded | Unreviewed](https://img.shields.io/badge/Claude%20Code%20Opus%204.5%20coded-Unreviewed-grey?logo=claude&logoColor=white&labelColor=D97757)

Comprehensive tooling for HDF5 file inspection, analysis, validation, and interactive editing.

## Tools Overview

### print_hdf5.py
Command-line tool for **reading and analyzing** HDF5 files:
- File structure inspection
- Quality analysis (NaN/Inf detection)
- Statistics computation (min/max/mean)
- Schema validation
- File comparison
- Histogram generation
- Batch processing

### edit_hdf5.py
Interactive terminal editor for **modifying** HDF5 files:
- Tree visualization with color-coded types
- Add/Remove/Edit datasets, groups, attributes
- Vector/Matrix operations (Cut, Subsample, Random sample)
- Multi-dimensional data viewing
- Configuration management
- Automatic backup on modifications

### visualize_hdf5.py
Graphical visualization tool for **understanding** HDF5 file structure:
- Hierarchical PDF graph of file structure
- Color-coded nodes for groups, datasets, virtual datasets, and links
- Complete metadata display (shape, dtype, compression, attributes)
- Support for all HDF5 features (hard/soft/external links, virtual datasets, dimension scales)
- Detailed storage information (chunking, compression, resizable dimensions)

## Programmatic API

```python
from hdf5.print_hdf5 import validate, analyze, check_quality, inspect, generate_schema

# Validate against schema - returns (passed: bool, errors: list)
passed, errors = validate('data.hdf5', 'schema.json', print_output=False)
if not passed:
    print(f"Validation failed with {len(errors)} errors")
    for err in errors:
        print(f"  {err.path}: {err.message}")

# Full analysis with schema validation and histogram
result = analyze('data.hdf5',
                 validate_values=True,    # Check for NaN/Inf
                 compute_stats=True,      # Compute min/max/mean
                 histogram_bins=10,       # Generate 10-bin histogram
                 schema='schema.json',    # Optional schema
                 print_output=False)

if not result.ok:
    print("Analysis failed")
    if result.has_file_errors:
        print("  File read errors:", result.file_info.errors)
    if result.has_quality_issues:
        print("  Quality issues:", result.quality_issues)
    if result.has_validation_errors:
        print("  Schema errors:", result.validation_errors)

# Check quality only
passed, issues = check_quality('data.hdf5', print_output=False)

# Inspect structure
file_info = inspect('data.hdf5', print_output=False)
print(f"Datasets: {list(file_info.datasets.keys())}")

# Generate schema from existing file
schema = generate_schema('data.hdf5')
with open('schema.json', 'w') as f:
    json.dump(schema, f, indent=2)
```

## CLI Usage

### print_hdf5.py - Inspection and Analysis

```bash
# Inspect file structure
python hdf5/print_hdf5.py inspect data.hdf5

# Inspect and print specific fields
python hdf5/print_hdf5.py inspect data.hdf5 --print body_names robot_colors

# Analyze with statistics
python hdf5/print_hdf5.py analyze data.hdf5 --stats --validate

# Analyze with histogram (10 bins)
python hdf5/print_hdf5.py analyze data.hdf5 --stats --histogram 10

# Analyze excluding specific datasets from histogram
python hdf5/print_hdf5.py analyze data.hdf5 --stats --histogram 10 --exclude heightmap

# Validate against schema
python hdf5/print_hdf5.py validate data.hdf5 --schema schema.json

# Compare multiple files
python hdf5/print_hdf5.py compare file1.hdf5 file2.hdf5

# Generate schema template from existing file
python hdf5/print_hdf5.py schema-template data.hdf5 -o schema.json

# Batch analysis
python hdf5/print_hdf5.py analyze --dir data/ --pattern "*.hdf5" --schema schema.json
```

### edit_hdf5.py - Interactive Editing

```bash
# Open file in interactive editor
python hdf5/edit_hdf5.py data.hdf5

# Key Bindings:
#   Enter   - Expand/collapse node or edit value
#   a       - Add new dataset/group/attribute
#   d       - Delete selected item
#   e       - Edit selected item
#   v       - View data (new tab)
#   c       - Cut operation (axis slicing)
#   s       - Subsample operation
#   r       - Random sample operation
#   C       - Open config tab
#   F       - Toggle fullscreen for operation panel
#   q       - Quit
#   Ctrl+S  - Save changes
#   ?       - Show help
```

### visualize_hdf5.py - Structure Visualization

```bash
# Generate default visualization
python visualize_hdf5.py data.hdf5

# Generate with custom output name
python visualize_hdf5.py data.hdf5 -o my_visualization

# The output is a PDF with hierarchical graph showing:
#   - Light blue boxes: Groups
#   - Light green boxes: Datasets
#   - Plum boxes: Virtual datasets
#   - Yellow dashed boxes: Soft links
#   - Red dotted boxes: External links
#   - Shape, dtype, storage info, attributes displayed inline
#   - All links and references visualized with appropriate edge styles
```

## API Reference (print_hdf5.py)

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `validate(path, schema)` | `(bool, List[ValidationError])` | Validate against schema |
| `analyze(path, ...)` | `AnalysisResult` | Full analysis with statistics/histograms |
| `check_quality(path)` | `(bool, List[Dict])` | Check for NaN/Inf/zero-dim |
| `inspect(path)` | `FileInfo` | Get file structure |
| `generate_schema(path)` | `Dict` | Generate schema template |

### AnalysisResult Properties

| Property | Type | Description |
|----------|------|-------------|
| `.ok` | `bool` | True if no errors/issues |
| `.has_file_errors` | `bool` | True if file read failed |
| `.has_quality_issues` | `bool` | True if NaN/Inf found |
| `.has_validation_errors` | `bool` | True if schema failed |
| `.file_info` | `FileInfo` | File structure |
| `.statistics` | `Dict` | Computed statistics (min/max/mean) |
| `.quality_issues` | `List[Dict]` | Quality issue details |
| `.validation_errors` | `List[ValidationError]` | Schema errors |
| `.histograms` | `Dict` | Histogram bin counts and ranges |

### analyze() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hdf5_path` | Path | - | Path to HDF5 file |
| `validate_values` | bool | False | Check for NaN/Inf values |
| `compute_stats` | bool | False | Compute min/max/mean |
| `histogram_bins` | int | None | Number of bins for histogram (None to skip) |
| `exclude_datasets` | List[str] | None | Datasets to exclude from stats/histogram |
| `schema` | Dict/Path | None | JSON schema for validation |
| `print_output` | bool | True | Print results to stdout |

## Schema Format

```json
{
    "allow_extra_datasets": true,
    "allow_extra_groups": true,
    "allow_extra_attributes": true,

    "datasets": {
        "data": {
            "required": true,
            "dtype": "float32",
            "ndim": 3,
            "shape": [null, 200, 100],
            "min_shape": [1, 200, 100],
            "max_shape": [10000, 200, 100],
            "attributes": {
                "min": {"required": false, "type": "float"}
            }
        }
    },

    "groups": {
        "metadata": {
            "required": false,
            "datasets": { ... }
        }
    }
}
```

### Shape Specification

- `null` - Any size allowed for this dimension
- `integer` - Exact size required
- `min_shape` - Minimum size per dimension (use `null` to skip)
- `max_shape` - Maximum size per dimension (use `null` to skip)

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `allow_extra_datasets` | `true` | Allow datasets not in schema |
| `allow_extra_groups` | `true` | Allow groups not in schema |
| `allow_extra_attributes` | `true` | Allow attributes not in schema |
| `required` | `true` | Dataset/group must exist |

## Tool Comparison

| Feature | print_hdf5.py | edit_hdf5.py | visualize_hdf5.py |
|---------|---------------|-------------|-------------------|
| **Read files** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Write/Modify** | ❌ No | ✅ Yes | ❌ No |
| **Interactive** | ❌ CLI only | ✅ Terminal UI | ❌ CLI only |
| **Inspect structure** | ✅ Yes | ✅ Yes | ✅ Yes |
| **View data** | ✅ Yes | ✅ Yes (multi-dimensional) | ❌ No |
| **Statistics** | ✅ Yes | ❌ No | ❌ No |
| **Histograms** | ✅ Yes | ❌ No | ❌ No |
| **Schema validation** | ✅ Yes | ❌ No | ❌ No |
| **Add/Remove items** | ❌ No | ✅ Yes | ❌ No |
| **Edit datasets** | ❌ No | ✅ Yes | ❌ No |
| **Vector operations** | ❌ No | ✅ Yes (Cut, Subsample, Sample) | ❌ No |
| **Batch processing** | ✅ Yes | ❌ No | ❌ No |
| **Automatic backups** | ❌ No | ✅ Yes | ❌ No |
| **Graph visualization** | ❌ No | ❌ No | ✅ Yes (PDF) |
| **Show all metadata** | ❌ No | ❌ No | ✅ Yes (inline) |
| **Link visualization** | ❌ No | ❌ No | ✅ Yes (hard/soft/external) |

## Workflow Examples

### Analysis Pipeline
```bash
# 1. Inspect file structure
python hdf5/print_hdf5.py inspect training_data.hdf5

# 2. Analyze with statistics and histogram
python hdf5/print_hdf5.py analyze training_data.hdf5 --stats --histogram 20 --validate

# 3. Validate against schema
python hdf5/print_hdf5.py validate training_data.hdf5 --schema schemas/marv_training_data.json

# 4. Generate visualization for comprehensive understanding
python visualize_hdf5.py training_data.hdf5
```

### Data Editing Pipeline
```bash
# 1. Inspect and analyze current file
python hdf5/print_hdf5.py analyze data.hdf5 --stats

# 2. Generate visualization to understand structure
python visualize_hdf5.py data.hdf5

# 3. Open in editor to modify
python hdf5/edit_hdf5.py data.hdf5

# 4. Re-analyze to verify changes
python hdf5/print_hdf5.py analyze data.hdf5 --stats --validate
```

### Structure Understanding
```bash
# Quickly visualize complex nested structure
python visualize_hdf5.py complex_data.hdf5 -o structure_overview

# Then analyze in detail if needed
python hdf5/print_hdf5.py inspect complex_data.hdf5
```

### Batch Quality Check
```bash
# Check all training files for quality issues
python hdf5/print_hdf5.py analyze --dir data/training --pattern "*.hdf5" --validate
```

## Example Schemas

See `schemas/` directory:
- `marv_training_data.json` - Training data format
- `marv_simulation_data.json` - Simulation trajectory format
