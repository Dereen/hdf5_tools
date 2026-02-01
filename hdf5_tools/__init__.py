"""
HDF5 Tools - Comprehensive tooling for HDF5 files.

Tools:
    - print_hdf5: CLI for inspection, analysis, validation
    - edit_hdf5: Interactive terminal editor
    - visualize_hdf5: Structure visualization (PDF)
    - compare_hdf5: File comparison with heatmaps
"""

from .hdf5_common import (
    DatasetInfo,
    GroupInfo,
    FileInfo,
    format_size,
    extract_file_info,
)

from .print_hdf5 import (
    inspect,
    analyze,
    validate,
    check_quality,
    generate_schema,
    AnalysisResult,
    ValidationError,
    HDF5Analyzer,
    SchemaValidator,
)

__version__ = "0.1.0"

__all__ = [
    # Data classes
    'DatasetInfo',
    'GroupInfo',
    'FileInfo',
    'AnalysisResult',
    'ValidationError',
    # Functions
    'inspect',
    'analyze',
    'validate',
    'check_quality',
    'generate_schema',
    'format_size',
    'extract_file_info',
    # Classes
    'HDF5Analyzer',
    'SchemaValidator',
]
