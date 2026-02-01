#!/usr/bin/env python3
"""
Unified HDF5 Analysis Tool

Comprehensive tool for inspecting, analyzing, comparing, and validating HDF5 files.
Combines capabilities from multiple scripts into one unified interface.

Features:
- Structure inspection (datasets, groups, attributes)
- Quality analysis (NaN/Inf detection, statistics)
- Schema validation (check against JSON schema)
- Multi-file comparison
- Batch processing

Schema JSON Format:
{
    "allow_extra_datasets": true/false,  // Allow datasets not in schema
    "allow_extra_groups": true/false,    // Allow groups not in schema
    "allow_extra_attributes": true/false, // Allow attributes not in schema
    "attributes": {                       // Root-level attributes
        "attr_name": {"type": "int", "required": true}
    },
    "datasets": {
        "dataset_name": {
            "required": true/false,
            "dtype": "float32",           // Expected dtype (optional)
            "shape": [null, 200, 100],    // null = any size, number = exact
            "ndim": 3,                    // Expected number of dimensions (optional)
            "attributes": {               // Dataset attributes
                "min": {"type": "float", "required": false},
                "max": {"type": "float", "required": false}
            }
        }
    },
    "groups": {
        "group_name": {
            "required": true/false,
            "allow_extra_datasets": true/false,  // Override for this group
            "datasets": { ... },
            "groups": { ... }
        }
    }
}

Usage:
    # Simple inspection
    python hdf5_tool.py inspect file.hdf5

    # Quality analysis with statistics
    python hdf5_tool.py analyze file.hdf5 --stats --validate

    # Analysis with histogram (10 bins)
    python hdf5_tool.py analyze file.hdf5 --stats --histogram 10

    # Analysis excluding specific datasets from histogram
    python hdf5_tool.py analyze file.hdf5 --stats --histogram 10 --exclude heightmap

    # Schema validation
    python hdf5_tool.py validate file.hdf5 --schema schema.json

    # Compare multiple files
    python hdf5_tool.py compare file1.hdf5 file2.hdf5

    # Batch analysis
    python hdf5_tool.py analyze --dir data/generated --pattern "*.hdf5"
"""

import argparse
import json
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from tqdm import tqdm

from .hdf5_common import DatasetInfo, GroupInfo, FileInfo, format_size, extract_file_info


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationError:
    """Represents a schema validation error."""
    path: str
    error_type: str
    message: str
    expected: Any = None
    actual: Any = None


# =============================================================================
# Histogram Functions
# =============================================================================

def compute_histogram(dataset: h5py.Dataset, num_bins: int,
                      min_val: float, max_val: float) -> List[int]:
    """
    Compute histogram distribution across bins.

    Args:
        dataset: HDF5 dataset
        num_bins: Number of bins
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        list: Count of samples in each bin
    """
    bin_counts = [0] * num_bins
    bin_width = (max_val - min_val) / num_bins

    # Process dataset in chunks
    chunk_size = 10000
    total_chunks = (dataset.shape[0] + chunk_size - 1) // chunk_size

    for i in tqdm(range(0, dataset.shape[0], chunk_size),
                  desc=f"  Computing histogram",
                  total=total_chunks,
                  leave=False):
        chunk = dataset[i:i+chunk_size]
        chunk_flat = chunk.flatten()

        for val in chunk_flat:
            # Find which bin this value belongs to
            bin_idx = int((val - min_val) / bin_width)
            # Handle edge case where val == max_val
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            if bin_idx < 0:
                bin_idx = 0
            bin_counts[bin_idx] += 1

    return bin_counts


def format_histogram(bin_counts: List[int], min_val: float, max_val: float,
                     max_bar_width: int = 40) -> str:
    """
    Format histogram as horizontal bars.

    Args:
        bin_counts: List of counts per bin
        min_val: Minimum value
        max_val: Maximum value
        max_bar_width: Maximum width of bar in characters

    Returns:
        str: Formatted histogram
    """
    num_bins = len(bin_counts)
    bin_width = (max_val - min_val) / num_bins
    total_count = sum(bin_counts)

    # Detect if this is integer class data: min/max are integers and bins match range
    is_integer_classes = (
        min_val == int(min_val) and
        max_val == int(max_val) and
        num_bins == int(max_val - min_val + 1)
    )

    output = []
    max_count = max(bin_counts) if bin_counts else 1

    for i, count in enumerate(bin_counts):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width

        # Calculate percentage and bar width
        if total_count > 0:
            percentage = (count / total_count) * 100
            bar_width = int((count / max_count) * max_bar_width) if max_count > 0 else 0
        else:
            percentage = 0
            bar_width = 0

        # Create bar
        bar = "█" * bar_width

        # Format bin label
        if is_integer_classes:
            class_val = int(min_val + i)
            bin_label = f"Class {class_val:3d}"
        else:
            bin_label = f"[{bin_start:8.3f}, {bin_end:8.3f}]"

        # Format line with count
        line = f"    {bin_label}: {bar:<{max_bar_width}} {count:8d} ({percentage:5.1f}%)"
        output.append(line)

    return "\n".join(output)




@dataclass
class AnalysisResult:
    """Result of file analysis."""
    file_info: FileInfo
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    quality_issues: List[Dict[str, Any]] = field(default_factory=list)
    validation_errors: List[ValidationError] = field(default_factory=list)
    histograms: Dict[str, Tuple[List[int], float, float]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if no errors or issues found."""
        return (not self.file_info.errors and
                not self.quality_issues and
                not self.validation_errors)

    @property
    def has_file_errors(self) -> bool:
        """True if file could not be read properly."""
        return bool(self.file_info.errors)

    @property
    def has_quality_issues(self) -> bool:
        """True if NaN/Inf/zero-dimension issues found."""
        return bool(self.quality_issues)

    @property
    def has_validation_errors(self) -> bool:
        """True if schema validation failed."""
        return bool(self.validation_errors)


# =============================================================================
# Schema Validation
# =============================================================================

class SchemaValidator:
    """Validates HDF5 file structure against JSON schema."""

    def __init__(self, schema: Dict):
        self.schema = schema
        self.errors: List[ValidationError] = []

    def validate(self, hdf5_path: Path) -> List[ValidationError]:
        """Validate HDF5 file against schema."""
        self.errors = []

        try:
            with h5py.File(hdf5_path, 'r') as f:
                self._validate_group(f, "/", self.schema)
        except Exception as e:
            self.errors.append(ValidationError(
                path="/",
                error_type="file_error",
                message=f"Failed to open file: {e}"
            ))

        return self.errors

    def _validate_group(self, group: h5py.Group, path: str, schema: Dict):
        """Validate a group against its schema."""
        allow_extra_datasets = schema.get("allow_extra_datasets", True)
        allow_extra_groups = schema.get("allow_extra_groups", True)
        allow_extra_attributes = schema.get("allow_extra_attributes", True)

        # Validate root/group attributes
        if "attributes" in schema:
            self._validate_attributes(
                dict(group.attrs),
                schema["attributes"],
                path,
                allow_extra_attributes
            )

        # Get actual datasets and groups in this group
        actual_datasets = set()
        actual_groups = set()
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                actual_datasets.add(key)
            elif isinstance(group[key], h5py.Group):
                actual_groups.add(key)

        # Validate datasets
        if "datasets" in schema:
            expected_datasets = set(schema["datasets"].keys())

            # Check for required datasets
            for ds_name, ds_schema in schema["datasets"].items():
                ds_path = f"{path}{ds_name}" if path == "/" else f"{path}/{ds_name}"

                if ds_schema.get("required", True):
                    if ds_name not in actual_datasets:
                        self.errors.append(ValidationError(
                            path=ds_path,
                            error_type="missing_dataset",
                            message=f"Required dataset '{ds_name}' not found"
                        ))
                        continue

                if ds_name in actual_datasets:
                    self._validate_dataset(group[ds_name], ds_path, ds_schema)

            # Check for extra datasets
            if not allow_extra_datasets:
                extra = actual_datasets - expected_datasets
                for ds_name in extra:
                    ds_path = f"{path}{ds_name}" if path == "/" else f"{path}/{ds_name}"
                    self.errors.append(ValidationError(
                        path=ds_path,
                        error_type="extra_dataset",
                        message=f"Unexpected dataset '{ds_name}' (not in schema)"
                    ))

        # Validate groups
        if "groups" in schema:
            expected_groups = set(schema["groups"].keys())

            # Check for required groups
            for grp_name, grp_schema in schema["groups"].items():
                grp_path = f"{path}{grp_name}" if path == "/" else f"{path}/{grp_name}"

                if grp_schema.get("required", True):
                    if grp_name not in actual_groups:
                        self.errors.append(ValidationError(
                            path=grp_path,
                            error_type="missing_group",
                            message=f"Required group '{grp_name}' not found"
                        ))
                        continue

                if grp_name in actual_groups:
                    self._validate_group(group[grp_name], grp_path, grp_schema)

            # Check for extra groups
            if not allow_extra_groups:
                extra = actual_groups - expected_groups
                for grp_name in extra:
                    grp_path = f"{path}{grp_name}" if path == "/" else f"{path}/{grp_name}"
                    self.errors.append(ValidationError(
                        path=grp_path,
                        error_type="extra_group",
                        message=f"Unexpected group '{grp_name}' (not in schema)"
                    ))

    def _validate_dataset(self, dataset: h5py.Dataset, path: str, schema: Dict):
        """Validate a dataset against its schema."""
        # Validate dtype
        if "dtype" in schema:
            expected_dtype = schema["dtype"]
            actual_dtype = str(dataset.dtype)

            # Normalize dtype strings for comparison
            if not self._dtypes_match(expected_dtype, actual_dtype):
                self.errors.append(ValidationError(
                    path=path,
                    error_type="wrong_dtype",
                    message=f"Wrong dtype",
                    expected=expected_dtype,
                    actual=actual_dtype
                ))

        # Validate ndim
        if "ndim" in schema:
            expected_ndim = schema["ndim"]
            actual_ndim = len(dataset.shape)
            if actual_ndim != expected_ndim:
                self.errors.append(ValidationError(
                    path=path,
                    error_type="wrong_ndim",
                    message=f"Wrong number of dimensions",
                    expected=expected_ndim,
                    actual=actual_ndim
                ))

        # Validate shape
        if "shape" in schema:
            expected_shape = schema["shape"]
            actual_shape = dataset.shape

            if len(expected_shape) != len(actual_shape):
                self.errors.append(ValidationError(
                    path=path,
                    error_type="wrong_shape",
                    message=f"Shape has wrong number of dimensions",
                    expected=f"{len(expected_shape)}D {expected_shape}",
                    actual=f"{len(actual_shape)}D {actual_shape}"
                ))
            else:
                for i, (exp, act) in enumerate(zip(expected_shape, actual_shape)):
                    if exp is not None and exp != act:
                        self.errors.append(ValidationError(
                            path=path,
                            error_type="wrong_shape",
                            message=f"Dimension {i} has wrong size",
                            expected=exp,
                            actual=act
                        ))

        # Validate min_shape (minimum size for each dimension)
        if "min_shape" in schema:
            min_shape = schema["min_shape"]
            actual_shape = dataset.shape

            if len(min_shape) == len(actual_shape):
                for i, (min_size, act) in enumerate(zip(min_shape, actual_shape)):
                    if min_size is not None and act < min_size:
                        self.errors.append(ValidationError(
                            path=path,
                            error_type="shape_too_small",
                            message=f"Dimension {i} is smaller than minimum",
                            expected=f">= {min_size}",
                            actual=act
                        ))

        # Validate max_shape (maximum size for each dimension)
        if "max_shape" in schema:
            max_shape = schema["max_shape"]
            actual_shape = dataset.shape

            if len(max_shape) == len(actual_shape):
                for i, (max_size, act) in enumerate(zip(max_shape, actual_shape)):
                    if max_size is not None and act > max_size:
                        self.errors.append(ValidationError(
                            path=path,
                            error_type="shape_too_large",
                            message=f"Dimension {i} is larger than maximum",
                            expected=f"<= {max_size}",
                            actual=act
                        ))

        # Validate attributes
        if "attributes" in schema:
            allow_extra = schema.get("allow_extra_attributes", True)
            self._validate_attributes(
                dict(dataset.attrs),
                schema["attributes"],
                path,
                allow_extra
            )

    def _validate_attributes(self, actual_attrs: Dict, schema_attrs: Dict,
                            path: str, allow_extra: bool):
        """Validate attributes against schema."""
        expected_attrs = set(schema_attrs.keys())
        actual_attr_names = set(actual_attrs.keys())

        # Check required attributes
        for attr_name, attr_schema in schema_attrs.items():
            attr_path = f"{path}@{attr_name}"

            if attr_schema.get("required", False):
                if attr_name not in actual_attrs:
                    self.errors.append(ValidationError(
                        path=attr_path,
                        error_type="missing_attribute",
                        message=f"Required attribute '{attr_name}' not found"
                    ))
                    continue

            if attr_name in actual_attrs:
                # Validate attribute type
                if "type" in attr_schema:
                    expected_type = attr_schema["type"]
                    actual_value = actual_attrs[attr_name]
                    if not self._check_attr_type(actual_value, expected_type):
                        self.errors.append(ValidationError(
                            path=attr_path,
                            error_type="wrong_attribute_type",
                            message=f"Wrong attribute type",
                            expected=expected_type,
                            actual=type(actual_value).__name__
                        ))

                # Validate attribute value
                if "value" in attr_schema:
                    expected_value = attr_schema["value"]
                    actual_value = actual_attrs[attr_name]
                    if actual_value != expected_value:
                        self.errors.append(ValidationError(
                            path=attr_path,
                            error_type="wrong_attribute_value",
                            message=f"Wrong attribute value",
                            expected=expected_value,
                            actual=actual_value
                        ))

        # Check for extra attributes
        if not allow_extra:
            extra = actual_attr_names - expected_attrs
            for attr_name in extra:
                attr_path = f"{path}@{attr_name}"
                self.errors.append(ValidationError(
                    path=attr_path,
                    error_type="extra_attribute",
                    message=f"Unexpected attribute '{attr_name}'"
                ))

    def _dtypes_match(self, expected: str, actual: str) -> bool:
        """Check if dtypes match (with normalization)."""
        # Normalize common dtype aliases
        dtype_aliases = {
            "float32": ["float32", "<f4", "f4"],
            "float64": ["float64", "<f8", "f8", "double"],
            "int32": ["int32", "<i4", "i4"],
            "int64": ["int64", "<i8", "i8"],
            "uint8": ["uint8", "|u1", "u1"],
            "bool": ["bool", "|b1", "b1"],
        }

        expected_lower = expected.lower()
        actual_lower = actual.lower()

        # Direct match
        if expected_lower == actual_lower:
            return True

        # Check aliases
        for canonical, aliases in dtype_aliases.items():
            if expected_lower in aliases or expected_lower == canonical:
                if actual_lower in aliases or actual_lower == canonical:
                    return True

        return False

    def _check_attr_type(self, value: Any, expected_type: str) -> bool:
        """Check if attribute value matches expected type."""
        type_map = {
            "int": (int, np.integer),
            "float": (float, np.floating),
            "str": (str, bytes),
            "bool": (bool, np.bool_),
            "array": (np.ndarray, list),
        }

        if expected_type in type_map:
            return isinstance(value, type_map[expected_type])

        return True  # Unknown type, allow


# =============================================================================
# File Analysis
# =============================================================================

class HDF5Analyzer:
    """Analyzes HDF5 files for structure and quality."""

    def __init__(self, validate_values: bool = False, compute_stats: bool = False,
                 histogram_bins: Optional[int] = None,
                 exclude_datasets: Optional[List[str]] = None):
        self.validate_values = validate_values
        self.compute_stats = compute_stats
        self.histogram_bins = histogram_bins
        self.exclude_datasets = exclude_datasets or []

    def analyze_file(self, hdf5_path: Path) -> AnalysisResult:
        """Perform complete analysis of an HDF5 file."""
        file_info = self._extract_file_info(hdf5_path)
        result = AnalysisResult(file_info=file_info)

        if file_info.errors:
            return result

        try:
            with h5py.File(hdf5_path, 'r') as f:
                # Quality checks
                if self.validate_values:
                    result.quality_issues = self._check_quality(f, file_info)

                # Statistics
                if self.compute_stats:
                    result.statistics = self._compute_statistics(f, file_info)

                # Histograms (requires statistics for min/max)
                if self.histogram_bins:
                    result.histograms = self._compute_histograms(f, file_info, result.statistics)

        except Exception as e:
            result.file_info.errors.append(f"Analysis error: {e}")

        return result

    def _extract_file_info(self, hdf5_path: Path) -> FileInfo:
        """Extract structure information from HDF5 file."""
        return extract_file_info(hdf5_path)

    def _check_quality(self, f: h5py.File, file_info: FileInfo) -> List[Dict]:
        """Check data quality (NaN, Inf, zero dimensions)."""
        issues = []

        for ds_name, ds_info in file_info.datasets.items():
            dataset = f[ds_name]

            # Check for zero dimensions
            if any(dim == 0 for dim in ds_info.shape):
                issues.append({
                    'type': 'zero_dimension',
                    'dataset': ds_name,
                    'detail': f'Zero-size dimension: {ds_info.shape}'
                })
                continue

            # Check for NaN/Inf in floating point datasets
            if np.issubdtype(dataset.dtype, np.floating):
                # Sample check for large datasets
                if dataset.size > 1_000_000:
                    sample = dataset[:min(100, ds_info.shape[0])].flatten()[:1000]
                else:
                    sample = dataset[:].flatten()

                nan_count = int(np.isnan(sample).sum())
                inf_count = int(np.isinf(sample).sum())

                if nan_count > 0:
                    issues.append({
                        'type': 'nan_values',
                        'dataset': ds_name,
                        'detail': f'Found {nan_count} NaN values in sample'
                    })

                if inf_count > 0:
                    issues.append({
                        'type': 'inf_values',
                        'dataset': ds_name,
                        'detail': f'Found {inf_count} Inf values in sample'
                    })

        return issues

    def _compute_statistics(self, f: h5py.File, file_info: FileInfo) -> Dict[str, Dict]:
        """Compute statistics for numeric datasets."""
        stats = {}

        for ds_name, ds_info in file_info.datasets.items():
            dataset = f[ds_name]

            if not (np.issubdtype(dataset.dtype, np.floating) or
                    np.issubdtype(dataset.dtype, np.integer)):
                continue

            # Check for stored statistics first
            if all(attr in dataset.attrs for attr in ['min', 'max', 'mean']):
                stats[ds_name] = {
                    'min': float(dataset.attrs['min']),
                    'max': float(dataset.attrs['max']),
                    'mean': float(dataset.attrs['mean']),
                    'source': 'stored'
                }
            else:
                # Compute statistics
                running_min = np.inf
                running_max = -np.inf
                running_sum = 0.0
                total_count = 0

                chunk_size = 1000
                for i in range(0, dataset.shape[0], chunk_size):
                    chunk = dataset[i:i+chunk_size]
                    running_min = min(running_min, float(np.min(chunk)))
                    running_max = max(running_max, float(np.max(chunk)))
                    running_sum += float(np.sum(chunk))
                    total_count += chunk.size

                stats[ds_name] = {
                    'min': float(running_min),
                    'max': float(running_max),
                    'mean': float(running_sum / total_count) if total_count > 0 else 0.0,
                    'source': 'computed'
                }

        return stats

    def _compute_histograms(self, f: h5py.File, file_info: FileInfo,
                           statistics: Dict[str, Dict]) -> Dict[str, Tuple[List[int], float, float]]:
        """Compute histograms for numeric datasets."""
        histograms = {}

        for ds_name, ds_info in file_info.datasets.items():
            # Skip excluded datasets
            if ds_name in self.exclude_datasets:
                continue

            dataset = f[ds_name]

            if not (np.issubdtype(dataset.dtype, np.floating) or
                    np.issubdtype(dataset.dtype, np.integer)):
                continue

            # Get min/max from statistics (computed or stored)
            if ds_name in statistics:
                min_val = statistics[ds_name]['min']
                max_val = statistics[ds_name]['max']
            elif all(attr in dataset.attrs for attr in ['min', 'max']):
                min_val = float(dataset.attrs['min'])
                max_val = float(dataset.attrs['max'])
            else:
                # Compute min/max on the fly
                min_val = float(np.min(dataset[:]))
                max_val = float(np.max(dataset[:]))

            # Compute histogram
            bin_counts = compute_histogram(dataset, self.histogram_bins, min_val, max_val)
            histograms[ds_name] = (bin_counts, min_val, max_val)

        return histograms


# =============================================================================
# Output Formatting
# =============================================================================

def print_inspection(file_info: FileInfo):
    """Print file inspection results."""
    print(f"\nFile: {file_info.path}")
    print(f"Size: {format_size(file_info.size_bytes)}")
    print("=" * 70)

    # Root attributes
    print("\nRoot Attributes:")
    if file_info.root_attributes:
        for key, value in file_info.root_attributes.items():
            print(f"  {key}: {value}")
    else:
        print("  (none)")

    # Datasets
    print(f"\nDatasets ({len(file_info.datasets)}):")
    for name, ds in sorted(file_info.datasets.items()):
        print(f"\n  {name}:")
        print(f"    Shape: {ds.shape}")
        print(f"    Dtype: {ds.dtype}")
        print(f"    Size: {format_size(ds.size_bytes)}")
        if ds.chunks:
            print(f"    Chunks: {ds.chunks}")
        if ds.compression:
            print(f"    Compression: {ds.compression}")
        if ds.attributes:
            print(f"    Attributes: {ds.attributes}")

    # Groups
    if file_info.groups:
        print(f"\nGroups ({len(file_info.groups)}):")
        for name, grp in sorted(file_info.groups.items()):
            print(f"\n  {name}:")
            print(f"    Datasets: {grp.num_datasets}")
            print(f"    Subgroups: {grp.num_groups}")
            if grp.attributes:
                print(f"    Attributes: {grp.attributes}")


def print_validation_errors(errors: List[ValidationError]):
    """Print validation errors."""
    if not errors:
        print("\nSchema Validation: PASSED")
        return

    print(f"\nSchema Validation: FAILED ({len(errors)} errors)")
    print("=" * 70)

    # Group by error type
    by_type = defaultdict(list)
    for err in errors:
        by_type[err.error_type].append(err)

    for error_type, type_errors in sorted(by_type.items()):
        print(f"\n{error_type.upper()} ({len(type_errors)}):")
        for err in type_errors[:10]:
            msg = f"  {err.path}: {err.message}"
            if err.expected is not None:
                msg += f" (expected: {err.expected}, actual: {err.actual})"
            print(msg)
        if len(type_errors) > 10:
            print(f"  ... and {len(type_errors) - 10} more")


def print_analysis(result: AnalysisResult, verbose: bool = False):
    """Print analysis results."""
    print_inspection(result.file_info)

    # Statistics
    if result.statistics:
        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        for ds_name, stats in sorted(result.statistics.items()):
            print(f"\n{ds_name} ({stats.get('source', 'unknown')}):")
            print(f"  min:  {stats['min']:.6f}")
            print(f"  max:  {stats['max']:.6f}")
            print(f"  mean: {stats['mean']:.6f}")

            # Print histogram if available
            if ds_name in result.histograms:
                bin_counts, min_val, max_val = result.histograms[ds_name]
                print(f"\n  Histogram ({len(bin_counts)} bins):")
                print(format_histogram(bin_counts, min_val, max_val))

    # Histograms for datasets not in statistics
    remaining_histograms = {k: v for k, v in result.histograms.items()
                           if k not in result.statistics}
    if remaining_histograms:
        print("\n" + "=" * 70)
        print("HISTOGRAMS")
        print("=" * 70)
        for ds_name, (bin_counts, min_val, max_val) in sorted(remaining_histograms.items()):
            print(f"\n{ds_name}:")
            print(f"  range: [{min_val:.6f}, {max_val:.6f}]")
            print(f"\n  Histogram ({len(bin_counts)} bins):")
            print(format_histogram(bin_counts, min_val, max_val))

    # Quality issues
    if result.quality_issues:
        print("\n" + "=" * 70)
        print(f"QUALITY ISSUES ({len(result.quality_issues)})")
        print("=" * 70)
        for issue in result.quality_issues:
            print(f"  [{issue['type']}] {issue['dataset']}: {issue['detail']}")
    elif result.file_info.errors:
        print("\nFile had errors, quality check skipped")

    # Validation errors
    if result.validation_errors:
        print_validation_errors(result.validation_errors)


# =============================================================================
# Index Parsing Helpers
# =============================================================================

def _parse_field_spec(field_spec: str) -> tuple[str, str | None]:
    """
    Parse field specification into name and index.

    Examples:
        "data" -> ("data", None)
        "data[0]" -> ("data", "0")
        "data[1:3]" -> ("data", "1:3")
        "data[:, 0]" -> ("data", ":, 0")
        "data[:-1]" -> ("data", ":-1")
    """
    if '[' in field_spec:
        bracket_pos = field_spec.index('[')
        field_name = field_spec[:bracket_pos]
        # Remove outer brackets
        index_str = field_spec[bracket_pos + 1:-1] if field_spec.endswith(']') else field_spec[bracket_pos + 1:]
        return field_name, index_str
    return field_spec, None


def _parse_slice_part(s: str) -> int | None:
    """Parse a single part of a slice (handles empty, positive, negative)."""
    s = s.strip()
    if s == '' or s == 'None':
        return None
    return int(s)


def _parse_single_index(idx_str: str) -> slice | int:
    """
    Parse a single index dimension.

    Examples:
        "0" -> 0
        "1:3" -> slice(1, 3)
        ":3" -> slice(None, 3)
        "1:" -> slice(1, None)
        ":" -> slice(None, None)
        ":-1" -> slice(None, -1)
        "-1" -> -1
        "1:10:2" -> slice(1, 10, 2)
    """
    idx_str = idx_str.strip()

    if ':' in idx_str:
        parts = idx_str.split(':')
        if len(parts) == 2:
            start = _parse_slice_part(parts[0])
            stop = _parse_slice_part(parts[1])
            return slice(start, stop)
        elif len(parts) == 3:
            start = _parse_slice_part(parts[0])
            stop = _parse_slice_part(parts[1])
            step = _parse_slice_part(parts[2])
            return slice(start, stop, step)
    else:
        return int(idx_str)


def _parse_index(index_str: str, shape: tuple) -> tuple:
    """
    Parse full index string into tuple for numpy/h5py indexing.

    Examples:
        "0" -> (0,)
        "1, 2" -> (1, 2)
        ":, 0" -> (slice(None), 0)
        "0:10, :, -1" -> (slice(0, 10), slice(None), -1)
    """
    # Split by comma, handling spaces
    parts = [p.strip() for p in index_str.split(',')]

    indices = []
    for part in parts:
        indices.append(_parse_single_index(part))

    return tuple(indices)


def _apply_index(value, index_str: str):
    """Apply index to a numpy array or similar."""
    if hasattr(value, '__getitem__'):
        idx = _parse_index(index_str, getattr(value, 'shape', (len(value),)))
        if len(idx) == 1:
            return value[idx[0]]
        return value[idx]
    return value


# =============================================================================
# Commands
# =============================================================================

def cmd_inspect(args):
    """Inspect command - show file structure."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return 1

    analyzer = HDF5Analyzer()
    result = analyzer.analyze_file(file_path)
    print_inspection(result.file_info)

    # Print field values if requested
    if hasattr(args, 'print_fields') and args.print_fields:
        print("\n" + "=" * 70)
        print("FIELD VALUES")
        print("=" * 70)

        with h5py.File(file_path, 'r') as f:
            for field_spec in args.print_fields:
                # Parse field name and optional index (e.g., "data[0]", "data[1:3]", "data[:, 0]")
                field_name, index_str = _parse_field_spec(field_spec)

                print(f"\n{field_spec}:")

                # Check if it's a root attribute
                if field_name in f.attrs:
                    value = f.attrs[field_name]
                    print(f"  (root attribute)")
                    if index_str:
                        value = _apply_index(value, index_str)
                    print(f"  {value}")

                # Check if it's a dataset
                elif field_name in f:
                    dataset = f[field_name]
                    if isinstance(dataset, h5py.Dataset):
                        # Apply indexing if specified
                        if index_str:
                            idx = _parse_index(index_str, dataset.shape)
                            data = dataset[idx]
                        else:
                            data = dataset[:]

                        print(f"  Shape: {data.shape}, Dtype: {data.dtype}")

                        # If user specified an index, always print full result (up to reasonable limit)
                        if index_str:
                            if data.size <= 1000:
                                print(f"  {data}")
                            else:
                                print(f"  (showing first 100 elements of {data.size})")
                                if data.ndim == 1:
                                    print(f"  {data[:100]}")
                                else:
                                    print(f"  {data.flat[:100]}")
                        # No index - default truncation for large arrays
                        elif data.size <= 100:
                            print(f"  {data}")
                        elif data.ndim == 1 and data.size <= 1000:
                            print(f"  {data}")
                        else:
                            print(f"  (large array, showing first elements)")
                            if data.ndim == 1:
                                print(f"  {data[:20]}...")
                            else:
                                print(f"  {data.flat[:20]}...")

                        # Print dataset attributes
                        if dataset.attrs:
                            print(f"  Attributes: {dict(dataset.attrs)}")
                    else:
                        print(f"  (group, not a dataset)")

                else:
                    print(f"  (not found)")

    return 0


def cmd_analyze(args):
    """Analyze command - quality checks and statistics."""
    # Handle single file or directory
    if args.file:
        files = [Path(args.file)]
    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory {dir_path} does not exist")
            return 1
        files = sorted(dir_path.glob(args.pattern))
    else:
        print("Error: Specify either a file or --dir")
        return 1

    if not files:
        print("No HDF5 files found")
        return 1

    # Validate histogram bins
    histogram_bins = getattr(args, 'histogram', None)
    if histogram_bins is not None:
        if histogram_bins < 2:
            print("Error: --histogram N must be at least 2")
            return 1

    exclude_datasets = getattr(args, 'exclude', []) or []

    analyzer = HDF5Analyzer(
        validate_values=args.validate,
        compute_stats=args.stats,
        histogram_bins=histogram_bins,
        exclude_datasets=exclude_datasets
    )

    # Load schema if provided
    schema = None
    if args.schema:
        schema_path = Path(args.schema)
        if not schema_path.exists():
            print(f"Error: Schema file {schema_path} does not exist")
            return 1
        with open(schema_path, 'r') as f:
            schema = json.load(f)

    # Analyze files
    all_results = []
    for file_path in tqdm(files, desc="Analyzing", disable=len(files) == 1):
        result = analyzer.analyze_file(file_path)

        # Schema validation
        if schema:
            validator = SchemaValidator(schema)
            result.validation_errors = validator.validate(file_path)

        all_results.append(result)

    # Print results
    if len(files) == 1:
        print_analysis(all_results[0], verbose=args.verbose)
    else:
        # Summary for multiple files
        print(f"\nAnalyzed {len(files)} files")
        print("=" * 70)

        # Aggregate statistics
        total_issues = sum(len(r.quality_issues) for r in all_results)
        total_errors = sum(len(r.file_info.errors) for r in all_results)
        total_validation = sum(len(r.validation_errors) for r in all_results)

        print(f"Files with read errors: {sum(1 for r in all_results if r.file_info.errors)}")
        print(f"Files with quality issues: {sum(1 for r in all_results if r.quality_issues)}")
        if schema:
            print(f"Files with validation errors: {sum(1 for r in all_results if r.validation_errors)}")

        if args.verbose:
            for result in all_results:
                if result.quality_issues or result.validation_errors or result.file_info.errors:
                    print(f"\n{result.file_info.path}:")
                    for issue in result.quality_issues:
                        print(f"  [{issue['type']}] {issue['dataset']}")
                    for err in result.validation_errors:
                        print(f"  [validation] {err.path}: {err.message}")

    return 0


def cmd_validate(args):
    """Validate command - check file against schema."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return 1

    schema_path = Path(args.schema)
    if not schema_path.exists():
        print(f"Error: Schema file {schema_path} does not exist")
        return 1

    with open(schema_path, 'r') as f:
        schema = json.load(f)

    validator = SchemaValidator(schema)
    errors = validator.validate(file_path)

    print(f"\nValidating: {file_path}")
    print(f"Schema: {schema_path}")
    print_validation_errors(errors)

    return 1 if errors else 0


def cmd_compare(args):
    """Compare command - compare multiple files."""
    files = [Path(f) for f in args.files]

    for f in files:
        if not f.exists():
            print(f"Error: File {f} does not exist")
            return 1

    if len(files) < 2:
        print("Error: Need at least 2 files to compare")
        return 1

    analyzer = HDF5Analyzer()
    results = [analyzer.analyze_file(f) for f in tqdm(files, desc="Reading")]

    # Compare datasets
    print("\n" + "=" * 70)
    print("DATASET COMPARISON")
    print("=" * 70)

    all_datasets = set()
    for r in results:
        all_datasets.update(r.file_info.datasets.keys())

    for ds_name in sorted(all_datasets):
        print(f"\n{ds_name}:")

        # Check presence
        present_in = [r.file_info.path.name for r in results if ds_name in r.file_info.datasets]
        if len(present_in) < len(files):
            print(f"  Present in: {len(present_in)}/{len(files)} files")

        # Check shapes
        shapes = {}
        for r in results:
            if ds_name in r.file_info.datasets:
                shape = r.file_info.datasets[ds_name].shape
                if shape not in shapes:
                    shapes[shape] = []
                shapes[shape].append(r.file_info.path.name)

        if len(shapes) == 1:
            print(f"  Shape: {list(shapes.keys())[0]} (consistent)")
        else:
            print(f"  Shapes: INCONSISTENT")
            for shape, filenames in shapes.items():
                print(f"    {shape}: {len(filenames)} files")

        # Check dtypes
        dtypes = {}
        for r in results:
            if ds_name in r.file_info.datasets:
                dtype = r.file_info.datasets[ds_name].dtype
                if dtype not in dtypes:
                    dtypes[dtype] = []
                dtypes[dtype].append(r.file_info.path.name)

        if len(dtypes) == 1:
            print(f"  Dtype: {list(dtypes.keys())[0]} (consistent)")
        else:
            print(f"  Dtypes: INCONSISTENT")
            for dtype, filenames in dtypes.items():
                print(f"    {dtype}: {len(filenames)} files")

    return 0


def cmd_schema_template(args):
    """Generate a schema template from an HDF5 file."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return 1

    analyzer = HDF5Analyzer()
    result = analyzer.analyze_file(file_path)

    # Build schema from file structure
    schema = {
        "allow_extra_datasets": True,
        "allow_extra_groups": True,
        "allow_extra_attributes": True,
        "datasets": {},
        "attributes": {}
    }

    # Add root attributes
    for attr_name, attr_value in result.file_info.root_attributes.items():
        schema["attributes"][attr_name] = {
            "required": False,
            "type": type(attr_value).__name__
        }

    # Add datasets
    for ds_name, ds_info in result.file_info.datasets.items():
        # Use null for variable dimensions (first dimension usually varies)
        shape_spec = list(ds_info.shape)
        if len(shape_spec) > 0:
            shape_spec[0] = None  # First dim is typically batch/sample dimension

        ds_schema = {
            "required": True,
            "dtype": ds_info.dtype,
            "shape": shape_spec,
            "ndim": len(ds_info.shape)
        }

        if ds_info.attributes:
            ds_schema["attributes"] = {}
            for attr_name in ds_info.attributes:
                ds_schema["attributes"][attr_name] = {"required": False}

        schema["datasets"][ds_name] = ds_schema

    # Output
    output = json.dumps(schema, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Schema template written to: {args.output}")
    else:
        print(output)

    return 0


# =============================================================================
# Public API
# =============================================================================

def inspect(hdf5_path: str | Path, print_output: bool = True) -> FileInfo:
    """
    Inspect HDF5 file structure.

    Args:
        hdf5_path: Path to HDF5 file
        print_output: If True, print results to stdout

    Returns:
        FileInfo with file structure details
    """
    analyzer = HDF5Analyzer()
    result = analyzer.analyze_file(Path(hdf5_path))

    if print_output:
        print_inspection(result.file_info)

    return result.file_info


def analyze(
    hdf5_path: str | Path,
    validate_values: bool = False,
    compute_stats: bool = False,
    histogram_bins: Optional[int] = None,
    exclude_datasets: Optional[List[str]] = None,
    schema: Dict | str | Path | None = None,
    print_output: bool = True
) -> AnalysisResult:
    """
    Analyze HDF5 file for quality and optionally validate against schema.

    Args:
        hdf5_path: Path to HDF5 file
        validate_values: Check for NaN/Inf values
        compute_stats: Compute min/max/mean statistics
        histogram_bins: Number of bins for histogram (None to skip)
        exclude_datasets: List of dataset names to exclude from stats/histogram
        schema: JSON schema dict, or path to schema file
        print_output: If True, print results to stdout

    Returns:
        AnalysisResult with all findings. Check result.ok for pass/fail.
    """
    hdf5_path = Path(hdf5_path)
    analyzer = HDF5Analyzer(
        validate_values=validate_values,
        compute_stats=compute_stats,
        histogram_bins=histogram_bins,
        exclude_datasets=exclude_datasets
    )
    result = analyzer.analyze_file(hdf5_path)

    # Schema validation
    if schema is not None:
        if isinstance(schema, (str, Path)):
            with open(schema, 'r') as f:
                schema = json.load(f)
        validator = SchemaValidator(schema)
        result.validation_errors = validator.validate(hdf5_path)

    if print_output:
        print_analysis(result)

    return result


def validate(
    hdf5_path: str | Path,
    schema: Dict | str | Path,
    print_output: bool = True
) -> tuple[bool, List[ValidationError]]:
    """
    Validate HDF5 file against JSON schema.

    Args:
        hdf5_path: Path to HDF5 file
        schema: JSON schema dict, or path to schema file
        print_output: If True, print results to stdout

    Returns:
        Tuple of (passed: bool, errors: List[ValidationError])
    """
    hdf5_path = Path(hdf5_path)

    if isinstance(schema, (str, Path)):
        with open(schema, 'r') as f:
            schema = json.load(f)

    validator = SchemaValidator(schema)
    errors = validator.validate(hdf5_path)

    if print_output:
        print(f"\nValidating: {hdf5_path}")
        print_validation_errors(errors)

    return (len(errors) == 0, errors)


def check_quality(
    hdf5_path: str | Path,
    print_output: bool = True
) -> tuple[bool, List[Dict[str, Any]]]:
    """
    Check HDF5 file for quality issues (NaN, Inf, zero dimensions).

    Args:
        hdf5_path: Path to HDF5 file
        print_output: If True, print results to stdout

    Returns:
        Tuple of (passed: bool, issues: List[Dict])
    """
    result = analyze(hdf5_path, validate_values=True, print_output=print_output)
    return (not result.has_quality_issues, result.quality_issues)


def generate_schema(hdf5_path: str | Path) -> Dict:
    """
    Generate a JSON schema template from an HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file to use as template

    Returns:
        Dict with schema that can be saved as JSON
    """
    analyzer = HDF5Analyzer()
    result = analyzer.analyze_file(Path(hdf5_path))

    schema = {
        "allow_extra_datasets": True,
        "allow_extra_groups": True,
        "allow_extra_attributes": True,
        "datasets": {},
        "attributes": {}
    }

    for attr_name, attr_value in result.file_info.root_attributes.items():
        schema["attributes"][attr_name] = {
            "required": False,
            "type": type(attr_value).__name__
        }

    for ds_name, ds_info in result.file_info.datasets.items():
        shape_spec = list(ds_info.shape)
        if len(shape_spec) > 0:
            shape_spec[0] = None

        ds_schema = {
            "required": True,
            "dtype": ds_info.dtype,
            "shape": shape_spec,
            "ndim": len(ds_info.shape)
        }

        if ds_info.attributes:
            ds_schema["attributes"] = {}
            for attr_name in ds_info.attributes:
                ds_schema["attributes"][attr_name] = {"required": False}

        schema["datasets"][ds_name] = ds_schema

    return schema


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified HDF5 Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect file structure
  python hdf5_tool.py inspect data.hdf5

  # Analyze with statistics
  python hdf5_tool.py analyze data.hdf5 --stats --validate

  # Analyze with histogram (10 bins)
  python hdf5_tool.py analyze data.hdf5 --stats --histogram 10

  # Analyze excluding datasets from histogram
  python hdf5_tool.py analyze data.hdf5 --stats --histogram 10 --exclude heightmap

  # Validate against schema
  python hdf5_tool.py validate data.hdf5 --schema schema.json

  # Compare multiple files
  python hdf5_tool.py compare file1.hdf5 file2.hdf5

  # Generate schema template
  python hdf5_tool.py schema-template data.hdf5 -o schema.json

  # Batch analysis with schema
  python hdf5_tool.py analyze --dir data/ --pattern "*.hdf5" --schema schema.json
        """
    )

    # Common arguments shared by all subcommands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--clip', action='store_true',
                               help='Copy output to clipboard')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Inspect command
    p_inspect = subparsers.add_parser('inspect', help='Inspect file structure',
                                      parents=[common_parser])
    p_inspect.add_argument('file', help='HDF5 file to inspect')
    p_inspect.add_argument('--print', dest='print_fields', nargs='+', metavar='FIELD',
                          help='Print values of specified datasets/attributes (e.g., --print body_names robot_colors)')

    # Analyze command
    p_analyze = subparsers.add_parser('analyze', help='Analyze file(s) quality',
                                      parents=[common_parser])
    p_analyze.add_argument('file', nargs='?', help='HDF5 file to analyze')
    p_analyze.add_argument('--dir', type=str, help='Directory to scan')
    p_analyze.add_argument('--pattern', type=str, default='**/*.hdf5',
                          help='File pattern (default: **/*.hdf5)')
    p_analyze.add_argument('--validate', action='store_true',
                          help='Check for NaN/Inf values')
    p_analyze.add_argument('--stats', action='store_true',
                          help='Compute statistics')
    p_analyze.add_argument('--schema', type=str,
                          help='JSON schema file for validation')
    p_analyze.add_argument('--histogram', type=int, metavar='N',
                          help='Compute histogram with N bins (requires --stats or stored statistics)')
    p_analyze.add_argument('--exclude', type=str, nargs='+', metavar='DATASET',
                          help='Exclude datasets from stats/histogram computation')
    p_analyze.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')

    # Validate command
    p_validate = subparsers.add_parser('validate', help='Validate against schema',
                                       parents=[common_parser])
    p_validate.add_argument('file', help='HDF5 file to validate')
    p_validate.add_argument('--schema', '-s', required=True,
                           help='JSON schema file')

    # Compare command
    p_compare = subparsers.add_parser('compare', help='Compare multiple files',
                                      parents=[common_parser])
    p_compare.add_argument('files', nargs='+', help='HDF5 files to compare')

    # Schema template command
    p_template = subparsers.add_parser('schema-template',
                                       help='Generate schema template from file',
                                       parents=[common_parser])
    p_template.add_argument('file', help='HDF5 file to use as template')
    p_template.add_argument('--output', '-o', help='Output file (default: stdout)')

    # Check if first arg is an HDF5 file (default to inspect)
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.hdf5', '.h5')):
        sys.argv.insert(1, 'inspect')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    from .clip_utils import ClipboardCapture

    commands = {
        'inspect': cmd_inspect,
        'analyze': cmd_analyze,
        'validate': cmd_validate,
        'compare': cmd_compare,
        'schema-template': cmd_schema_template,
    }

    with ClipboardCapture(clip=args.clip):
        result = commands[args.command](args)
    return result


if __name__ == '__main__':
    exit(main())
