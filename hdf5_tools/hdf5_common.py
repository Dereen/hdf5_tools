#!/usr/bin/env python3
"""
Common utilities for HDF5 tools.
"""

import h5py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    shape: tuple
    dtype: str
    size_bytes: int
    attributes: Dict[str, Any]
    chunks: Optional[tuple] = None
    compression: Optional[str] = None


@dataclass
class GroupInfo:
    """Information about a group."""
    name: str
    attributes: Dict[str, Any]
    num_datasets: int = 0
    num_groups: int = 0


@dataclass
class FileInfo:
    """Complete information about an HDF5 file."""
    path: Path
    size_bytes: int
    root_attributes: Dict[str, Any]
    datasets: Dict[str, DatasetInfo]
    groups: Dict[str, GroupInfo]
    errors: List[str] = field(default_factory=list)


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def extract_file_info(hdf5_path: Path) -> FileInfo:
    """Extract structure information from HDF5 file."""
    file_info = FileInfo(
        path=hdf5_path,
        size_bytes=hdf5_path.stat().st_size if hdf5_path.exists() else 0,
        root_attributes={},
        datasets={},
        groups={}
    )

    try:
        with h5py.File(hdf5_path, 'r') as f:
            file_info.root_attributes = dict(f.attrs)

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    file_info.datasets[name] = DatasetInfo(
                        name=name,
                        shape=obj.shape,
                        dtype=str(obj.dtype),
                        size_bytes=obj.nbytes,
                        attributes=dict(obj.attrs),
                        chunks=obj.chunks,
                        compression=obj.compression
                    )
                elif isinstance(obj, h5py.Group):
                    num_ds = sum(1 for k in obj.keys() if isinstance(obj[k], h5py.Dataset))
                    num_grp = sum(1 for k in obj.keys() if isinstance(obj[k], h5py.Group))
                    file_info.groups[name] = GroupInfo(
                        name=name,
                        attributes=dict(obj.attrs),
                        num_datasets=num_ds,
                        num_groups=num_grp
                    )

            f.visititems(visitor)

    except Exception as e:
        file_info.errors.append(str(e))

    return file_info