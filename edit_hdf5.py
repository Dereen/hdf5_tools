#!/usr/bin/env python3
"""
Interactive HDF5 Terminal Editor

A comprehensive terminal-based HDF5 file editor using the Textual TUI framework.

Features:
- Tree visualization of HDF5 structure with color-coded data types
- Add, Remove, Edit operations for datasets, groups, and attributes
- Vector/Matrix operations: Cut, Subsample, Random sample
- Multi-dimensional data viewing with axis selection
- Configuration tab for timestep and other settings
- Backup-on-modify file handling with lazy loading

Usage:
    python edit_hdf5.py <hdf5_file>
    python edit_hdf5.py data.hdf5

Key Bindings:
    Enter   - Expand/collapse node or edit value
    a       - Add new dataset/group/attribute
    d       - Delete selected item
    e       - Edit selected item
    v       - View data (new tab)
    c       - Cut operation (axis slicing)
    s       - Subsample operation
    r       - Random sample operation
    C       - Open config tab
    F       - Toggle fullscreen for operation panel
    q       - Quit
    Ctrl+S  - Save changes
    ?       - Show help
"""

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    Pretty,
    RadioButton,
    RadioSet,
    RichLog,
    Rule,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)
from textual.widgets.option_list import Option
from textual.widgets.tree import TreeNode


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EditorConfig:
    """Configuration for the HDF5 editor."""

    timestep: float = 0.01  # Default timestep in seconds
    backup_suffix: str = ".bak"
    max_preview_size: int = 1000  # Max elements to show in preview
    max_table_rows: int = 500  # Max rows in data table

    def to_dict(self) -> Dict:
        return {
            "timestep": self.timestep,
            "backup_suffix": self.backup_suffix,
            "max_preview_size": self.max_preview_size,
            "max_table_rows": self.max_table_rows,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "EditorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HDF5NodeData:
    """Data associated with a tree node."""

    path: str  # Full HDF5 path
    node_type: str  # 'group', 'dataset', 'attribute'
    dtype: Optional[str] = None
    shape: Optional[Tuple] = None
    parent_path: Optional[str] = None
    attr_name: Optional[str] = None  # For attribute nodes


@dataclass
class Operation:
    """Represents an operation for undo/display."""

    name: str
    target: str
    details: str
    timestamp: str
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# Color Scheme
# =============================================================================


class Colors:
    """Color scheme for different data types."""

    GROUP = "blue"
    DATASET = "green"
    ATTRIBUTE = "yellow"
    INT = "cyan"
    FLOAT = "cyan"
    STRING = "magenta"
    BOOL = "red"
    BYTES = "dark_orange"
    UNKNOWN = "white"

    @classmethod
    def for_dtype(cls, dtype: str) -> str:
        """Get color for a numpy dtype."""
        dtype_lower = str(dtype).lower()
        if "int" in dtype_lower:
            return cls.INT
        if "float" in dtype_lower:
            return cls.FLOAT
        if "bool" in dtype_lower:
            return cls.BOOL
        if dtype_lower in ("object", "|o") or "str" in dtype_lower or "s" in dtype_lower:
            return cls.STRING
        if "bytes" in dtype_lower:
            return cls.BYTES
        return cls.UNKNOWN


# =============================================================================
# HDF5 Operations
# =============================================================================


class HDF5Operations:
    """Operations on HDF5 files."""

    def __init__(self, filepath: Path, config: EditorConfig):
        self.filepath = filepath
        self.config = config
        self._file: Optional[h5py.File] = None
        self.modified = False
        self.backup_created = False

    def open(self, mode: str = "r+") -> h5py.File:
        """Open HDF5 file."""
        if self._file is None or not self._file.id.valid:
            self._file = h5py.File(self.filepath, mode)
        return self._file

    def close(self):
        """Close HDF5 file."""
        if self._file is not None and self._file.id.valid:
            self._file.close()
            self._file = None

    def create_backup(self):
        """Create backup of the file."""
        if not self.backup_created:
            backup_path = self.filepath.with_suffix(
                self.filepath.suffix + self.config.backup_suffix
            )
            shutil.copy2(self.filepath, backup_path)
            self.backup_created = True
            return backup_path
        return None

    def get_tree_structure(self) -> Dict:
        """Get complete tree structure of HDF5 file."""
        structure = {"type": "root", "path": "/", "attributes": {}, "children": {}}

        with h5py.File(self.filepath, "r") as f:
            # Root attributes
            structure["attributes"] = dict(f.attrs)

            def visit_item(name, obj):
                parts = name.split("/")
                current = structure

                for i, part in enumerate(parts[:-1]):
                    current = current["children"][part]

                item_name = parts[-1]

                if isinstance(obj, h5py.Dataset):
                    current["children"][item_name] = {
                        "type": "dataset",
                        "path": "/" + name,
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "attributes": dict(obj.attrs),
                        "size": obj.nbytes,
                        "chunks": obj.chunks,
                        "compression": obj.compression,
                    }
                elif isinstance(obj, h5py.Group):
                    current["children"][item_name] = {
                        "type": "group",
                        "path": "/" + name,
                        "attributes": dict(obj.attrs),
                        "children": {},
                    }

            f.visititems(visit_item)

        return structure

    def get_dataset_slice(
        self,
        path: str,
        indices: Optional[Tuple] = None,
        max_elements: Optional[int] = None,
    ) -> np.ndarray:
        """Get a slice of dataset data (lazy loading)."""
        with h5py.File(self.filepath, "r") as f:
            dataset = f[path]
            if indices is not None:
                data = dataset[indices]
            elif max_elements and dataset.size > max_elements:
                # Load only first portion
                flat_shape = dataset.shape
                if len(flat_shape) > 0:
                    limit = min(max_elements, flat_shape[0])
                    data = dataset[:limit]
                else:
                    data = dataset[()]
            else:
                data = dataset[()]
            return np.array(data)

    def get_attribute(self, path: str, attr_name: str) -> Any:
        """Get an attribute value."""
        with h5py.File(self.filepath, "r") as f:
            if path == "/":
                return f.attrs[attr_name]
            return f[path].attrs[attr_name]

    def delete_item(self, path: str, attr_name: Optional[str] = None):
        """Delete a dataset, group, or attribute."""
        self.create_backup()
        f = self.open("r+")
        try:
            if attr_name:
                if path == "/":
                    del f.attrs[attr_name]
                else:
                    del f[path].attrs[attr_name]
            else:
                del f[path]
            self.modified = True
        finally:
            pass  # Keep file open for more operations

    def create_group(self, parent_path: str, name: str):
        """Create a new group."""
        self.create_backup()
        f = self.open("r+")
        full_path = f"{parent_path.rstrip('/')}/{name}"
        f.create_group(full_path)
        self.modified = True
        return full_path

    def create_dataset(
        self,
        parent_path: str,
        name: str,
        shape: Tuple,
        dtype: str,
        data: Optional[np.ndarray] = None,
    ):
        """Create a new dataset."""
        self.create_backup()
        f = self.open("r+")
        full_path = f"{parent_path.rstrip('/')}/{name}"
        if data is not None:
            f.create_dataset(full_path, data=data)
        else:
            f.create_dataset(full_path, shape=shape, dtype=dtype)
        self.modified = True
        return full_path

    def set_attribute(self, path: str, attr_name: str, value: Any):
        """Set an attribute value."""
        self.create_backup()
        f = self.open("r+")
        if path == "/":
            f.attrs[attr_name] = value
        else:
            f[path].attrs[attr_name] = value
        self.modified = True

    def set_dataset(self, path: str, data: Any, preserve_attrs: bool = True):
        """Replace dataset with new data."""
        self.create_backup()
        f = self.open("r+")

        dataset = f[path]

        # Preserve attributes if requested
        attrs = dict(dataset.attrs) if preserve_attrs else {}
        chunks = dataset.chunks
        compression = dataset.compression

        # Convert data to numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Check if we can resize in place (same dtype, compatible shape)
        if dataset.dtype == data.dtype and dataset.maxshape is not None:
            # Check if maxshape allows the new shape
            can_resize = all(
                (mx is None or s <= mx) for s, mx in zip(data.shape, dataset.maxshape)
            )
            if can_resize and len(data.shape) == len(dataset.shape):
                try:
                    dataset.resize(data.shape)
                    dataset[...] = data
                    self.modified = True
                    return
                except Exception:
                    pass

        # Fall back to delete and recreate
        del f[path]
        ds = f.create_dataset(path, data=data, chunks=chunks, compression=compression)

        # Restore attributes
        for k, v in attrs.items():
            ds.attrs[k] = v

        self.modified = True

    def get_dataset_info(self, path: str) -> Dict:
        """Get detailed information about a dataset."""
        with h5py.File(self.filepath, "r") as f:
            dataset = f[path]
            return {
                "shape": dataset.shape,
                "dtype": str(dataset.dtype),
                "size": dataset.size,
                "nbytes": dataset.nbytes,
                "chunks": dataset.chunks,
                "compression": dataset.compression,
                "maxshape": dataset.maxshape,
            }

    def cut_dataset(
        self,
        path: str,
        axis: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):
        """Cut dataset along axis."""
        self.create_backup()
        f = self.open("r+")

        dataset = f[path]
        shape = dataset.shape

        # Build slice
        slices = [slice(None)] * len(shape)
        slices[axis] = slice(start, end)

        # Read sliced data
        new_data = dataset[tuple(slices)]

        # Get attributes to preserve
        attrs = dict(dataset.attrs)
        dtype = dataset.dtype
        chunks = dataset.chunks
        compression = dataset.compression

        # Delete old dataset
        del f[path]

        # Create new dataset with sliced data
        ds = f.create_dataset(
            path, data=new_data, chunks=chunks, compression=compression
        )

        # Restore attributes
        for k, v in attrs.items():
            ds.attrs[k] = v

        self.modified = True

    def subsample_dataset(self, path: str, axis: int, step: int):
        """Subsample dataset along axis."""
        self.create_backup()
        f = self.open("r+")

        dataset = f[path]
        shape = dataset.shape

        # Build slice with step
        slices = [slice(None)] * len(shape)
        slices[axis] = slice(None, None, step)

        # Read subsampled data
        new_data = dataset[tuple(slices)]

        # Get attributes to preserve
        attrs = dict(dataset.attrs)
        chunks = dataset.chunks
        compression = dataset.compression

        # Delete old dataset
        del f[path]

        # Create new dataset
        ds = f.create_dataset(
            path, data=new_data, chunks=chunks, compression=compression
        )

        # Restore attributes
        for k, v in attrs.items():
            ds.attrs[k] = v

        self.modified = True

    def random_sample_dataset(self, path: str, axis: int, num_samples: int, seed: Optional[int] = None):
        """Random sample from dataset along axis."""
        self.create_backup()
        f = self.open("r+")

        dataset = f[path]
        shape = dataset.shape
        axis_size = shape[axis]

        if num_samples >= axis_size:
            return  # Nothing to do

        # Generate random indices
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(axis_size, size=num_samples, replace=False))

        # Build index tuple
        idx = [slice(None)] * len(shape)
        idx[axis] = indices

        # Read sampled data
        new_data = dataset[tuple(idx)]

        # Get attributes to preserve
        attrs = dict(dataset.attrs)
        chunks = dataset.chunks
        compression = dataset.compression

        # Delete old dataset
        del f[path]

        # Create new dataset
        ds = f.create_dataset(
            path, data=new_data, chunks=chunks, compression=compression
        )

        # Restore attributes
        for k, v in attrs.items():
            ds.attrs[k] = v

        self.modified = True

    def time_to_index(self, seconds: float) -> int:
        """Convert seconds to index using config timestep."""
        return int(seconds / self.config.timestep)


# =============================================================================
# Custom Widgets
# =============================================================================


class HDF5Tree(Tree):
    """Tree widget for HDF5 file structure with multi-selection support."""

    # Use different CSS class for selected nodes
    DEFAULT_CSS = """
    HDF5Tree .selected-node {
        background: $success 30%;
    }
    HDF5Tree .selected-node:focus {
        background: $success 50%;
    }
    """

    class NodeSelected(Message):
        """Message sent when a node is selected."""

        def __init__(self, node_data: HDF5NodeData):
            self.node_data = node_data
            super().__init__()

    class SelectionChanged(Message):
        """Message sent when multi-selection changes."""

        def __init__(self, selected: List[HDF5NodeData]):
            self.selected = selected
            super().__init__()

    def __init__(self, hdf5_ops: HDF5Operations, **kwargs):
        super().__init__("HDF5 File", **kwargs)
        self.hdf5_ops = hdf5_ops
        self.guide_depth = 3
        # Multi-selection support
        self.multi_select_mode = False
        self.selected_nodes: Dict[str, HDF5NodeData] = {}  # path -> node_data
        self._node_map: Dict[str, TreeNode] = {}  # path -> tree_node

    def on_mount(self):
        """Load tree structure when mounted."""
        self.load_structure()

    def load_structure(self):
        """Load HDF5 structure into tree."""
        self.clear()
        self._node_map.clear()
        structure = self.hdf5_ops.get_tree_structure()

        # Set root label
        root_label = Text(f" {self.hdf5_ops.filepath.name}", style=f"bold {Colors.GROUP}")
        self.root.set_label(root_label)
        self.root.data = HDF5NodeData(path="/", node_type="root")
        self._node_map["/"] = self.root

        self._add_node_children(self.root, structure)
        self.root.expand()

    def _add_node_children(self, tree_node: TreeNode, structure: Dict):
        """Recursively add children to tree node."""
        # Add attributes first
        for attr_name, attr_value in structure.get("attributes", {}).items():
            attr_label = self._format_attribute(attr_name, attr_value, selected=False)
            attr_node = tree_node.add_leaf(attr_label)
            attr_path = f"{structure.get('path', '/')}@{attr_name}"
            attr_node.data = HDF5NodeData(
                path=structure.get("path", "/"),
                node_type="attribute",
                attr_name=attr_name,
            )
            self._node_map[attr_path] = attr_node

        # Add children (groups and datasets)
        for name, child in structure.get("children", {}).items():
            if child["type"] == "group":
                label = self._format_group(name, selected=False)
                child_node = tree_node.add(label, expand=False)
                child_node.data = HDF5NodeData(
                    path=child["path"],
                    node_type="group",
                )
                self._node_map[child["path"]] = child_node
                self._add_node_children(child_node, child)
            elif child["type"] == "dataset":
                label = self._format_dataset(
                    name, child["shape"], child["dtype"], selected=False
                )
                child_node = tree_node.add_leaf(label)
                child_node.data = HDF5NodeData(
                    path=child["path"],
                    node_type="dataset",
                    shape=child["shape"],
                    dtype=child["dtype"],
                )
                self._node_map[child["path"]] = child_node

    def _get_node_key(self, node_data: HDF5NodeData) -> str:
        """Get unique key for a node."""
        if node_data.node_type == "attribute":
            return f"{node_data.path}@{node_data.attr_name}"
        return node_data.path

    def toggle_select(self, node: TreeNode):
        """Toggle selection of a node."""
        if node.data is None:
            return

        key = self._get_node_key(node.data)
        if key in self.selected_nodes:
            del self.selected_nodes[key]
            node.remove_class("selected-node")
        else:
            self.selected_nodes[key] = node.data
            node.add_class("selected-node")

        self._update_node_label(node)
        self.post_message(self.SelectionChanged(list(self.selected_nodes.values())))

    def add_to_selection(self, node: TreeNode):
        """Add a node to selection."""
        if node.data is None:
            return

        key = self._get_node_key(node.data)
        if key not in self.selected_nodes:
            self.selected_nodes[key] = node.data
            node.add_class("selected-node")
            self._update_node_label(node)
            self.post_message(self.SelectionChanged(list(self.selected_nodes.values())))

    def remove_from_selection(self, node: TreeNode):
        """Remove a node from selection."""
        if node.data is None:
            return

        key = self._get_node_key(node.data)
        if key in self.selected_nodes:
            del self.selected_nodes[key]
            node.remove_class("selected-node")
            self._update_node_label(node)
            self.post_message(self.SelectionChanged(list(self.selected_nodes.values())))

    def clear_selection(self):
        """Clear all selections."""
        old_keys = list(self.selected_nodes.keys())
        self.selected_nodes.clear()

        # Update all previously selected nodes
        for key in old_keys:
            if key in self._node_map:
                self._node_map[key].remove_class("selected-node")
                self._update_node_label(self._node_map[key])

        self.post_message(self.SelectionChanged([]))

    def _update_node_label(self, node: TreeNode):
        """Update node label to reflect selection state."""
        if node.data is None:
            return

        key = self._get_node_key(node.data)
        selected = key in self.selected_nodes

        if node.data.node_type == "group":
            name = node.data.path.split("/")[-1] or node.data.path
            label = self._format_group(name, selected=selected)
        elif node.data.node_type == "dataset":
            name = node.data.path.split("/")[-1]
            label = self._format_dataset(name, node.data.shape, node.data.dtype, selected=selected)
        elif node.data.node_type == "attribute":
            try:
                value = self.hdf5_ops.get_attribute(node.data.path, node.data.attr_name)
            except Exception:
                value = "???"
            label = self._format_attribute(node.data.attr_name, value, selected=selected)
        else:
            return

        node.set_label(label)

    def _format_group(self, name: str, selected: bool = False) -> Text:
        """Format group label."""
        prefix = "[*] " if selected else "[G] "
        style = "reverse" if selected else ""
        return Text.assemble(
            (prefix, f"bold {Colors.GROUP} {style}"),
            (name, f"{Colors.GROUP} {style}"),
        )

    def _format_dataset(self, name: str, shape: Tuple, dtype: str, selected: bool = False) -> Text:
        """Format dataset label."""
        dtype_color = Colors.for_dtype(dtype)
        shape_str = "x".join(str(s) for s in shape) if shape else "scalar"
        prefix = "[*] " if selected else "[D] "
        style = "reverse" if selected else ""
        return Text.assemble(
            (prefix, f"bold {Colors.DATASET} {style}"),
            (name, f"{Colors.DATASET} {style}"),
            (" (", f"dim {style}"),
            (shape_str, f"{dtype_color} {style}"),
            (") ", f"dim {style}"),
            (dtype, f"{dtype_color} {style}"),
        )

    def _format_attribute(self, name: str, value: Any, selected: bool = False) -> Text:
        """Format attribute label."""
        value_str = str(value)
        if len(value_str) > 50:
            value_str = value_str[:47] + "..."
        prefix = "[*] " if selected else "[@] "
        style = "reverse" if selected else ""
        return Text.assemble(
            (prefix, f"bold {Colors.ATTRIBUTE} {style}"),
            (name, f"{Colors.ATTRIBUTE} {style}"),
            (": ", f"dim {style}"),
            (value_str, f"white {style}"),
        )

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        """Handle node selection."""
        if event.node.data:
            self.post_message(self.NodeSelected(event.node.data))

    def get_selected_datasets(self) -> List[HDF5NodeData]:
        """Get only selected datasets."""
        return [n for n in self.selected_nodes.values() if n.node_type == "dataset"]


class OperationLog(RichLog):
    """Log widget for displaying operations."""

    def log_operation(self, op: Operation):
        """Log an operation."""
        status = "[green]OK[/green]" if op.success else "[red]FAIL[/red]"
        self.write(f"[dim]{op.timestamp}[/dim] [{status}] {op.name}: {op.target}")
        if op.details:
            self.write(f"  [dim]{op.details}[/dim]")
        if op.error:
            self.write(f"  [red]{op.error}[/red]")


class InfoPanel(Static):
    """Panel showing information about selected node."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.node_data: Optional[HDF5NodeData] = None

    def update_info(self, node_data: Optional[HDF5NodeData], hdf5_ops: HDF5Operations):
        """Update panel with node information."""
        self.node_data = node_data

        if node_data is None:
            self.update("Select an item to see details")
            return

        lines = []
        lines.append(f"[bold]Path:[/bold] {node_data.path}")
        lines.append(f"[bold]Type:[/bold] {node_data.node_type}")

        if node_data.node_type == "dataset":
            lines.append(f"[bold]Shape:[/bold] {node_data.shape}")
            lines.append(f"[bold]Dtype:[/bold] {node_data.dtype}")

            # Show preview
            try:
                data = hdf5_ops.get_dataset_slice(
                    node_data.path, max_elements=10
                )
                if data.size <= 10:
                    preview = str(data)
                else:
                    preview = str(data.flat[:10]) + "..."
                lines.append(f"[bold]Preview:[/bold] {preview}")
            except Exception as e:
                lines.append(f"[bold]Preview:[/bold] [red]Error: {e}[/red]")

        elif node_data.node_type == "attribute":
            try:
                value = hdf5_ops.get_attribute(node_data.path, node_data.attr_name)
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:197] + "..."
                lines.append(f"[bold]Value:[/bold] {value_str}")
            except Exception as e:
                lines.append(f"[bold]Value:[/bold] [red]Error: {e}[/red]")

        self.update("\n".join(lines))


class SelectionPanel(Static):
    """Panel showing multi-selected items."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_items: List[HDF5NodeData] = []

    def update_selection(self, selected: List[HDF5NodeData]):
        """Update panel with selected items."""
        self.selected_items = selected

        if not selected:
            self.update("[dim]No items selected[/dim]\n[dim]Press 'm' to enter selection mode[/dim]")
            return

        lines = []
        lines.append(f"[bold]Selected: {len(selected)} items[/bold]")

        # Count by type
        datasets = [s for s in selected if s.node_type == "dataset"]
        groups = [s for s in selected if s.node_type == "group"]
        attrs = [s for s in selected if s.node_type == "attribute"]

        if datasets:
            lines.append(f"  [green]Datasets:[/green] {len(datasets)}")
        if groups:
            lines.append(f"  [blue]Groups:[/blue] {len(groups)}")
        if attrs:
            lines.append(f"  [yellow]Attributes:[/yellow] {len(attrs)}")

        lines.append("")

        # Show first few items
        for i, item in enumerate(selected[:8]):
            if item.node_type == "dataset":
                shape_str = "x".join(str(s) for s in item.shape) if item.shape else "scalar"
                lines.append(f"[green]{item.path}[/green] ({shape_str})")
            elif item.node_type == "attribute":
                lines.append(f"[yellow]{item.path}@{item.attr_name}[/yellow]")
            else:
                lines.append(f"[blue]{item.path}[/blue]")

        if len(selected) > 8:
            lines.append(f"[dim]... and {len(selected) - 8} more[/dim]")

        # Check if batch operations are available
        if len(datasets) >= 2:
            lines.append("")
            lines.append("[bold]Batch operations available:[/bold]")
            lines.append("  [cyan]Shift+C[/cyan] - Cut all")
            lines.append("  [cyan]Shift+S[/cyan] - Subsample all")
            lines.append("  [cyan]Shift+R[/cyan] - Random sample all")

        self.update("\n".join(lines))

    def get_compatible_axis_info(self) -> Optional[Dict]:
        """Check if selected datasets have compatible axes for batch operations."""
        datasets = [s for s in self.selected_items if s.node_type == "dataset"]

        if len(datasets) < 2:
            return None

        # Build axis info for each dataset
        axis_info = {}
        for ds in datasets:
            if not ds.shape:
                continue
            for axis, size in enumerate(ds.shape):
                if axis not in axis_info:
                    axis_info[axis] = {"sizes": set(), "datasets": []}
                axis_info[axis]["sizes"].add(size)
                axis_info[axis]["datasets"].append(ds)

        # Find axes where all datasets have the same size
        compatible_axes = {}
        for axis, info in axis_info.items():
            if len(info["sizes"]) == 1 and len(info["datasets"]) == len(datasets):
                compatible_axes[axis] = {
                    "size": list(info["sizes"])[0],
                    "count": len(info["datasets"]),
                }

        return compatible_axes if compatible_axes else None


# =============================================================================
# Dialog Screens
# =============================================================================


class ConfirmDialog(ModalScreen):
    """Confirmation dialog."""

    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm-dialog"):
            yield Label(self.message, id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", variant="error", id="yes")
                yield Button("No", variant="primary", id="no")

    def on_button_pressed(self, event: Button.Pressed):
        self.dismiss(event.button.id == "yes")


class InputDialog(ModalScreen):
    """Simple input dialog."""

    def __init__(
        self,
        title: str,
        prompt: str,
        default: str = "",
        placeholder: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title = title
        self.prompt = prompt
        self.default = default
        self.placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Container(id="input-dialog"):
            yield Label(self.title, id="dialog-title")
            yield Label(self.prompt)
            yield Input(value=self.default, placeholder=self.placeholder, id="input")
            with Horizontal(id="dialog-buttons"):
                yield Button("OK", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self.query_one("#input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "ok":
            value = self.query_one("#input", Input).value
            self.dismiss(value)
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted):
        self.dismiss(event.value)


class StringEditDialog(ModalScreen):
    """Dialog for editing string values with a text area."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, title: str, current_value: Any, **kwargs):
        super().__init__(**kwargs)
        self.title_text = title
        # Convert bytes to string if needed
        if isinstance(current_value, bytes):
            try:
                self.current_value = current_value.decode("utf-8")
            except UnicodeDecodeError:
                self.current_value = current_value.hex()
        else:
            self.current_value = str(current_value) if current_value is not None else ""

    def compose(self) -> ComposeResult:
        with Container(id="string-edit-dialog"):
            yield Label(self.title_text, id="dialog-title")
            yield Label("[dim]Edit text content below:[/dim]")
            yield TextArea(self.current_value, id="text-editor", language=None)
            yield Label("[dim]Ctrl+S to save, Escape to cancel[/dim]", id="edit-hint")
            with Horizontal(id="dialog-buttons"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self.query_one("#text-editor", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "save":
            text = self.query_one("#text-editor", TextArea).text
            self.dismiss({"value": text, "type": "string"})
        else:
            self.dismiss(None)

    def action_cancel(self):
        self.dismiss(None)


class BoolEditDialog(ModalScreen):
    """Dialog for editing boolean values with a switch."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, title: str, current_value: Any, **kwargs):
        super().__init__(**kwargs)
        self.title_text = title
        # Convert to bool
        if isinstance(current_value, (bool, np.bool_)):
            self.current_value = bool(current_value)
        else:
            self.current_value = bool(current_value) if current_value else False

    def compose(self) -> ComposeResult:
        with Container(id="bool-edit-dialog"):
            yield Label(self.title_text, id="dialog-title")
            yield Label("")
            with Horizontal(id="switch-container"):
                yield Label("Value: ", id="switch-label")
                yield Switch(value=self.current_value, id="bool-switch")
                yield Label("True" if self.current_value else "False", id="switch-value-label")
            yield Label("")
            with Horizontal(id="dialog-buttons"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self.query_one("#bool-switch", Switch).focus()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Update label when switch changes."""
        label = self.query_one("#switch-value-label", Label)
        label.update("True" if event.value else "False")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "save":
            value = self.query_one("#bool-switch", Switch).value
            self.dismiss({"value": value, "type": "bool"})
        else:
            self.dismiss(None)

    def action_cancel(self):
        self.dismiss(None)


class NumberEditDialog(ModalScreen):
    """Dialog for editing numeric values using a data table."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        title: str,
        current_value: np.ndarray,
        dtype: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title_text = title
        self.original_value = np.array(current_value)
        self.dtype = dtype or str(self.original_value.dtype)
        self.original_shape = self.original_value.shape
        # For >2D, we need to select view axes
        self.view_axis_0 = 0
        self.view_axis_1 = 1 if len(self.original_shape) > 1 else 0
        # Working data - sliced for display
        self._editing_data = None

    def compose(self) -> ComposeResult:
        with Container(id="number-edit-dialog"):
            yield Label(self.title_text, id="dialog-title")

            shape_str = "x".join(str(s) for s in self.original_shape) if self.original_shape else "scalar"
            yield Label(f"[bold]Shape:[/bold] {shape_str}  [bold]Dtype:[/bold] {self.dtype}", id="shape-info")

            # For >2D arrays, show axis selection
            if len(self.original_shape) > 2:
                yield Label("")
                yield Label("[yellow]Array has more than 2 dimensions. Select 2 axes to view/edit:[/yellow]")
                with Horizontal(id="axis-selection"):
                    yield Label("Row axis: ")
                    axis_options = [(f"Axis {i} (size {s})", str(i)) for i, s in enumerate(self.original_shape)]
                    yield Select(axis_options, value="0", id="row-axis")
                    yield Label("  Column axis: ")
                    yield Select(axis_options, value="1", id="col-axis")
                yield Label("[dim]Other axes will be fixed at index 0[/dim]")

            yield Label("")
            yield DataTable(id="number-table", cursor_type="cell")
            yield Label("[dim]Double-click or press Enter to edit a cell. Tab to move.[/dim]", id="table-hint")

            with Horizontal(id="dialog-buttons"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self._load_data_into_table()
        self.query_one("#number-table", DataTable).focus()

    def _load_data_into_table(self):
        """Load data into the table based on selected axes."""
        table = self.query_one("#number-table", DataTable)
        table.clear(columns=True)

        data = self.original_value

        # Handle different dimensions
        if data.ndim == 0:
            # Scalar
            table.add_column("Value", key="val")
            table.add_row(str(data.item()), key="0")
            self._editing_data = data.reshape(1)
        elif data.ndim == 1:
            # 1D
            table.add_column("Index", key="idx")
            table.add_column("Value", key="val")
            max_rows = min(500, len(data))
            for i in range(max_rows):
                table.add_row(str(i), str(data[i]), key=str(i))
            self._editing_data = data[:max_rows].copy()
        elif data.ndim == 2:
            # 2D - show directly
            max_cols = min(20, data.shape[1])
            max_rows = min(500, data.shape[0])

            for j in range(max_cols):
                table.add_column(f"[{j}]", key=f"c{j}")

            for i in range(max_rows):
                row_data = [str(data[i, j]) for j in range(max_cols)]
                table.add_row(*row_data, key=str(i))

            self._editing_data = data[:max_rows, :max_cols].copy()
        else:
            # >2D - need to slice
            # Get axis selections
            try:
                row_axis = int(self.query_one("#row-axis", Select).value)
                col_axis = int(self.query_one("#col-axis", Select).value)
            except Exception:
                row_axis = 0
                col_axis = 1

            if row_axis == col_axis:
                col_axis = (row_axis + 1) % len(data.shape)

            self.view_axis_0 = row_axis
            self.view_axis_1 = col_axis

            # Create slice that selects first element of all other axes
            slice_tuple = [0] * len(data.shape)
            slice_tuple[row_axis] = slice(None)
            slice_tuple[col_axis] = slice(None)

            sliced = data[tuple(slice_tuple)]

            # Now sliced is 2D, display it
            max_cols = min(20, sliced.shape[1] if sliced.ndim > 1 else 1)
            max_rows = min(500, sliced.shape[0])

            if sliced.ndim == 1:
                table.add_column("Value", key="c0")
                for i in range(max_rows):
                    table.add_row(str(sliced[i]), key=str(i))
                self._editing_data = sliced[:max_rows].copy()
            else:
                for j in range(max_cols):
                    table.add_column(f"[{j}]", key=f"c{j}")
                for i in range(max_rows):
                    row_data = [str(sliced[i, j]) for j in range(max_cols)]
                    table.add_row(*row_data, key=str(i))
                self._editing_data = sliced[:max_rows, :max_cols].copy()

    @on(Select.Changed)
    def on_axis_changed(self, event: Select.Changed):
        """Reload table when axis selection changes."""
        if event.select.id in ("row-axis", "col-axis"):
            self._load_data_into_table()

    @on(DataTable.CellSelected)
    def on_cell_selected(self, event: DataTable.CellSelected):
        """Handle cell selection for editing."""
        # The DataTable doesn't have built-in cell editing,
        # so we'll use a simple Input dialog for editing
        pass

    async def _edit_cell(self, row_key: str, col_key: str, current_value: str):
        """Edit a single cell value."""
        result = await self.app.push_screen_wait(
            InputDialog("Edit Cell", f"Enter new value:", default=current_value)
        )
        if result is not None:
            table = self.query_one("#number-table", DataTable)
            try:
                # Parse value based on dtype
                if "int" in self.dtype.lower():
                    new_val = int(float(result))
                elif "bool" in self.dtype.lower():
                    new_val = result.lower() in ("true", "1", "yes")
                else:
                    new_val = float(result)

                # Update table display
                table.update_cell(row_key, col_key, str(new_val))

                # Update editing data
                row_idx = int(row_key)
                col_idx = int(col_key[1:]) if col_key.startswith("c") else 0

                if self._editing_data.ndim == 1:
                    self._editing_data[row_idx] = new_val
                else:
                    self._editing_data[row_idx, col_idx] = new_val

            except ValueError as e:
                self.notify(f"Invalid value: {e}", severity="error")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "save":
            # For scalar/1D/2D, return the edited data
            # For >2D, we can only edit a slice, so return original with slice updated
            if self.original_value.ndim <= 2:
                result = self._editing_data
                # Resize to match original if needed
                if result.shape != self.original_shape:
                    # Only return what we edited
                    final = self.original_value.copy()
                    if final.ndim == 1:
                        final[:len(result)] = result
                    else:
                        final[:result.shape[0], :result.shape[1]] = result
                    result = final
            else:
                # For >2D, update the slice in the original
                result = self.original_value.copy()
                slice_tuple = [0] * len(result.shape)
                slice_tuple[self.view_axis_0] = slice(None)
                slice_tuple[self.view_axis_1] = slice(None)

                sliced_shape = result[tuple(slice_tuple)].shape
                if self._editing_data.shape == sliced_shape:
                    result[tuple(slice_tuple)] = self._editing_data
                else:
                    # Partial update
                    if self._editing_data.ndim == 1:
                        result[tuple(slice_tuple)][:len(self._editing_data)] = self._editing_data
                    else:
                        result[tuple(slice_tuple)][:self._editing_data.shape[0], :self._editing_data.shape[1]] = self._editing_data

            self.dismiss({"value": result, "type": "array"})
        else:
            self.dismiss(None)

    def action_cancel(self):
        self.dismiss(None)


class ScalarNumberDialog(ModalScreen):
    """Dialog for editing scalar numeric values."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, title: str, current_value: Any, dtype: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.title_text = title
        self.current_value = current_value
        self.dtype = dtype

    def compose(self) -> ComposeResult:
        with Container(id="scalar-edit-dialog"):
            yield Label(self.title_text, id="dialog-title")

            dtype_str = self.dtype if self.dtype else type(self.current_value).__name__
            yield Label(f"[bold]Type:[/bold] {dtype_str}", id="type-info")

            yield Label("Value:")
            yield Input(value=str(self.current_value), id="value-input", type="number" if "int" in str(self.dtype or "").lower() or "float" in str(self.dtype or "").lower() else "text")

            yield Label("[dim]Enter a numeric value[/dim]", id="edit-hint")

            with Horizontal(id="dialog-buttons"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self.query_one("#value-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "save":
            text = self.query_one("#value-input", Input).value
            try:
                dtype_str = str(self.dtype or "").lower()
                if "int" in dtype_str:
                    value = int(float(text))
                elif "float" in dtype_str:
                    value = float(text)
                else:
                    # Try to infer
                    if "." in text or "e" in text.lower():
                        value = float(text)
                    else:
                        value = int(text)
                self.dismiss({"value": value, "type": "number"})
            except ValueError as e:
                self.notify(f"Invalid number: {e}", severity="error")
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted):
        """Handle Enter key in input."""
        self.on_button_pressed(Button.Pressed(Button("save", id="save")))

    def action_cancel(self):
        self.dismiss(None)


class AddItemDialog(ModalScreen):
    """Dialog for adding new items."""

    def __init__(self, parent_path: str, **kwargs):
        super().__init__(**kwargs)
        self.parent_path = parent_path

    def compose(self) -> ComposeResult:
        with Container(id="add-dialog"):
            yield Label("Add New Item", id="dialog-title")
            yield Label(f"Parent: {self.parent_path}")

            yield Label("Type:")
            with RadioSet(id="item-type"):
                yield RadioButton("Group", id="group", value=True)
                yield RadioButton("Dataset", id="dataset")
                yield RadioButton("Attribute", id="attribute")

            yield Label("Name:")
            yield Input(placeholder="item_name", id="name")

            # Dataset-specific fields
            yield Label("Shape (comma-separated):", id="shape-label")
            yield Input(placeholder="100, 50, 3", id="shape")

            yield Label("Dtype:", id="dtype-label")
            yield Select(
                [
                    ("float32", "float32"),
                    ("float64", "float64"),
                    ("int32", "int32"),
                    ("int64", "int64"),
                    ("bool", "bool"),
                ],
                value="float32",
                id="dtype",
            )

            # Attribute-specific fields
            yield Label("Value:", id="value-label")
            yield Input(placeholder="attribute value", id="value")

            with Horizontal(id="dialog-buttons"):
                yield Button("Create", variant="primary", id="create")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self._update_visibility()
        self.query_one("#name", Input).focus()

    def on_radio_set_changed(self, event: RadioSet.Changed):
        self._update_visibility()

    def _update_visibility(self):
        """Show/hide fields based on selected type."""
        radio_set = self.query_one("#item-type", RadioSet)
        selected = radio_set.pressed_button
        if selected is None:
            return

        item_type = selected.id

        # Dataset fields
        for elem_id in ["shape-label", "shape", "dtype-label", "dtype"]:
            self.query_one(f"#{elem_id}").display = item_type == "dataset"

        # Attribute fields
        for elem_id in ["value-label", "value"]:
            self.query_one(f"#{elem_id}").display = item_type == "attribute"

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "create":
            radio_set = self.query_one("#item-type", RadioSet)
            selected = radio_set.pressed_button
            item_type = selected.id if selected else "group"
            name = self.query_one("#name", Input).value

            if not name:
                self.notify("Name is required", severity="error")
                return

            result = {"type": item_type, "name": name, "parent": self.parent_path}

            if item_type == "dataset":
                shape_str = self.query_one("#shape", Input).value
                dtype = self.query_one("#dtype", Select).value
                try:
                    shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
                except ValueError:
                    self.notify("Invalid shape format", severity="error")
                    return
                result["shape"] = shape
                result["dtype"] = dtype

            elif item_type == "attribute":
                result["value"] = self.query_one("#value", Input).value

            self.dismiss(result)
        else:
            self.dismiss(None)


class CutDialog(ModalScreen):
    """Dialog for cut operation."""

    def __init__(self, shape: Tuple, config: EditorConfig, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.config = config

    def compose(self) -> ComposeResult:
        with Container(id="cut-dialog"):
            yield Label("Cut Along Axis", id="dialog-title")
            yield Label(f"Shape: {self.shape}")
            yield Label(f"Timestep: {self.config.timestep}s")

            yield Label("Axis:")
            axis_options = [(f"Axis {i} (size {s})", str(i)) for i, s in enumerate(self.shape)]
            yield Select(axis_options, value="0" if axis_options else None, id="axis")

            yield Label("Start (int=index, float=seconds, empty=0):")
            yield Input(placeholder="0", id="start")

            yield Label("End (int=index, float=seconds, empty=max):")
            yield Input(placeholder="", id="end")

            with Horizontal(id="dialog-buttons"):
                yield Button("Cut", variant="warning", id="cut")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cut":
            axis = int(self.query_one("#axis", Select).value)
            start_str = self.query_one("#start", Input).value.strip()
            end_str = self.query_one("#end", Input).value.strip()

            start = None
            end = None

            try:
                if start_str:
                    if "." in start_str:
                        start = int(float(start_str) / self.config.timestep)
                    else:
                        start = int(start_str)

                if end_str:
                    if "." in end_str:
                        end = int(float(end_str) / self.config.timestep)
                    else:
                        end = int(end_str)
            except ValueError:
                self.notify("Invalid number format", severity="error")
                return

            self.dismiss({"axis": axis, "start": start, "end": end})
        else:
            self.dismiss(None)


class SubsampleDialog(ModalScreen):
    """Dialog for subsample operation."""

    def __init__(self, shape: Tuple, config: EditorConfig, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.config = config

    def compose(self) -> ComposeResult:
        with Container(id="subsample-dialog"):
            yield Label("Subsample Along Axis", id="dialog-title")
            yield Label(f"Shape: {self.shape}")
            yield Label(f"Timestep: {self.config.timestep}s")

            yield Label("Axis:")
            axis_options = [(f"Axis {i} (size {s})", str(i)) for i, s in enumerate(self.shape)]
            yield Select(axis_options, value="0" if axis_options else None, id="axis")

            yield Label("Rate (int=every N samples, float=every N seconds):")
            yield Input(placeholder="2", id="rate")

            with Horizontal(id="dialog-buttons"):
                yield Button("Subsample", variant="warning", id="subsample")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "subsample":
            axis = int(self.query_one("#axis", Select).value)
            rate_str = self.query_one("#rate", Input).value.strip()

            try:
                if "." in rate_str:
                    rate = max(1, int(float(rate_str) / self.config.timestep))
                else:
                    rate = int(rate_str)

                if rate < 1:
                    self.notify("Rate must be at least 1", severity="error")
                    return
            except ValueError:
                self.notify("Invalid rate format", severity="error")
                return

            self.dismiss({"axis": axis, "rate": rate})
        else:
            self.dismiss(None)


class RandomSampleDialog(ModalScreen):
    """Dialog for random sample operation."""

    def __init__(self, shape: Tuple, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def compose(self) -> ComposeResult:
        with Container(id="random-dialog"):
            yield Label("Random Sample", id="dialog-title")
            yield Label(f"Shape: {self.shape}")

            yield Label("Axis (batch dimension):")
            axis_options = [(f"Axis {i} (size {s})", str(i)) for i, s in enumerate(self.shape)]
            yield Select(axis_options, value="0" if axis_options else None, id="axis")

            yield Label("Number of samples:")
            yield Input(placeholder="100", id="num-samples")

            yield Label("Random seed (optional):")
            yield Input(placeholder="", id="seed")

            with Horizontal(id="dialog-buttons"):
                yield Button("Sample", variant="warning", id="sample")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "sample":
            axis = int(self.query_one("#axis", Select).value)
            num_str = self.query_one("#num-samples", Input).value.strip()
            seed_str = self.query_one("#seed", Input).value.strip()

            try:
                num_samples = int(num_str)
                if num_samples < 1:
                    self.notify("Need at least 1 sample", severity="error")
                    return
            except ValueError:
                self.notify("Invalid number", severity="error")
                return

            seed = None
            if seed_str:
                try:
                    seed = int(seed_str)
                except ValueError:
                    self.notify("Invalid seed", severity="error")
                    return

            self.dismiss({"axis": axis, "num_samples": num_samples, "seed": seed})
        else:
            self.dismiss(None)


class BatchCutDialog(ModalScreen):
    """Dialog for batch cut operation on multiple datasets."""

    def __init__(
        self,
        datasets: List[HDF5NodeData],
        compatible_axes: Dict[int, Dict],
        config: EditorConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.datasets = datasets
        self.compatible_axes = compatible_axes
        self.config = config

    def compose(self) -> ComposeResult:
        with Container(id="batch-cut-dialog"):
            yield Label("Batch Cut - Multiple Datasets", id="dialog-title")
            yield Label(f"[bold]Datasets:[/bold] {len(self.datasets)}")

            # List datasets
            dataset_list = ", ".join(d.path.split("/")[-1] for d in self.datasets[:5])
            if len(self.datasets) > 5:
                dataset_list += f" (+{len(self.datasets) - 5} more)"
            yield Label(f"[dim]{dataset_list}[/dim]")

            yield Label(f"[bold]Timestep:[/bold] {self.config.timestep}s")

            yield Label("")
            yield Label("Select axis (only axes with matching sizes shown):")

            # Build axis options from compatible axes
            axis_options = []
            for axis, info in sorted(self.compatible_axes.items()):
                axis_options.append(
                    (f"Axis {axis} (size {info['size']}, {info['count']} datasets)", str(axis))
                )

            if axis_options:
                yield Select(axis_options, value=axis_options[0][1], id="axis")
            else:
                yield Label("[red]No compatible axes found![/red]")

            yield Label("Start (int=index, float=seconds, empty=0):")
            yield Input(placeholder="0", id="start")

            yield Label("End (int=index, float=seconds, empty=max):")
            yield Input(placeholder="", id="end")

            with Horizontal(id="dialog-buttons"):
                yield Button("Cut All", variant="warning", id="cut")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cut":
            try:
                axis = int(self.query_one("#axis", Select).value)
            except Exception:
                self.notify("No axis selected", severity="error")
                return

            start_str = self.query_one("#start", Input).value.strip()
            end_str = self.query_one("#end", Input).value.strip()

            start = None
            end = None

            try:
                if start_str:
                    if "." in start_str:
                        start = int(float(start_str) / self.config.timestep)
                    else:
                        start = int(start_str)

                if end_str:
                    if "." in end_str:
                        end = int(float(end_str) / self.config.timestep)
                    else:
                        end = int(end_str)
            except ValueError:
                self.notify("Invalid number format", severity="error")
                return

            self.dismiss({"axis": axis, "start": start, "end": end})
        else:
            self.dismiss(None)


class BatchSubsampleDialog(ModalScreen):
    """Dialog for batch subsample operation on multiple datasets."""

    def __init__(
        self,
        datasets: List[HDF5NodeData],
        compatible_axes: Dict[int, Dict],
        config: EditorConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.datasets = datasets
        self.compatible_axes = compatible_axes
        self.config = config

    def compose(self) -> ComposeResult:
        with Container(id="batch-subsample-dialog"):
            yield Label("Batch Subsample - Multiple Datasets", id="dialog-title")
            yield Label(f"[bold]Datasets:[/bold] {len(self.datasets)}")

            dataset_list = ", ".join(d.path.split("/")[-1] for d in self.datasets[:5])
            if len(self.datasets) > 5:
                dataset_list += f" (+{len(self.datasets) - 5} more)"
            yield Label(f"[dim]{dataset_list}[/dim]")

            yield Label(f"[bold]Timestep:[/bold] {self.config.timestep}s")

            yield Label("")
            yield Label("Select axis (only axes with matching sizes shown):")

            axis_options = []
            for axis, info in sorted(self.compatible_axes.items()):
                axis_options.append(
                    (f"Axis {axis} (size {info['size']}, {info['count']} datasets)", str(axis))
                )

            if axis_options:
                yield Select(axis_options, value=axis_options[0][1], id="axis")
            else:
                yield Label("[red]No compatible axes found![/red]")

            yield Label("Rate (int=every N samples, float=every N seconds):")
            yield Input(placeholder="2", id="rate")

            with Horizontal(id="dialog-buttons"):
                yield Button("Subsample All", variant="warning", id="subsample")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "subsample":
            try:
                axis = int(self.query_one("#axis", Select).value)
            except Exception:
                self.notify("No axis selected", severity="error")
                return

            rate_str = self.query_one("#rate", Input).value.strip()

            try:
                if "." in rate_str:
                    rate = max(1, int(float(rate_str) / self.config.timestep))
                else:
                    rate = int(rate_str)

                if rate < 1:
                    self.notify("Rate must be at least 1", severity="error")
                    return
            except ValueError:
                self.notify("Invalid rate format", severity="error")
                return

            self.dismiss({"axis": axis, "rate": rate})
        else:
            self.dismiss(None)


class BatchRandomSampleDialog(ModalScreen):
    """Dialog for batch random sample operation on multiple datasets."""

    def __init__(
        self,
        datasets: List[HDF5NodeData],
        compatible_axes: Dict[int, Dict],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.datasets = datasets
        self.compatible_axes = compatible_axes

    def compose(self) -> ComposeResult:
        with Container(id="batch-random-dialog"):
            yield Label("Batch Random Sample - Multiple Datasets", id="dialog-title")
            yield Label(f"[bold]Datasets:[/bold] {len(self.datasets)}")

            dataset_list = ", ".join(d.path.split("/")[-1] for d in self.datasets[:5])
            if len(self.datasets) > 5:
                dataset_list += f" (+{len(self.datasets) - 5} more)"
            yield Label(f"[dim]{dataset_list}[/dim]")

            yield Label("")
            yield Label("Select axis (only axes with matching sizes shown):")

            axis_options = []
            for axis, info in sorted(self.compatible_axes.items()):
                axis_options.append(
                    (f"Axis {axis} (size {info['size']}, {info['count']} datasets)", str(axis))
                )

            if axis_options:
                yield Select(axis_options, value=axis_options[0][1], id="axis")
            else:
                yield Label("[red]No compatible axes found![/red]")

            yield Label("Number of samples:")
            yield Input(placeholder="100", id="num-samples")

            yield Label("Random seed (optional, same seed for all datasets):")
            yield Input(placeholder="", id="seed")

            with Horizontal(id="dialog-buttons"):
                yield Button("Sample All", variant="warning", id="sample")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "sample":
            try:
                axis = int(self.query_one("#axis", Select).value)
            except Exception:
                self.notify("No axis selected", severity="error")
                return

            num_str = self.query_one("#num-samples", Input).value.strip()
            seed_str = self.query_one("#seed", Input).value.strip()

            try:
                num_samples = int(num_str)
                if num_samples < 1:
                    self.notify("Need at least 1 sample", severity="error")
                    return
            except ValueError:
                self.notify("Invalid number", severity="error")
                return

            seed = None
            if seed_str:
                try:
                    seed = int(seed_str)
                except ValueError:
                    self.notify("Invalid seed", severity="error")
                    return

            self.dismiss({"axis": axis, "num_samples": num_samples, "seed": seed})
        else:
            self.dismiss(None)


class DataViewDialog(ModalScreen):
    """Dialog for viewing dataset data."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    def __init__(self, path: str, hdf5_ops: HDF5Operations, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.hdf5_ops = hdf5_ops

    def compose(self) -> ComposeResult:
        with Container(id="view-dialog"):
            yield Label(f"Data: {self.path}", id="dialog-title")
            yield DataTable(id="data-table")
            yield Label("Use arrow keys to scroll. Press 'q' or Escape to close.", id="view-hint")

    def on_mount(self):
        """Load data into table."""
        table = self.query_one("#data-table", DataTable)
        table.cursor_type = "row"

        try:
            data = self.hdf5_ops.get_dataset_slice(
                self.path, max_elements=self.hdf5_ops.config.max_table_rows * 100
            )

            # Handle different dimensions
            if data.ndim == 0:
                # Scalar
                table.add_column("Value")
                table.add_row(str(data.item()))
            elif data.ndim == 1:
                # 1D array
                table.add_column("Index")
                table.add_column("Value")
                for i, val in enumerate(data[: self.hdf5_ops.config.max_table_rows]):
                    table.add_row(str(i), str(val))
            elif data.ndim == 2:
                # 2D array
                for j in range(min(data.shape[1], 20)):
                    table.add_column(f"[{j}]")
                for i in range(min(data.shape[0], self.hdf5_ops.config.max_table_rows)):
                    row = [str(v) for v in data[i, :20]]
                    table.add_row(*row)
            else:
                # Higher dimensions - flatten to 2D view
                table.add_column("Index")
                table.add_column("Shape")
                table.add_column("Values (first 10)")
                flat = data.reshape(data.shape[0], -1)
                for i in range(min(flat.shape[0], self.hdf5_ops.config.max_table_rows)):
                    vals = ", ".join(str(v) for v in flat[i, :10])
                    table.add_row(str(i), str(data.shape[1:]), vals + "...")

        except Exception as e:
            table.add_column("Error")
            table.add_row(str(e))

    def action_dismiss(self):
        self.app.pop_screen()


class ConfigScreen(ModalScreen):
    """Configuration screen."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]

    def __init__(self, config: EditorConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def compose(self) -> ComposeResult:
        with Container(id="config-dialog"):
            yield Label("Configuration", id="dialog-title")

            yield Label("Timestep (seconds):")
            yield Input(value=str(self.config.timestep), id="timestep")

            yield Label("Backup suffix:")
            yield Input(value=self.config.backup_suffix, id="backup-suffix")

            yield Label("Max preview size:")
            yield Input(value=str(self.config.max_preview_size), id="max-preview")

            yield Label("Max table rows:")
            yield Input(value=str(self.config.max_table_rows), id="max-rows")

            with Horizontal(id="dialog-buttons"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "save":
            try:
                self.config.timestep = float(self.query_one("#timestep", Input).value)
                self.config.backup_suffix = self.query_one("#backup-suffix", Input).value
                self.config.max_preview_size = int(
                    self.query_one("#max-preview", Input).value
                )
                self.config.max_table_rows = int(self.query_one("#max-rows", Input).value)
                self.dismiss(self.config)
            except ValueError as e:
                self.notify(f"Invalid value: {e}", severity="error")
        else:
            self.dismiss(None)

    def action_dismiss(self):
        self.app.pop_screen()


class HelpScreen(ModalScreen):
    """Help screen showing key bindings."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-dialog"):
            yield Label("HDF5 Editor - Help", id="dialog-title")
            yield Static(
                """
[bold]Navigation[/bold]
  Arrow keys    Navigate tree
  Enter         Expand/collapse or edit
  Tab           Switch panels

[bold]Operations[/bold]
  a             Add new item (group/dataset/attribute)
  d             Delete selected item
  e             Edit selected item (attributes AND datasets)
  v             View data in table

[bold]Type-Specific Editors[/bold]
  Strings       Full text editor (TextArea)
  Booleans      Toggle switch (True/False)
  Numbers       Input field for scalars
  Num Arrays    Table view with cell editing
  >2D Arrays    Select 2 axes to view/edit

[bold]Array Operations (Single Dataset)[/bold]
  c             Cut along axis
  s             Subsample along axis
  r             Random sample from axis

[bold]Multi-Selection Mode[/bold]
  m             Toggle selection mode ON/OFF
  Space         Toggle select current item
  Shift+Space   Add to selection
  Shift+Z       Deselect current item
  Escape        Clear all selections

[bold]Batch Operations (Multiple Datasets)[/bold]
  Shift+C       Batch cut (all selected datasets)
  Shift+S       Batch subsample (all selected)
  Shift+R       Batch random sample (all selected)
  Note: Only axes with matching sizes are available

[bold]Panels[/bold]
  F             Toggle operation panel fullscreen
  Ctrl+C        Open configuration

[bold]File[/bold]
  Ctrl+S        Save changes
  q             Quit (prompts to save)

[bold]Index Format for Cut/Subsample[/bold]
  int           Direct index (e.g., 100)
  float         Seconds (e.g., 1.5 = 1.5s / timestep)

Press 'q' or Escape to close this help.
""",
                id="help-content",
            )

    def action_dismiss(self):
        self.app.pop_screen()


# =============================================================================
# Main Application
# =============================================================================


class HDF5EditorApp(App):
    """Main HDF5 Editor Application."""

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        layout: horizontal;
    }

    #tree-container {
        width: 60%;
        height: 100%;
        border: solid $primary;
    }

    #right-panel {
        width: 40%;
        height: 100%;
        layout: vertical;
    }

    #info-panel {
        height: auto;
        min-height: 5;
        max-height: 10;
        border: solid $secondary;
        padding: 1;
    }

    #selection-panel {
        height: auto;
        min-height: 3;
        max-height: 15;
        border: solid $success;
        padding: 1;
    }

    #selection-panel.hidden {
        display: none;
    }

    #operation-top {
        height: 1fr;
        border: solid $accent;
    }

    #operation-bottom {
        height: 1fr;
        border: solid $warning;
    }

    #batch-cut-dialog, #batch-subsample-dialog, #batch-random-dialog {
        width: 70;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }

    .select-mode-indicator {
        background: $warning;
        color: $text;
        padding: 0 1;
    }

    .fullscreen {
        width: 100%;
        height: 100%;
    }

    #confirm-dialog, #input-dialog, #add-dialog, #cut-dialog,
    #subsample-dialog, #random-dialog, #config-dialog {
        width: 60;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }

    #string-edit-dialog {
        width: 80%;
        height: 80%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }

    #string-edit-dialog #text-editor {
        height: 1fr;
        min-height: 20;
    }

    #bool-edit-dialog {
        width: 50;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }

    #switch-container {
        height: auto;
        align: center middle;
    }

    #switch-label {
        width: auto;
    }

    #switch-value-label {
        width: auto;
        margin-left: 1;
        color: $success;
    }

    #number-edit-dialog {
        width: 90%;
        height: 85%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }

    #number-edit-dialog #number-table {
        height: 1fr;
        min-height: 15;
    }

    #axis-selection {
        height: auto;
        margin-bottom: 1;
    }

    #axis-selection Select {
        width: 20;
    }

    #scalar-edit-dialog {
        width: 50;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }

    #edit-help {
        margin-top: 1;
        margin-bottom: 1;
    }

    #type-info, #shape-info {
        color: $text-muted;
        margin-bottom: 1;
    }

    #table-hint {
        color: $text-muted;
        margin-top: 1;
    }

    #view-dialog {
        width: 90%;
        height: 90%;
        padding: 1;
        border: thick $primary;
        background: $surface;
    }

    #help-dialog {
        width: 70;
        height: 80%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }

    #dialog-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #dialog-buttons {
        margin-top: 1;
        height: auto;
    }

    #dialog-buttons Button {
        margin-right: 1;
    }

    #confirm-buttons {
        margin-top: 1;
        height: auto;
    }

    #confirm-buttons Button {
        margin-right: 1;
    }

    #data-table {
        height: 1fr;
    }

    #view-hint {
        height: auto;
        color: $text-muted;
    }

    OperationLog {
        height: 1fr;
    }

    Input {
        margin-bottom: 1;
    }

    Select {
        margin-bottom: 1;
    }

    RadioSet {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("a", "add_item", "Add"),
        Binding("d", "delete_item", "Delete"),
        Binding("e", "edit_item", "Edit"),
        Binding("v", "view_data", "View"),
        Binding("c", "cut_data", "Cut"),
        Binding("s", "subsample_data", "Subsample"),
        Binding("r", "random_sample", "Random"),
        Binding("ctrl+c", "show_config", "Config"),
        Binding("f", "toggle_fullscreen", "Fullscreen"),
        Binding("ctrl+s", "save_file", "Save"),
        Binding("question_mark", "show_help", "Help"),
        # Multi-selection bindings
        Binding("m", "toggle_select_mode", "Select Mode"),
        Binding("space", "toggle_selection", "Toggle Select", show=False),
        Binding("shift+space", "add_selection", "Add to Select", show=False),
        Binding("shift+z", "deselect_current", "Deselect", show=False),
        Binding("escape", "clear_selection", "Clear Selection", show=False),
        # Batch operations (capital letters)
        Binding("shift+c", "batch_cut", "Batch Cut", show=False),
        Binding("shift+s", "batch_subsample", "Batch Subsample", show=False),
        Binding("shift+r", "batch_random", "Batch Random", show=False),
    ]

    operation_panel_fullscreen = reactive(False)
    multi_select_mode = reactive(False)

    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = filepath
        self.config = EditorConfig()
        self.hdf5_ops = HDF5Operations(filepath, self.config)
        self.selected_node: Optional[HDF5NodeData] = None
        self.operations: List[Operation] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Container(id="tree-container"):
                yield HDF5Tree(self.hdf5_ops, id="hdf5-tree")
            with Vertical(id="right-panel"):
                yield InfoPanel(id="info-panel")
                yield SelectionPanel(id="selection-panel", classes="hidden")
                with Container(id="operation-top"):
                    yield Label("Operations", id="op-top-label")
                    yield OperationLog(id="operation-log-top", highlight=True, markup=True)
                with Container(id="operation-bottom"):
                    yield Label("History", id="op-bottom-label")
                    yield OperationLog(id="operation-log-bottom", highlight=True, markup=True)
        yield Footer()

    def on_mount(self):
        """Initialize app state."""
        self.title = f"HDF5 Editor - {self.filepath.name}"
        self.sub_title = "Press ? for help | m=Select Mode"

    def watch_multi_select_mode(self, value: bool):
        """React to multi-select mode changes."""
        tree = self.query_one("#hdf5-tree", HDF5Tree)
        tree.multi_select_mode = value
        selection_panel = self.query_one("#selection-panel", SelectionPanel)

        if value:
            self.sub_title = "SELECT MODE | Space=Toggle | Shift+Z=Deselect | Esc=Clear | m=Exit"
            selection_panel.remove_class("hidden")
        else:
            self.sub_title = "Press ? for help | m=Select Mode"
            # Keep panel visible if there are selections
            if not tree.selected_nodes:
                selection_panel.add_class("hidden")

    @on(HDF5Tree.SelectionChanged)
    def handle_selection_changed(self, event: HDF5Tree.SelectionChanged):
        """Handle multi-selection changes."""
        selection_panel = self.query_one("#selection-panel", SelectionPanel)
        selection_panel.update_selection(event.selected)

        # Show panel if items selected, hide if empty and not in select mode
        if event.selected:
            selection_panel.remove_class("hidden")
        elif not self.multi_select_mode:
            selection_panel.add_class("hidden")

    @on(HDF5Tree.NodeSelected)
    def handle_node_selected(self, event: HDF5Tree.NodeSelected):
        """Handle node selection in tree."""
        self.selected_node = event.node_data
        info_panel = self.query_one("#info-panel", InfoPanel)
        info_panel.update_info(event.node_data, self.hdf5_ops)

    def _log_operation(self, op: Operation, to_top: bool = True):
        """Log an operation to the appropriate panel."""
        self.operations.append(op)
        log_id = "#operation-log-top" if to_top else "#operation-log-bottom"
        log = self.query_one(log_id, OperationLog)
        log.log_operation(op)

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime

        return datetime.now().strftime("%H:%M:%S")

    def _refresh_tree(self):
        """Refresh the tree view."""
        tree = self.query_one("#hdf5-tree", HDF5Tree)
        tree.load_structure()

    @work
    async def action_quit_app(self):
        """Quit with save prompt if modified."""
        if self.hdf5_ops.modified:
            result = await self.push_screen_wait(
                ConfirmDialog("File has been modified. Save before quitting?")
            )
            if result:
                self._do_save_file()
        self.hdf5_ops.close()
        self.exit()

    def action_save_file(self):
        """Save changes to file."""
        self._do_save_file()

    def _do_save_file(self):
        """Internal save implementation."""
        try:
            self.hdf5_ops.close()  # Close to flush changes
            self._log_operation(
                Operation(
                    name="Save",
                    target=str(self.filepath),
                    details="File saved",
                    timestamp=self._get_timestamp(),
                    success=True,
                ),
                to_top=False,
            )
            self.notify("File saved", severity="information")
        except Exception as e:
            self._log_operation(
                Operation(
                    name="Save",
                    target=str(self.filepath),
                    details="",
                    timestamp=self._get_timestamp(),
                    success=False,
                    error=str(e),
                ),
                to_top=False,
            )
            self.notify(f"Save failed: {e}", severity="error")

    @work
    async def action_add_item(self):
        """Add new item dialog."""
        if self.selected_node is None:
            self.notify("Select a parent node first", severity="warning")
            return

        if self.selected_node.node_type == "attribute":
            parent_path = self.selected_node.path
        elif self.selected_node.node_type in ("group", "root"):
            parent_path = self.selected_node.path
        else:
            # Dataset - use parent
            parent_path = "/".join(self.selected_node.path.split("/")[:-1]) or "/"

        result = await self.push_screen_wait(AddItemDialog(parent_path))
        if result:
            try:
                if result["type"] == "group":
                    self.hdf5_ops.create_group(result["parent"], result["name"])
                elif result["type"] == "dataset":
                    self.hdf5_ops.create_dataset(
                        result["parent"],
                        result["name"],
                        result["shape"],
                        result["dtype"],
                    )
                elif result["type"] == "attribute":
                    path = result["parent"]
                    self.hdf5_ops.set_attribute(path, result["name"], result["value"])

                self._log_operation(
                    Operation(
                        name="Add",
                        target=f"{result['parent']}/{result['name']}",
                        details=f"Type: {result['type']}",
                        timestamp=self._get_timestamp(),
                    )
                )
                self._refresh_tree()
                self.notify(f"Created {result['type']}: {result['name']}")
            except Exception as e:
                self._log_operation(
                    Operation(
                        name="Add",
                        target=f"{result['parent']}/{result['name']}",
                        details="",
                        timestamp=self._get_timestamp(),
                        success=False,
                        error=str(e),
                    )
                )
                self.notify(f"Failed to create: {e}", severity="error")

    @work
    async def action_delete_item(self):
        """Delete selected item."""
        if self.selected_node is None:
            self.notify("Select an item to delete", severity="warning")
            return

        if self.selected_node.node_type == "root":
            self.notify("Cannot delete root", severity="error")
            return

        target = self.selected_node.path
        if self.selected_node.node_type == "attribute":
            target = f"{self.selected_node.path}@{self.selected_node.attr_name}"

        result = await self.push_screen_wait(
            ConfirmDialog(f"Delete {target}?")
        )
        if result:
            try:
                if self.selected_node.node_type == "attribute":
                    self.hdf5_ops.delete_item(
                        self.selected_node.path, self.selected_node.attr_name
                    )
                else:
                    self.hdf5_ops.delete_item(self.selected_node.path)

                self._log_operation(
                    Operation(
                        name="Delete",
                        target=target,
                        details="",
                        timestamp=self._get_timestamp(),
                    )
                )
                self._refresh_tree()
                self.notify(f"Deleted: {target}")
            except Exception as e:
                self._log_operation(
                    Operation(
                        name="Delete",
                        target=target,
                        details="",
                        timestamp=self._get_timestamp(),
                        success=False,
                        error=str(e),
                    )
                )
                self.notify(f"Failed to delete: {e}", severity="error")

    def _infer_value_type(self, value: Any) -> str:
        """Infer the type category of a value."""
        if isinstance(value, (bool, np.bool_)):
            return "bool"
        if isinstance(value, (int, np.integer)):
            return "int"
        if isinstance(value, (float, np.floating)):
            return "float"
        if isinstance(value, (str, bytes)):
            return "string"
        if isinstance(value, np.ndarray):
            # Check array dtype
            dtype_str = str(value.dtype).lower()
            if "bool" in dtype_str:
                return "bool_array"
            if "str" in dtype_str or "object" in dtype_str or dtype_str.startswith("|s") or dtype_str.startswith("<u"):
                return "string_array"
            return "number_array"
        return "unknown"

    @work
    async def action_edit_item(self):
        """Edit selected item with type-specific dialog."""
        if self.selected_node is None:
            self.notify("Select an item to edit", severity="warning")
            return

        if self.selected_node.node_type == "attribute":
            # Edit attribute
            current_value = self.hdf5_ops.get_attribute(
                self.selected_node.path, self.selected_node.attr_name
            )
            title = f"Edit Attribute: {self.selected_node.attr_name}"

            # Choose dialog based on type
            value_type = self._infer_value_type(current_value)
            result = await self._show_edit_dialog(title, current_value, value_type, None)

            if result is not None:
                try:
                    new_value = result["value"]
                    self.hdf5_ops.set_attribute(
                        self.selected_node.path,
                        self.selected_node.attr_name,
                        new_value,
                    )
                    self._log_operation(
                        Operation(
                            name="Edit",
                            target=f"{self.selected_node.path}@{self.selected_node.attr_name}",
                            details=f"Type: {type(new_value).__name__}",
                            timestamp=self._get_timestamp(),
                        )
                    )
                    self._refresh_tree()
                    self.notify("Attribute updated")
                except Exception as e:
                    self._log_operation(
                        Operation(
                            name="Edit",
                            target=f"{self.selected_node.path}@{self.selected_node.attr_name}",
                            details="",
                            timestamp=self._get_timestamp(),
                            success=False,
                            error=str(e),
                        )
                    )
                    self.notify(f"Failed to update: {e}", severity="error")

        elif self.selected_node.node_type == "dataset":
            # Edit dataset
            info = self.hdf5_ops.get_dataset_info(self.selected_node.path)
            dtype = self.selected_node.dtype

            if info["size"] > 50000:
                # Large dataset - ask if they want to proceed
                proceed = await self.push_screen_wait(
                    ConfirmDialog(
                        f"Dataset has {info['size']} elements. "
                        f"Loading may be slow. Proceed?"
                    )
                )
                if not proceed:
                    return

            current_value = self.hdf5_ops.get_dataset_slice(
                self.selected_node.path, max_elements=100000
            )
            title = f"Edit Dataset: {self.selected_node.path}"

            # Determine value type based on dtype
            dtype_lower = dtype.lower() if dtype else ""
            if "bool" in dtype_lower:
                if current_value.size == 1:
                    value_type = "bool"
                else:
                    value_type = "bool_array"
            elif "str" in dtype_lower or "object" in dtype_lower or dtype_lower.startswith("|s") or dtype_lower.startswith("<u"):
                value_type = "string_array" if current_value.size > 1 else "string"
            elif current_value.size == 1:
                value_type = "number_scalar"
            else:
                value_type = "number_array"

            result = await self._show_edit_dialog(title, current_value, value_type, dtype)

            if result is not None:
                try:
                    new_value = result["value"]
                    self.hdf5_ops.set_dataset(self.selected_node.path, new_value)
                    self._log_operation(
                        Operation(
                            name="Edit",
                            target=self.selected_node.path,
                            details=f"Shape: {np.array(new_value).shape}, dtype: {np.array(new_value).dtype}",
                            timestamp=self._get_timestamp(),
                        )
                    )
                    self._refresh_tree()
                    self.notify("Dataset updated")
                except Exception as e:
                    self._log_operation(
                        Operation(
                            name="Edit",
                            target=self.selected_node.path,
                            details="",
                            timestamp=self._get_timestamp(),
                            success=False,
                            error=str(e),
                        )
                    )
                    self.notify(f"Failed to update: {e}", severity="error")

        elif self.selected_node.node_type in ("group", "root"):
            self.notify("Cannot edit groups directly. Use add/delete operations.", severity="warning")
        else:
            self.notify(f"Unknown node type: {self.selected_node.node_type}", severity="error")

    async def _show_edit_dialog(self, title: str, value: Any, value_type: str, dtype: Optional[str]):
        """Show the appropriate edit dialog based on value type."""
        if value_type == "bool":
            return await self.push_screen_wait(BoolEditDialog(title, value))

        elif value_type == "string":
            return await self.push_screen_wait(StringEditDialog(title, value))

        elif value_type in ("int", "float", "number_scalar"):
            # Scalar number
            scalar_val = value.item() if isinstance(value, np.ndarray) else value
            return await self.push_screen_wait(ScalarNumberDialog(title, scalar_val, dtype))

        elif value_type == "number_array":
            # Numeric array - use table editor
            return await self.push_screen_wait(NumberEditDialog(title, value, dtype))

        elif value_type == "string_array":
            # String array - use text editor with each element on a line
            if isinstance(value, np.ndarray):
                text = "\n".join(str(v) for v in value.flat)
            else:
                text = str(value)
            result = await self.push_screen_wait(StringEditDialog(title, text))
            if result:
                # Convert back to array
                lines = result["value"].split("\n")
                result["value"] = np.array(lines)
            return result

        elif value_type == "bool_array":
            # Bool array - use number table (shows as 0/1)
            return await self.push_screen_wait(NumberEditDialog(title, value.astype(int), dtype))

        else:
            # Fallback to string editor
            return await self.push_screen_wait(StringEditDialog(title, str(value)))

    def action_view_data(self):
        """View dataset data."""
        if self.selected_node is None or self.selected_node.node_type != "dataset":
            self.notify("Select a dataset to view", severity="warning")
            return

        self.push_screen(DataViewDialog(self.selected_node.path, self.hdf5_ops))

    @work
    async def action_cut_data(self):
        """Cut dataset along axis."""
        if self.selected_node is None or self.selected_node.node_type != "dataset":
            self.notify("Select a dataset to cut", severity="warning")
            return

        if not self.selected_node.shape:
            self.notify("Cannot cut scalar dataset", severity="error")
            return

        result = await self.push_screen_wait(
            CutDialog(self.selected_node.shape, self.config)
        )
        if result:
            try:
                self.hdf5_ops.cut_dataset(
                    self.selected_node.path,
                    result["axis"],
                    result["start"],
                    result["end"],
                )
                self._log_operation(
                    Operation(
                        name="Cut",
                        target=self.selected_node.path,
                        details=f"axis={result['axis']}, [{result['start']}:{result['end']}]",
                        timestamp=self._get_timestamp(),
                    )
                )
                self._refresh_tree()
                self.notify("Dataset cut successfully")
            except Exception as e:
                self._log_operation(
                    Operation(
                        name="Cut",
                        target=self.selected_node.path,
                        details="",
                        timestamp=self._get_timestamp(),
                        success=False,
                        error=str(e),
                    )
                )
                self.notify(f"Cut failed: {e}", severity="error")

    @work
    async def action_subsample_data(self):
        """Subsample dataset."""
        if self.selected_node is None or self.selected_node.node_type != "dataset":
            self.notify("Select a dataset to subsample", severity="warning")
            return

        if not self.selected_node.shape:
            self.notify("Cannot subsample scalar dataset", severity="error")
            return

        result = await self.push_screen_wait(
            SubsampleDialog(self.selected_node.shape, self.config)
        )
        if result:
            try:
                self.hdf5_ops.subsample_dataset(
                    self.selected_node.path, result["axis"], result["rate"]
                )
                self._log_operation(
                    Operation(
                        name="Subsample",
                        target=self.selected_node.path,
                        details=f"axis={result['axis']}, rate={result['rate']}",
                        timestamp=self._get_timestamp(),
                    )
                )
                self._refresh_tree()
                self.notify("Dataset subsampled successfully")
            except Exception as e:
                self._log_operation(
                    Operation(
                        name="Subsample",
                        target=self.selected_node.path,
                        details="",
                        timestamp=self._get_timestamp(),
                        success=False,
                        error=str(e),
                    )
                )
                self.notify(f"Subsample failed: {e}", severity="error")

    @work
    async def action_random_sample(self):
        """Random sample from dataset."""
        if self.selected_node is None or self.selected_node.node_type != "dataset":
            self.notify("Select a dataset to sample", severity="warning")
            return

        if not self.selected_node.shape:
            self.notify("Cannot sample scalar dataset", severity="error")
            return

        result = await self.push_screen_wait(
            RandomSampleDialog(self.selected_node.shape)
        )
        if result:
            try:
                self.hdf5_ops.random_sample_dataset(
                    self.selected_node.path,
                    result["axis"],
                    result["num_samples"],
                    result.get("seed"),
                )
                self._log_operation(
                    Operation(
                        name="Random",
                        target=self.selected_node.path,
                        details=f"axis={result['axis']}, n={result['num_samples']}",
                        timestamp=self._get_timestamp(),
                    )
                )
                self._refresh_tree()
                self.notify("Random sample successful")
            except Exception as e:
                self._log_operation(
                    Operation(
                        name="Random",
                        target=self.selected_node.path,
                        details="",
                        timestamp=self._get_timestamp(),
                        success=False,
                        error=str(e),
                    )
                )
                self.notify(f"Random sample failed: {e}", severity="error")

    @work
    async def action_show_config(self):
        """Show configuration screen."""
        result = await self.push_screen_wait(ConfigScreen(self.config))
        if result:
            self.config = result
            self.hdf5_ops.config = result
            self.notify("Configuration updated")

    def action_toggle_fullscreen(self):
        """Toggle fullscreen for operation panels."""
        self.operation_panel_fullscreen = not self.operation_panel_fullscreen
        tree_container = self.query_one("#tree-container")
        right_panel = self.query_one("#right-panel")

        if self.operation_panel_fullscreen:
            tree_container.display = False
            right_panel.styles.width = "100%"
        else:
            tree_container.display = True
            right_panel.styles.width = "40%"

    def action_show_help(self):
        """Show help screen."""
        self.push_screen(HelpScreen())

    # =========================================================================
    # Multi-Selection Actions
    # =========================================================================

    def action_toggle_select_mode(self):
        """Toggle multi-selection mode."""
        self.multi_select_mode = not self.multi_select_mode
        if self.multi_select_mode:
            self.notify("Selection mode ON - Space to select, Shift+Z to deselect")
        else:
            self.notify("Selection mode OFF")

    def action_toggle_selection(self):
        """Toggle selection of current node (Space key)."""
        tree = self.query_one("#hdf5-tree", HDF5Tree)
        if tree.cursor_node:
            tree.toggle_select(tree.cursor_node)

    def action_add_selection(self):
        """Add current node to selection (Shift+Space)."""
        tree = self.query_one("#hdf5-tree", HDF5Tree)
        if tree.cursor_node:
            tree.add_to_selection(tree.cursor_node)

    def action_deselect_current(self):
        """Remove current node from selection (Shift+Z)."""
        tree = self.query_one("#hdf5-tree", HDF5Tree)
        if tree.cursor_node:
            tree.remove_from_selection(tree.cursor_node)

    def action_clear_selection(self):
        """Clear all selections (Escape)."""
        tree = self.query_one("#hdf5-tree", HDF5Tree)
        if tree.selected_nodes:
            tree.clear_selection()
            self.notify("Selection cleared")

    def _get_compatible_axes(self, datasets: List[HDF5NodeData]) -> Dict[int, Dict]:
        """Get axes with matching sizes across all datasets."""
        if len(datasets) < 2:
            return {}

        # Build axis info for each dataset
        axis_info = {}
        for ds in datasets:
            if not ds.shape:
                continue
            for axis, size in enumerate(ds.shape):
                if axis not in axis_info:
                    axis_info[axis] = {"sizes": set(), "datasets": []}
                axis_info[axis]["sizes"].add(size)
                axis_info[axis]["datasets"].append(ds)

        # Find axes where all datasets have the same size
        compatible_axes = {}
        for axis, info in axis_info.items():
            if len(info["sizes"]) == 1 and len(info["datasets"]) == len(datasets):
                compatible_axes[axis] = {
                    "size": list(info["sizes"])[0],
                    "count": len(info["datasets"]),
                }

        return compatible_axes

    # =========================================================================
    # Batch Operations
    # =========================================================================

    @work
    async def action_batch_cut(self):
        """Batch cut operation on selected datasets (Shift+C)."""
        tree = self.query_one("#hdf5-tree", HDF5Tree)
        datasets = tree.get_selected_datasets()

        if len(datasets) < 2:
            self.notify("Select at least 2 datasets for batch operations", severity="warning")
            return

        # Check for compatible axes
        compatible_axes = self._get_compatible_axes(datasets)
        if not compatible_axes:
            self.notify("No compatible axes found (need same size on at least one axis)", severity="error")
            return

        result = await self.push_screen_wait(
            BatchCutDialog(datasets, compatible_axes, self.config)
        )
        if result:
            success_count = 0
            fail_count = 0

            for ds in datasets:
                try:
                    self.hdf5_ops.cut_dataset(
                        ds.path,
                        result["axis"],
                        result["start"],
                        result["end"],
                    )
                    success_count += 1
                    self._log_operation(
                        Operation(
                            name="BatchCut",
                            target=ds.path,
                            details=f"axis={result['axis']}, [{result['start']}:{result['end']}]",
                            timestamp=self._get_timestamp(),
                        ),
                        to_top=False,
                    )
                except Exception as e:
                    fail_count += 1
                    self._log_operation(
                        Operation(
                            name="BatchCut",
                            target=ds.path,
                            details="",
                            timestamp=self._get_timestamp(),
                            success=False,
                            error=str(e),
                        ),
                        to_top=False,
                    )

            self._refresh_tree()
            tree.clear_selection()
            self.notify(f"Batch cut: {success_count} succeeded, {fail_count} failed")

    @work
    async def action_batch_subsample(self):
        """Batch subsample operation on selected datasets (Shift+S)."""
        tree = self.query_one("#hdf5-tree", HDF5Tree)
        datasets = tree.get_selected_datasets()

        if len(datasets) < 2:
            self.notify("Select at least 2 datasets for batch operations", severity="warning")
            return

        compatible_axes = self._get_compatible_axes(datasets)
        if not compatible_axes:
            self.notify("No compatible axes found (need same size on at least one axis)", severity="error")
            return

        result = await self.push_screen_wait(
            BatchSubsampleDialog(datasets, compatible_axes, self.config)
        )
        if result:
            success_count = 0
            fail_count = 0

            for ds in datasets:
                try:
                    self.hdf5_ops.subsample_dataset(
                        ds.path,
                        result["axis"],
                        result["rate"],
                    )
                    success_count += 1
                    self._log_operation(
                        Operation(
                            name="BatchSubsample",
                            target=ds.path,
                            details=f"axis={result['axis']}, rate={result['rate']}",
                            timestamp=self._get_timestamp(),
                        ),
                        to_top=False,
                    )
                except Exception as e:
                    fail_count += 1
                    self._log_operation(
                        Operation(
                            name="BatchSubsample",
                            target=ds.path,
                            details="",
                            timestamp=self._get_timestamp(),
                            success=False,
                            error=str(e),
                        ),
                        to_top=False,
                    )

            self._refresh_tree()
            tree.clear_selection()
            self.notify(f"Batch subsample: {success_count} succeeded, {fail_count} failed")

    @work
    async def action_batch_random(self):
        """Batch random sample operation on selected datasets (Shift+R)."""
        tree = self.query_one("#hdf5-tree", HDF5Tree)
        datasets = tree.get_selected_datasets()

        if len(datasets) < 2:
            self.notify("Select at least 2 datasets for batch operations", severity="warning")
            return

        compatible_axes = self._get_compatible_axes(datasets)
        if not compatible_axes:
            self.notify("No compatible axes found (need same size on at least one axis)", severity="error")
            return

        result = await self.push_screen_wait(
            BatchRandomSampleDialog(datasets, compatible_axes)
        )
        if result:
            success_count = 0
            fail_count = 0

            for ds in datasets:
                try:
                    self.hdf5_ops.random_sample_dataset(
                        ds.path,
                        result["axis"],
                        result["num_samples"],
                        result.get("seed"),
                    )
                    success_count += 1
                    self._log_operation(
                        Operation(
                            name="BatchRandom",
                            target=ds.path,
                            details=f"axis={result['axis']}, n={result['num_samples']}",
                            timestamp=self._get_timestamp(),
                        ),
                        to_top=False,
                    )
                except Exception as e:
                    fail_count += 1
                    self._log_operation(
                        Operation(
                            name="BatchRandom",
                            target=ds.path,
                            details="",
                            timestamp=self._get_timestamp(),
                            success=False,
                            error=str(e),
                        ),
                        to_top=False,
                    )

            self._refresh_tree()
            tree.clear_selection()
            self.notify(f"Batch random sample: {success_count} succeeded, {fail_count} failed")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Interactive HDF5 Terminal Editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python edit_hdf5.py data.hdf5
    python edit_hdf5.py /path/to/file.hdf5

Press '?' in the editor for help with key bindings.
        """,
    )
    parser.add_argument("file", type=str, help="HDF5 file to edit")

    args = parser.parse_args()
    filepath = Path(args.file)

    if not filepath.exists():
        print(f"Error: File '{filepath}' does not exist")
        sys.exit(1)

    if not filepath.suffix.lower() in (".hdf5", ".h5", ".hdf"):
        print(f"Warning: File '{filepath}' may not be an HDF5 file")

    app = HDF5EditorApp(filepath)
    app.run()


if __name__ == "__main__":
    main()
