#!/usr/bin/env python3
"""
HDF5 Structure Visualizer

This script analyzes HDF5 files and generates a hierarchical graph visualization
in PDF format using Graphviz. It handles all HDF5 features including:
- All atomic and composite data types
- Storage layouts (chunked, compressed, resizable)
- Links (hard, soft, external)
- Attributes
- Virtual datasets
- Dimension scales
- Filters and compression

Usage:
    python visualize_hdf5_structure.py <hdf5_file> [-o output_name]

Examples:
    python visualize_hdf5_structure.py data.hdf5
    python visualize_hdf5_structure.py data.hdf5 -o my_visualization

Requirements:
    - h5py
    - graphviz (Python package)
    - Graphviz (system binary - install via: brew install graphviz)

Visualization Legend:
    Nodes (boxes):
        - Light blue: Groups
        - Light green: Datasets
        - Plum: Virtual datasets

    Edges (arrows):
        - Gray solid: Normal hierarchy
        - Orange dashed: Soft link
        - Red dotted: External link
        - Black bold: Hard link
"""

import h5py
import argparse
from graphviz import Digraph
import os
import warnings

def get_storage_info(dataset):
    """Extract storage layout information from dataset."""
    info = {}
    if dataset.chunks:
        info['chunks'] = dataset.chunks
    if dataset.compression:
        info['compression'] = dataset.compression
        if dataset.compression_opts:
            info['compression_opts'] = dataset.compression_opts
    if dataset.shuffle:
        info['shuffle'] = True
    if dataset.fletcher32:
        info['fletcher32'] = True
    if dataset.scaleoffset is not None:
        info['scaleoffset'] = dataset.scaleoffset
    if dataset.fillvalue is not None:
        info['fillvalue'] = dataset.fillvalue
    if dataset.maxshape != dataset.shape:
        info['maxshape'] = dataset.maxshape
    return info

def get_link_info(file_obj, name):
    """Determine if item is a link and get link details."""
    parent_path = '/'.join(name.split('/')[:-1]) if '/' in name else '/'
    item_name = name.split('/')[-1]

    parent = file_obj[parent_path] if parent_path else file_obj
    link_info = parent.get(item_name, getlink=True)

    if isinstance(link_info, h5py.SoftLink):
        return {'type': 'SoftLink', 'target': link_info.path}
    elif isinstance(link_info, h5py.ExternalLink):
        return {'type': 'ExternalLink', 'filename': link_info.filename, 'target': link_info.path}
    elif isinstance(link_info, h5py.HardLink):
        obj = parent[item_name]
        if hasattr(obj, 'ref'):
            ref_count = file_obj[obj.ref]
            if hasattr(ref_count, '__len__'):
                return {'type': 'HardLink'}
    return None

def add_unresolved_links(f, group_path, structure, visited_names):
    """Add soft/external links and hard link aliases that couldn't be resolved by visititems."""
    try:
        group = f[group_path] if group_path != '/' else f
        for key in group.keys():
            full_path = f"{group_path}/{key}" if group_path != '/' else key

            if full_path in visited_names:
                continue

            link = group.get(key, getlink=True)

            if isinstance(link, h5py.SoftLink):
                parent_path = group_path
                structure.append({
                    'name': full_path,
                    'type': 'SoftLink',
                    'parent': parent_path,
                    'link': {'type': 'SoftLink', 'target': link.path},
                    'attributes': {}
                })
                visited_names.add(full_path)
            elif isinstance(link, h5py.ExternalLink):
                parent_path = group_path
                structure.append({
                    'name': full_path,
                    'type': 'ExternalLink',
                    'parent': parent_path,
                    'link': {'type': 'ExternalLink', 'filename': link.filename, 'target': link.path},
                    'attributes': {}
                })
                visited_names.add(full_path)
            elif isinstance(link, h5py.HardLink):
                # Add hard link aliases that weren't visited by visititems
                try:
                    obj = group[key]
                    parent_path = group_path

                    if isinstance(obj, h5py.Dataset):
                        structure.append({
                            'name': full_path,
                            'type': 'Dataset',
                            'shape': obj.shape,
                            'dtype': str(obj.dtype),
                            'parent': parent_path,
                            'link': {'type': 'HardLink'},
                            'attributes': dict(obj.attrs)
                        })
                    elif isinstance(obj, h5py.Group):
                        structure.append({
                            'name': full_path,
                            'type': 'Group',
                            'parent': parent_path,
                            'link': {'type': 'HardLink'},
                            'attributes': dict(obj.attrs)
                        })
                    visited_names.add(full_path)
                except Exception:
                    pass
    except Exception:
        pass

def analyze_hdf5_structure(filepath):
    """
    Analyze HDF5 file structure and return hierarchical information.

    Args:
        filepath: Path to HDF5 file

    Returns:
        Dictionary with structure information
    """
    structure = []
    visited_names = set()

    with h5py.File(filepath, 'r') as f:
        def visitor(name, obj):
            visited_names.add(name)

            link_info = None
            try:
                link_info = get_link_info(f, name)
            except Exception:
                pass

            attributes = {}
            try:
                attributes = dict(obj.attrs)
            except Exception:
                pass

            if isinstance(obj, h5py.Group):
                parent_path = '/'.join(name.split('/')[:-1]) if name and '/' in name else '/'
                if parent_path == '':
                    parent_path = '/'
                item_info = {
                    'name': name if name else '/',
                    'type': 'Group',
                    'parent': parent_path,
                    'attributes': attributes
                }
                if link_info:
                    item_info['link'] = link_info
                structure.append(item_info)

                add_unresolved_links(f, name, structure, visited_names)

            elif isinstance(obj, h5py.Dataset):
                parent_path = '/'.join(name.split('/')[:-1]) if '/' in name else '/'
                if parent_path == '':
                    parent_path = '/'
                item_info = {
                    'name': name,
                    'type': 'Dataset',
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'parent': parent_path,
                    'attributes': attributes
                }

                storage_info = get_storage_info(obj)
                if storage_info:
                    item_info['storage'] = storage_info

                if link_info:
                    item_info['link'] = link_info

                if obj.is_virtual:
                    item_info['virtual'] = True

                if hasattr(obj, 'dims') and len(obj.dims) > 0:
                    dim_info = []
                    for i, dim in enumerate(obj.dims):
                        if dim.label:
                            dim_info.append({'index': i, 'label': dim.label})
                    if dim_info:
                        item_info['dimensions'] = dim_info

                structure.append(item_info)

        root_attrs = {}
        try:
            root_attrs = dict(f.attrs)
        except Exception:
            pass

        structure.append({
            'name': '/',
            'type': 'Group',
            'parent': None,
            'attributes': root_attrs
        })
        visited_names.add('/')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                f.visititems(visitor)
            except Exception as e:
                print(f"Warning: Error during file traversal: {e}")

        add_unresolved_links(f, '/', structure, visited_names)

    return structure

def format_shape(shape):
    """Format shape tuple for display."""
    if not shape:
        return "scalar"
    return f"{'x'.join(map(str, shape))}"

def html_escape(text):
    """Escape special characters for HTML/GraphViz labels."""
    text = str(text)
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace('[', '&#91;')
    text = text.replace(']', '&#93;')
    return text

def format_attributes(attributes):
    """Format attributes for display."""
    if not attributes:
        return ""

    lines = []
    for key, value in attributes.items():
        key = html_escape(key)
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4g}")
            else:
                lines.append(f"{key}: {value}")
        else:
            value_str = str(value)
            if len(value_str) > 30:
                value_str = value_str[:27] + "..."
            value_str = html_escape(value_str)
            lines.append(f"{key}: {value_str}")

    return "\n".join(lines)

def format_storage_info(storage):
    """Format storage information for display."""
    lines = []
    if 'chunks' in storage:
        chunk_str = 'x'.join(map(str, storage['chunks']))
        lines.append(f"Chunks: {chunk_str}")
    if 'compression' in storage:
        comp_str = storage['compression']
        if 'compression_opts' in storage:
            comp_str += f"({storage['compression_opts']})"
        lines.append(f"Compression: {comp_str}")
    if 'maxshape' in storage:
        maxshape_str = 'x'.join('∞' if s is None else str(s) for s in storage['maxshape'])
        lines.append(f"MaxShape: {maxshape_str}")
    return "\n".join(lines)

def create_graph_visualization(structure, output_path):
    """
    Create hierarchical graph visualization using Graphviz.

    Args:
        structure: List of dictionaries with HDF5 structure information
        output_path: Path for output PDF file
    """
    dot = Digraph(comment='HDF5 File Structure', format='pdf')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Courier', fontsize='8')
    dot.attr('edge', color='gray40', arrowsize='0.7')
    dot.attr(dpi='150')

    for item in structure:
        node_name = item['name'] if item['name'] else 'root'
        display_name = node_name.split('/')[-1] if '/' in node_name else node_name

        if not display_name:
            display_name = 'ROOT'

        display_name = html_escape(display_name)

        attributes = item.get('attributes', {})
        attr_str = format_attributes(attributes)

        if item['type'] == 'Group':
            label = f"<{display_name}<BR/><I>(Group)</I>"
            if attr_str:
                label += "<BR/>---<BR/>Attributes:<BR/>"
                for line in attr_str.split('\n'):
                    label += f"<FONT POINT-SIZE='7'>{line}</FONT><BR/>"
            label += ">"
            dot.node(node_name, label, fillcolor='lightblue', penwidth='1.5')

        elif item['type'] == 'SoftLink':
            link_info = item.get('link', {})
            target = html_escape(link_info.get('target', ''))
            label = f"<{display_name}<BR/><I>(Soft Link)</I><BR/><FONT POINT-SIZE='7'>→ {target}</FONT>>"
            dot.node(node_name, label, fillcolor='lightyellow', penwidth='1.5', style='rounded,filled,dashed')

        elif item['type'] == 'ExternalLink':
            link_info = item.get('link', {})
            filename = html_escape(link_info.get('filename', ''))
            target = html_escape(link_info.get('target', ''))
            label = f"<{display_name}<BR/><I>(External Link)</I><BR/><FONT POINT-SIZE='7'>→ {filename}:{target}</FONT>>"
            dot.node(node_name, label, fillcolor='lightcoral', penwidth='1.5', style='rounded,filled,dotted')

        elif item['type'] == 'Dataset':
            shape_str = format_shape(item['shape'])
            dtype_str = html_escape(item['dtype'])

            label = f"<{display_name}<BR/><I>(Dataset)</I>"

            fillcolor = 'lightgreen'
            if item.get('virtual'):
                label += "<BR/><B>VIRTUAL</B>"
                fillcolor = 'plum'

            label += f"<BR/>Shape: {shape_str}<BR/>Type: {dtype_str}"

            storage = item.get('storage')
            if storage:
                storage_str = format_storage_info(storage)
                if storage_str:
                    label += "<BR/>---<BR/>Storage:<BR/>"
                    for line in storage_str.split('\n'):
                        label += f"<FONT POINT-SIZE='7'>{line}</FONT><BR/>"

            dimensions = item.get('dimensions')
            if dimensions:
                label += "<BR/>---<BR/>Dimensions:<BR/>"
                for dim in dimensions:
                    dim_label = html_escape(dim['label'])
                    label += f"<FONT POINT-SIZE='7'>&#91;{dim['index']}&#93;: {dim_label}</FONT><BR/>"

            if attr_str:
                label += "<BR/>---<BR/>Attributes:<BR/>"
                for line in attr_str.split('\n'):
                    label += f"<FONT POINT-SIZE='7'>{line}</FONT><BR/>"

            label += ">"
            dot.node(node_name, label, fillcolor=fillcolor, penwidth='1.2')

    # Create hierarchy edges (parent to child)
    for item in structure:
        if item['parent'] is not None:
            parent_name = item['parent'] if item['parent'] else '/'
            edge_attrs = {}
            link_info = item.get('link')
            if link_info and link_info.get('type') == 'HardLink':
                edge_attrs['penwidth'] = '2.0'
            dot.edge(parent_name, item['name'], **edge_attrs)

    # Create link target edges (for soft/external links pointing to targets)
    node_names = {item['name'] for item in structure}
    for item in structure:
        if item['type'] == 'SoftLink':
            link_info = item.get('link', {})
            target = link_info.get('target')
            if target:
                # Normalize target path (remove leading /)
                normalized_target = target.lstrip('/')
                if normalized_target == '':
                    normalized_target = '/'
                if normalized_target in node_names:
                    dot.edge(item['name'], normalized_target, style='dotted', color='orange',
                            constraint='false', arrowhead='vee')
        elif item['type'] == 'ExternalLink':
            # External links point outside the file, so we don't draw target edges
            pass

    dot.render(output_path, cleanup=True)

def main():
    parser = argparse.ArgumentParser(description='Analyze HDF5 file structure and generate PDF visualization')
    parser.add_argument('hdf5_file', help='Path to HDF5 file to analyze')
    parser.add_argument('-o', '--output', help='Output PDF path (without .pdf extension)',
                        default=None)
    parser.add_argument('--clip', action='store_true',
                        help='Copy output to clipboard')

    args = parser.parse_args()

    from .clip_utils import ClipboardCapture

    with ClipboardCapture(clip=args.clip):
        if not os.path.exists(args.hdf5_file):
            print(f"Error: File {args.hdf5_file} does not exist")
            return 1

        if args.output is None:
            base_name = os.path.splitext(os.path.basename(args.hdf5_file))[0]
            args.output = f"{base_name}_structure"

        print(f"Analyzing HDF5 file: {args.hdf5_file}")
        structure = analyze_hdf5_structure(args.hdf5_file)

        print(f"Found {len(structure)} objects in HDF5 file")
        print(f"\nStructure summary:")
        groups = sum(1 for item in structure if item['type'] == 'Group')
        datasets = sum(1 for item in structure if item['type'] == 'Dataset')
        softlinks = sum(1 for item in structure if item['type'] == 'SoftLink')
        externallinks = sum(1 for item in structure if item['type'] == 'ExternalLink')
        total_attrs = sum(len(item.get('attributes', {})) for item in structure)
        print(f"  Groups: {groups}")
        print(f"  Datasets: {datasets}")
        print(f"  Soft links: {softlinks}")
        print(f"  External links: {externallinks}")
        print(f"  Total attributes: {total_attrs}")

        links_count = {
            'hard': sum(1 for item in structure if item.get('link', {}).get('type') == 'HardLink')
        }
        if links_count['hard'] > 0:
            print(f"\nHard links found: {links_count['hard']}")

        storage_features = {
            'chunked': sum(1 for item in structure if 'chunks' in item.get('storage', {})),
            'compressed': sum(1 for item in structure if 'compression' in item.get('storage', {})),
            'virtual': sum(1 for item in structure if item.get('virtual')),
            'resizable': sum(1 for item in structure if 'maxshape' in item.get('storage', {}))
        }
        if any(storage_features.values()):
            print(f"\nStorage features:")
            if storage_features['chunked'] > 0:
                print(f"  Chunked datasets: {storage_features['chunked']}")
            if storage_features['compressed'] > 0:
                print(f"  Compressed datasets: {storage_features['compressed']}")
            if storage_features['virtual'] > 0:
                print(f"  Virtual datasets: {storage_features['virtual']}")
            if storage_features['resizable'] > 0:
                print(f"  Resizable datasets: {storage_features['resizable']}")

        dim_scales = sum(1 for item in structure if 'dimensions' in item)
        if dim_scales > 0:
            print(f"\nDimension scales: {dim_scales} datasets")

        if total_attrs > 0:
            print(f"\nAttributes found:")
            for item in structure:
                attrs = item.get('attributes', {})
                if attrs:
                    name = item['name'] if item['name'] else 'ROOT'
                    print(f"  {name} ({item['type']}): {', '.join(attrs.keys())}")

        print(f"\nGenerating visualization...")
        create_graph_visualization(structure, args.output)

        print(f"PDF generated: {args.output}.pdf")

    return 0

if __name__ == '__main__':
    exit(main())
