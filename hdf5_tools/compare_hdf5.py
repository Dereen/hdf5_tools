#!/usr/bin/env python3
"""
Compare two HDF5 files: structure and dimensions.

Usage:
    python compare_hdf5.py file1.hdf5 file2.hdf5
    python compare_hdf5.py file1.hdf5 file2.hdf5 --full              # compare values
    python compare_hdf5.py file1.hdf5 file2.hdf5 --heatmaps          # save heatmap images
    python compare_hdf5.py file1.hdf5 file2.hdf5 --heatmaps ./diffs  # save to specific folder
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from .hdf5_common import extract_file_info, FileInfo

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def compute_diff_values(d1, d2, shape) -> np.ndarray:
    """Compute absolute difference values for 1D and 2D arrays only."""
    if len(shape) == 0:
        return None

    # Skip arrays with more than 2 dimensions
    if len(shape) > 2:
        return None

    # Skip non-numeric dtypes
    if not (np.issubdtype(d1.dtype, np.floating) or np.issubdtype(d1.dtype, np.integer)):
        return None

    if len(shape) == 1:
        arr1 = d1[:].astype(float)
        arr2 = d2[:].astype(float)
        diff = np.abs(arr1 - arr2)
        return diff.reshape(1, -1)

    # For 2D arrays, compute absolute diff per cell
    dim0, dim1 = shape[0], shape[1]
    diff_map = np.zeros((dim0, dim1), dtype=float)

    # Process in chunks along dim0
    chunk_size = 100
    for i in range(0, dim0, chunk_size):
        end_i = min(i + chunk_size, dim0)
        chunk1 = d1[i:end_i].astype(float)
        chunk2 = d2[i:end_i].astype(float)

        diff = np.abs(chunk1 - chunk2)
        diff_map[i:end_i] = diff

    return diff_map


def save_heatmap(ds_name: str, diff_map: np.ndarray, shape: tuple,
                 diff_count: int, total: int, output_dir: Path):
    """Save difference heatmap as PNG image."""
    # Calculate figure size based on data dimensions
    dim0, dim1 = diff_map.shape
    dpi = 100
    max_pixels = 4000

    width_px = min(max(dim1, 800), max_pixels)
    height_px = min(max(dim0, 400), max_pixels)

    fig_width = width_px / dpi
    fig_height = height_px / dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Use continuous colormap: viridis (dark=small diff, bright=large diff)
    # Set 0 (no diff) to black
    cmap = plt.cm.hot.copy()
    cmap.set_under('black')

    # Find max value for normalization, use small epsilon to avoid log(0)
    max_val = np.max(diff_map)
    if max_val == 0:
        max_val = 1

    im = ax.imshow(diff_map, aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=1e-10, vmax=max_val)

    pct = 100 * diff_count / total
    ax.set_title(f"{ds_name}\n{diff_count:,}/{total:,} values differ ({pct:.2f}%)\nmax diff: {max_val:.6g}")
    ax.set_xlabel(f"dim1 (0-{shape[1] if len(shape) > 1 else shape[0]})")
    ax.set_ylabel(f"dim0 (0-{shape[0]})")

    # Add colorbar with label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Absolute Difference')

    # Save
    safe_name = ds_name.replace('/', '_')
    output_path = output_dir / f"diff_{safe_name}.png"
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return output_path


def compare_values(path1: Path, path2: Path, datasets: list,
                   heatmaps_dir: Path = None) -> dict:
    """Compare actual values for datasets with matching shapes."""
    results = {}

    with h5py.File(path1, 'r') as f1, h5py.File(path2, 'r') as f2:
        pbar = tqdm(datasets, desc="Comparing")
        for ds_name, shape in pbar:
            pbar.set_postfix_str(ds_name[-40:])
            d1 = f1[ds_name]
            d2 = f2[ds_name]

            # Handle scalar datasets
            if shape == ():
                v1 = d1[()]
                v2 = d2[()]
                if isinstance(v1, bytes):
                    v1 = v1.decode('utf-8', errors='replace')
                if isinstance(v2, bytes):
                    v2 = v2.decode('utf-8', errors='replace')
                if v1 != v2:
                    results[ds_name] = {
                        'diff_count': 1,
                        'total': 1,
                        'first_diffs': [((), v1, v2)],
                        'diff_map': None,
                        'shape': shape,
                        'diff_sum': None
                    }
                else:
                    results[ds_name] = {
                        'diff_count': 0,
                        'total': 1,
                        'first_diffs': [],
                        'diff_map': None,
                        'shape': shape,
                        'diff_sum': None
                    }
                continue

            total_elements = int(np.prod(shape)) if shape else 1
            chunk_size = 100000

            # Compute diff map for heatmaps
            diff_map = None
            if heatmaps_dir and HAS_MATPLOTLIB and len(shape) >= 1:
                diff_map = compute_diff_values(d1, d2, shape)

            if total_elements <= chunk_size:
                arr1 = d1[:]
                arr2 = d2[:]
                if arr1.dtype.kind == 'f':
                    diff_mask = ~np.isclose(arr1, arr2, equal_nan=True)
                else:
                    diff_mask = arr1 != arr2
                diff_count = int(np.sum(diff_mask))
                first_diffs = []
                if diff_count > 0:
                    diff_indices = np.argwhere(diff_mask)[:5]
                    for idx in diff_indices:
                        idx_tuple = tuple(int(i) for i in idx)
                        first_diffs.append((idx_tuple, float(arr1[idx_tuple]), float(arr2[idx_tuple])))
                # Compute sum of differences for numeric arrays
                diff_sum = None
                if np.issubdtype(arr1.dtype, np.floating) or np.issubdtype(arr1.dtype, np.integer):
                    diff_sum = float(np.nansum(arr1.astype(float) - arr2.astype(float)))
                # Always store result for heatmaps
                results[ds_name] = {
                    'diff_count': diff_count,
                    'total': total_elements,
                    'first_diffs': first_diffs,
                    'diff_map': diff_map,
                    'shape': shape,
                    'diff_sum': diff_sum
                }
            else:
                # Chunked comparison for large datasets
                diff_count = 0
                first_diffs = []
                diff_sum = 0.0
                is_numeric = np.issubdtype(d1.dtype, np.floating) or np.issubdtype(d1.dtype, np.integer)

                stride = max(1, chunk_size // int(np.prod(d1.shape[1:])) if len(d1.shape) > 1 else chunk_size)

                for i in range(0, d1.shape[0], stride):
                    end_i = min(i + stride, d1.shape[0])
                    arr1 = d1[i:end_i]
                    arr2 = d2[i:end_i]

                    if arr1.dtype.kind == 'f':
                        diff_mask = ~np.isclose(arr1, arr2, equal_nan=True)
                    else:
                        diff_mask = arr1 != arr2

                    chunk_diffs = int(np.sum(diff_mask))
                    diff_count += chunk_diffs

                    # Accumulate sum of differences
                    if is_numeric:
                        diff_sum += float(np.nansum(arr1.astype(float) - arr2.astype(float)))

                    if chunk_diffs > 0 and len(first_diffs) < 5:
                        local_indices = np.argwhere(diff_mask)
                        for idx in local_indices[:5 - len(first_diffs)]:
                            idx_tuple = tuple(int(i) for i in idx)
                            global_idx = (i + idx[0],) + tuple(int(x) for x in idx[1:]) if len(idx) > 1 else (i + idx[0],)
                            first_diffs.append((global_idx, float(arr1[idx_tuple]), float(arr2[idx_tuple])))

                # Always store result for heatmaps
                results[ds_name] = {
                    'diff_count': diff_count,
                    'total': total_elements,
                    'first_diffs': first_diffs,
                    'diff_map': diff_map,
                    'shape': shape,
                    'diff_sum': diff_sum if is_numeric else None
                }

    return results


def compare_files(path1: Path, path2: Path, full: bool = False,
                  heatmaps_dir: Path = None):
    """Compare two HDF5 files."""
    paths = [path1, path2]
    infos = []
    for p in tqdm(paths, desc="Reading files"):
        infos.append(extract_file_info(p))
    info1, info2 = infos

    if info1.errors:
        print(f"Error reading {path1}: {info1.errors}")
        return
    if info2.errors:
        print(f"Error reading {path2}: {info2.errors}")
        return

    ds1 = set(info1.datasets.keys())
    ds2 = set(info2.datasets.keys())
    grp1 = set(info1.groups.keys())
    grp2 = set(info2.groups.keys())

    # Structure comparison
    print("=" * 70)
    print("STRUCTURE COMPARISON")
    print("=" * 70)

    # Groups
    only_in_1_grp = grp1 - grp2
    only_in_2_grp = grp2 - grp1
    common_grp = grp1 & grp2

    print(f"\nGroups in both: {len(common_grp)}")
    if only_in_1_grp:
        print(f"\nGroups only in {path1.name}:")
        for g in sorted(only_in_1_grp):
            print(f"  - {g}")
    if only_in_2_grp:
        print(f"\nGroups only in {path2.name}:")
        for g in sorted(only_in_2_grp):
            print(f"  + {g}")

    # Datasets
    only_in_1 = ds1 - ds2
    only_in_2 = ds2 - ds1
    common = ds1 & ds2

    print(f"\nDatasets in both: {len(common)}")
    if only_in_1:
        print(f"\nDatasets only in {path1.name}:")
        for ds in sorted(only_in_1):
            info = info1.datasets[ds]
            print(f"  - {ds}: {info.shape} {info.dtype}")
    if only_in_2:
        print(f"\nDatasets only in {path2.name}:")
        for ds in sorted(only_in_2):
            info = info2.datasets[ds]
            print(f"  + {ds}: {info.shape} {info.dtype}")

    # Dimension comparison for common datasets
    matching = []
    mismatched = []

    if common:
        print("\n" + "=" * 70)
        print("DIMENSION COMPARISON (common datasets)")
        print("=" * 70)

        for ds in sorted(common):
            shape1 = info1.datasets[ds].shape
            shape2 = info2.datasets[ds].shape

            if shape1 == shape2:
                matching.append((ds, shape1))
            else:
                mismatched.append((ds, shape1, shape2))

        if matching:
            print(f"\nMatching dimensions ({len(matching)}):")
            for ds, shape in matching:
                print(f"  {ds}: {shape}")

        if mismatched:
            print(f"\nMismatched dimensions ({len(mismatched)}):")
            for ds, shape1, shape2 in mismatched:
                print(f"\n  {ds}:")
                print(f"    {path1.name}: {shape1}")
                print(f"    {path2.name}: {shape2}")

                if len(shape1) == len(shape2):
                    for i, (d1, d2) in enumerate(zip(shape1, shape2)):
                        status = "OK" if d1 == d2 else "DIFF"
                        print(f"      dim[{i}]: {d1} vs {d2} [{status}]")
                else:
                    print(f"      ndim mismatch: {len(shape1)} vs {len(shape2)}")

    # Value comparison (--full or --heatmaps)
    value_diffs = {}
    if (full or heatmaps_dir) and matching:
        print("\n" + "=" * 70)
        print("VALUE COMPARISON")
        print("=" * 70)

        # Create heatmaps dir early if needed
        if heatmaps_dir and HAS_MATPLOTLIB:
            heatmaps_dir.mkdir(parents=True, exist_ok=True)

        value_diffs = compare_values(path1, path2, matching, heatmaps_dir=heatmaps_dir)

        # Separate into identical and different
        different = {k: v for k, v in value_diffs.items() if v['diff_count'] > 0}
        identical = {k: v for k, v in value_diffs.items() if v['diff_count'] == 0}

        heatmaps_saved = 0

        if different:
            print(f"\nDatasets with different values ({len(different)}):")
            for ds_name, diff_info in sorted(different.items()):
                pct = 100 * diff_info['diff_count'] / diff_info['total']
                diff_sum = diff_info.get('diff_sum')
                sum_str = f", sum(d1-d2)={diff_sum:.6g}" if diff_sum is not None else ""
                print(f"\n  {ds_name}: {diff_info['diff_count']}/{diff_info['total']} values differ ({pct:.2f}%){sum_str}")

                if full:
                    for idx, v1, v2 in diff_info['first_diffs']:
                        print(f"    {idx}: {v1} vs {v2}")

                # Save heatmap and report status
                if heatmaps_dir and HAS_MATPLOTLIB:
                    if diff_info.get('diff_map') is not None:
                        output_path = save_heatmap(
                            ds_name,
                            diff_info['diff_map'],
                            diff_info['shape'],
                            diff_info['diff_count'],
                            diff_info['total'],
                            heatmaps_dir
                        )
                        print(f"    -> heatmap: {output_path}")
                        heatmaps_saved += 1
                    else:
                        shape = diff_info.get('shape', ())
                        if len(shape) == 0:
                            print(f"    -> no heatmap: scalar value")
                        elif len(shape) > 2:
                            print(f"    -> no heatmap: {len(shape)}D array (only 1D/2D supported)")
                        else:
                            print(f"    -> no heatmap: non-numeric dtype")

        if identical:
            print(f"\nIdentical datasets ({len(identical)}):")
            for ds_name, diff_info in sorted(identical.items()):
                diff_sum = diff_info.get('diff_sum')
                sum_str = f" (sum(d1-d2)={diff_sum:.6g})" if diff_sum is not None else ""
                print(f"  {ds_name}{sum_str}")

                # Save heatmap (all zeros) and report status
                if heatmaps_dir and HAS_MATPLOTLIB:
                    if diff_info.get('diff_map') is not None:
                        output_path = save_heatmap(
                            ds_name,
                            diff_info['diff_map'],
                            diff_info['shape'],
                            diff_info['diff_count'],
                            diff_info['total'],
                            heatmaps_dir
                        )
                        print(f"    -> heatmap: {output_path}")
                        heatmaps_saved += 1
                    else:
                        shape = diff_info.get('shape', ())
                        if len(shape) == 0:
                            print(f"    -> no heatmap: scalar value")
                        elif len(shape) > 2:
                            print(f"    -> no heatmap: {len(shape)}D array (only 1D/2D supported)")
                        else:
                            print(f"    -> no heatmap: non-numeric dtype")

        if heatmaps_dir and HAS_MATPLOTLIB:
            print(f"\nSaved {heatmaps_saved} heatmaps to {heatmaps_dir}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Datasets only in {path1.name}: {len(only_in_1)}")
    print(f"  Datasets only in {path2.name}: {len(only_in_2)}")
    print(f"  Common datasets: {len(common)}")
    if common:
        print(f"    - matching shapes: {len(matching)}")
        print(f"    - mismatched shapes: {len(mismatched)}")
    if (full or heatmaps_dir) and matching:
        num_identical = len([v for v in value_diffs.values() if v['diff_count'] == 0])
        num_different = len([v for v in value_diffs.values() if v['diff_count'] > 0])
        print(f"    - identical values: {num_identical}")
        print(f"    - different values: {num_different}")


def main():
    parser = argparse.ArgumentParser(description='Compare two HDF5 files')
    parser.add_argument('file1', help='First HDF5 file')
    parser.add_argument('file2', help='Second HDF5 file')
    parser.add_argument('--full', action='store_true',
                        help='Compare values, show first differences')
    parser.add_argument('--heatmaps', nargs='?', const='.', metavar='DIR',
                        help='Save difference heatmaps as images (default: current dir)')
    parser.add_argument('--clip', action='store_true',
                        help='Copy output to clipboard')
    args = parser.parse_args()

    from .clip_utils import ClipboardCapture

    with ClipboardCapture(clip=args.clip):
        heatmaps_dir = None
        if args.heatmaps is not None:
            if not HAS_MATPLOTLIB:
                print("Error: --heatmaps requires matplotlib. Install with: pip install matplotlib")
                return 1
            heatmaps_dir = Path(args.heatmaps)

        path1 = Path(args.file1)
        path2 = Path(args.file2)

        for p in [path1, path2]:
            if not p.exists():
                print(f"Error: {p} does not exist")
                return 1

        print(f"Comparing:")
        print(f"  1: {path1}")
        print(f"  2: {path2}")
        if heatmaps_dir:
            print(f"  Heatmaps: {heatmaps_dir.absolute()}")
        if args.full:
            print("  Mode: full (comparing values)")

        compare_files(path1, path2, full=args.full, heatmaps_dir=heatmaps_dir)
    return 0


if __name__ == '__main__':
    exit(main())
