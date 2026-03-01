#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_z_centroid.py — Compute the ⟨z⟩ centroid of orbital probability density.

This script reads ``.vasp`` files (CHGCAR-format |ψ|² exported by
``vasp_analyzer.py``), crops an integration box centred on the cell
centre, and computes

    ⟨z⟩ = ∫ z · ρ(r) dV  /  ∫ ρ(r) dV

where z is measured from the geometric centre of the box (z = 0 at box
mid-plane).  Both the original density (``_p``) and the z-flipped
density (``_m``) are computed.

Usage
-----
Run in the directory containing ``.vasp`` files::

    python compute_z_centroid.py

Custom box and output::

    python compute_z_centroid.py -p '*.vasp' --box 5.0 5.0 8.0 -o results.csv
"""

import argparse
import functools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from io_utils import read_chgcar


# ------------------------------------------------------------------
# Core computation
# ------------------------------------------------------------------
def calculate_z_expectation(density_data, z_coordinates_grid, dV):
    """
    Compute ⟨z⟩ for a 3-D density block.

    Parameters
    ----------
    density_data : ndarray (Nz, Ny, Nx)
    z_coordinates_grid : ndarray (Nz, Ny, Nx)
        Physical z coordinate at each voxel, centred at box mid-plane.
    dV : float
        Volume element (Å³).

    Returns
    -------
    z_expectation : float  (Å)
    """
    numerator = np.sum(z_coordinates_grid * density_data) * dV
    denominator = np.sum(density_data) * dV

    if denominator == 0:
        return 0.0

    return numerator / denominator


def process_file(
    filename,
    *,
    k_start,
    k_end,
    j_start,
    j_end,
    i_start,
    i_end,
    z_coords_grid,
    dV,
):
    """
    Worker function: read one file, crop, compute ⟨z⟩_p and ⟨z⟩_m.

    Returns ``(expectation_p, expectation_m)`` or ``None`` on error.
    """
    try:
        charge_density, _, _, _ = read_chgcar(filename)
        sub_volume = charge_density[
            k_start : k_end + 1, j_start : j_end + 1, i_start : i_end + 1
        ]

        rho_up = sub_volume
        rho_down = np.flip(rho_up, axis=0)

        expectation_p = calculate_z_expectation(rho_up, z_coords_grid, dV)
        expectation_m = calculate_z_expectation(rho_down, z_coords_grid, dV)

        return (expectation_p, expectation_m)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute ⟨z⟩ centroid of orbital density (box-centre origin)."
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default="../data/dz2_dn/*.vasp",
        help="Glob pattern for input files (default: '../data/dz2_dn/*.vasp').",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="z_expectation.csv",
        help="Output CSV path (default: z_expectation.csv).",
    )
    parser.add_argument(
        "--box",
        nargs=3,
        type=float,
        default=[4.0, 4.0, 6.0],
        metavar=("BX", "BY", "BZ"),
        help="Integration box size in Å (default: 4.0 4.0 6.0).",
    )
    args = parser.parse_args()

    all_files = sorted(glob(args.pattern))
    if not all_files:
        print(f"No files matching '{args.pattern}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_files)} files.")

    # --- Compute cropping indices from the first file ---
    ref_density, ref_lattice, ref_atoms, ref_grid_dims = read_chgcar(all_files[0])
    ngx, ngy, ngz = ref_grid_dims

    box_dims = {"x": args.box[0], "y": args.box[1], "z": args.box[2]}
    print(
        f"Integration box: "
        f"{box_dims['x']} x {box_dims['y']} x {box_dims['z']} Å"
    )

    # Centre of the grid
    center_indices = (np.array(ref_grid_dims) * 0.5).round().astype(int)
    center_i, center_j, center_k = center_indices

    # Resolution (Å / grid point), assumes orthorhombic cell
    lattice_lengths = np.diag(ref_lattice)
    resolutions = lattice_lengths / np.array(ref_grid_dims)

    # Half-width in grid points
    delta_points = (
        np.array([box_dims["x"], box_dims["y"], box_dims["z"]]) / 2 / resolutions
    ).round().astype(int)
    delta_i, delta_j, delta_k = delta_points

    i_start = max(0, center_i - delta_i)
    i_end = min(ngx - 1, center_i + delta_i)
    j_start = max(0, center_j - delta_j)
    j_end = min(ngy - 1, center_j + delta_j)
    k_start = max(0, center_k - delta_k)
    k_end = min(ngz - 1, center_k + delta_k)

    print(
        f"Grid index ranges: "
        f"i=[{i_start},{i_end}], j=[{j_start},{j_end}], k=[{k_start},{k_end}]"
    )

    # --- Build z coordinate grid (box-centre = 0) ---
    lobe_shape = (
        k_end - k_start + 1,
        j_end - j_start + 1,
        i_end - i_start + 1,
    )
    dV = resolutions[0] * resolutions[1] * resolutions[2]

    z_physical_coords_vector = np.linspace(
        -box_dims["z"] / 2.0,
        box_dims["z"] / 2.0,
        lobe_shape[0],
    )
    Z_COORDS_GRID = np.zeros(lobe_shape)
    Z_COORDS_GRID += z_physical_coords_vector.reshape(-1, 1, 1)

    # --- Parallel processing ---
    worker_func = functools.partial(
        process_file,
        k_start=k_start,
        k_end=k_end,
        j_start=j_start,
        j_end=j_end,
        i_start=i_start,
        i_end=i_end,
        z_coords_grid=Z_COORDS_GRID,
        dV=dV,
    )

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(worker_func, all_files),
                total=len(all_files),
                desc="Computing ⟨z⟩",
            )
        )

    # --- Collect results ---
    records = []
    valid_count = 0
    for fpath, result in zip(all_files, results):
        if result is not None:
            exp_p, exp_m = result
            label = os.path.basename(fpath).replace("_r.vasp", "").replace(".vasp", "")
            records.append(
                {
                    "structure_id": label,
                    "expectation_p": exp_p,
                    "expectation_m": exp_m,
                }
            )
            valid_count += 1

    print(f"\nProcessed {valid_count} / {len(all_files)} files successfully.")

    if records:
        df = pd.DataFrame(records)
        df.to_csv(args.output, index=False, float_format="%.8f")
        print(f"Results saved to {args.output}")
    else:
        print("No files processed successfully — no CSV generated.")


if __name__ == "__main__":
    main()
