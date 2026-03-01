#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pca_orbital_analysis.py — PCA decomposition of orbital probability densities.

Reads a set of ``_r.vasp`` files (CHGCAR-format |ψ|² from
``vasp_analyzer.py``), crops a sub-volume around Fe, constructs an
"up + z-flipped" data matrix, and performs PCA to identify the dominant
shape variations.

Outputs
-------
- ``pca.csv``               — PCA projections for each structure
- ``intensity.csv``         — Integrated density per structure
- ``pca_variance_ratio.png``— Scree plot
- ``pca_projection_2d.png`` — 2-D projection onto PC1 vs PC2
- ``pca_results/``          — CHGCAR-format files for mean density
                              and leading principal components

Usage
-----
::

    cd directory_with_vasp_files/
    python pca_orbital_analysis.py            # process all *.vasp
    python pca_orbital_analysis.py -p '*.vasp' --box 4.0 4.0 6.0
"""

import argparse
import functools
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm

from io_utils import create_subspace_header, read_chgcar, write_chgcar_like


# ------------------------------------------------------------------
# Worker function (runs in sub-process)
# ------------------------------------------------------------------
def process_and_save_file(
    filename,
    *,
    k_start,
    k_end,
    j_start,
    j_end,
    i_start,
    i_end,
    output_check_dir,
    up_header,
    down_header,
):
    """
    Process a single ``.vasp`` file: crop, normalise, and save check files.

    Returns ``(up_vector, down_vector, intensity)`` or ``None`` on error.
    """
    try:
        charge_density, _, _, _ = read_chgcar(filename)
        sub_volume = charge_density[
            k_start : k_end + 1, j_start : j_end + 1, i_start : i_end + 1
        ]

        rho_up = sub_volume
        intensity = sub_volume.sum()
        rho_down = np.flip(rho_up, axis=0)

        up_flat = rho_up.flatten().reshape(1, -1)
        down_flat = rho_down.flatten().reshape(1, -1)

        up_normalized = normalize(up_flat, norm="l2")
        down_normalized = normalize(down_flat, norm="l2")

        # Save intermediate check files
        base_filename = os.path.basename(filename).replace(".vasp", "")
        up_lattice, up_atoms = up_header
        down_lattice, down_atoms = down_header

        write_chgcar_like(
            os.path.join(output_check_dir, f"{base_filename}_up_lobe.vasp"),
            rho_up,
            up_lattice,
            up_atoms,
        )
        write_chgcar_like(
            os.path.join(output_check_dir, f"{base_filename}_down_lobe_flipped.vasp"),
            rho_down,
            down_lattice,
            down_atoms,
        )

        return up_normalized.flatten(), down_normalized.flatten(), intensity

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PCA analysis of orbital probability densities."
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default="../data/dz2_dn/*.vasp",
        help="Glob pattern for input files (default: '../data/dz2_dn/*.vasp').",
    )
    parser.add_argument(
        "--box",
        nargs=3,
        type=float,
        default=[4.0, 4.0, 6.0],
        metavar=("BX", "BY", "BZ"),
        help="Sub-volume box size in Å (default: 4.0 4.0 6.0).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of PCA components (default: 10).",
    )
    args = parser.parse_args()

    all_files = sorted(glob(args.pattern))
    if not all_files:
        print(f"No files matching '{args.pattern}'.")
        return

    print(f"Found {len(all_files)} files for PCA analysis.")

    # --- Compute cropping indices from the first file ---
    ref_density, ref_lattice, ref_atoms, ref_grid_dims = read_chgcar(all_files[0])
    ngx, ngy, ngz = ref_grid_dims

    box_dims = {"x": args.box[0], "y": args.box[1], "z": args.box[2]}
    print(
        f"Sub-volume box: {box_dims['x']} x {box_dims['y']} x {box_dims['z']} Å"
    )

    center_indices = (np.array(ref_grid_dims) * 0.5).round().astype(int)
    center_i, center_j, center_k = center_indices

    lattice_lengths = np.diag(ref_lattice)
    resolutions = lattice_lengths / np.array(ref_grid_dims)

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

    print(f"Grid index ranges: i=[{i_start},{i_end}], j=[{j_start},{j_end}], k=[{k_start},{k_end}]")

    ref_sub = ref_density[k_start : k_end + 1, j_start : j_end + 1, i_start : i_end + 1]
    lobe_shape = ref_sub.shape

    ref_slice_indices = (i_start, i_end, j_start, j_end, k_start, k_end)
    REFERENCE_LATTICE, REFERENCE_ATOMS = create_subspace_header(
        ref_lattice, ref_atoms, ref_slice_indices, ref_grid_dims
    )

    # --- Parallel file processing ---
    output_check_dir = "sliced_lobes_check"
    os.makedirs(output_check_dir, exist_ok=True)

    header_info = (ref_lattice, ref_atoms)
    worker_func = functools.partial(
        process_and_save_file,
        k_start=k_start,
        k_end=k_end,
        j_start=j_start,
        j_end=j_end,
        i_start=i_start,
        i_end=i_end,
        output_check_dir=output_check_dir,
        up_header=header_info,
        down_header=header_info,
    )

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(worker_func, all_files),
                total=len(all_files),
                desc="Processing files",
            )
        )

    # --- Collect results ---
    data_matrix_combined = []
    intensity_list = []
    valid_count = 0
    for result in results:
        if result is not None:
            up_vector, down_vector, intensity = result
            data_matrix_combined.append(up_vector)
            data_matrix_combined.append(down_vector)
            intensity_list.append(intensity)
            valid_count += 1

    # Save intensity
    with open("intensity.csv", "w") as f:
        f.write("structure_id,intensity\n")
        for i, fpath in enumerate(all_files):
            if i < len(intensity_list):
                label = os.path.basename(fpath).replace(".vasp", "")
                f.write(f"{label},{intensity_list[i]}\n")

    if not data_matrix_combined:
        print("No data processed successfully — aborting PCA.")
        return

    # --- PCA ---
    X = np.array(data_matrix_combined)
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    n_components = min(args.n_components, X.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_centered)

    print("Explained variance ratios:", pca.explained_variance_ratio_)

    # --- Save PCA component volumes ---
    output_dir = "pca_results"
    os.makedirs(output_dir, exist_ok=True)

    mean_lobe_3d = X_mean.reshape(lobe_shape)
    write_chgcar_like(
        os.path.join(output_dir, "mean_lobe.vasp"),
        mean_lobe_3d,
        REFERENCE_LATTICE,
        REFERENCE_ATOMS,
    )

    for i in range(min(5, n_components)):
        pc_3d = pca.components_[i].reshape(lobe_shape)
        write_chgcar_like(
            os.path.join(output_dir, f"principal_component_{i + 1}.vasp"),
            pc_3d,
            REFERENCE_LATTICE,
            REFERENCE_ATOMS,
        )

    # --- Scree plot ---
    plt.figure(figsize=(8, 5))
    plt.bar(
        range(1, n_components + 1),
        pca.explained_variance_ratio_,
        alpha=0.8,
        align="center",
        label="Individual",
    )
    plt.step(
        range(1, n_components + 1),
        np.cumsum(pca.explained_variance_ratio_),
        where="mid",
        label="Cumulative",
    )
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal component index")
    plt.legend(loc="best")
    plt.title("PCA Scree Plot")
    plt.tight_layout()
    plt.savefig("pca_variance_ratio.png", dpi=150)

    # --- 2-D projection ---
    plt.figure(figsize=(8, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8)

    with open("pca.csv", "w") as f:
        f.write("structure_id,pca1_p,pca2_p,pca3_p,pca1_m,pca2_m,pca3_m\n")
        for i, fpath in enumerate(all_files):
            up_idx = 2 * i
            down_idx = 2 * i + 1
            label = os.path.basename(fpath).replace(".vasp", "")

            plt.text(
                X_pca[up_idx, 0], X_pca[up_idx, 1], label, fontsize=9, color="blue"
            )
            plt.text(
                X_pca[down_idx, 0],
                X_pca[down_idx, 1],
                f"-{label}",
                fontsize=9,
                color="red",
            )
            f.write(
                f"{label},"
                f"{X_pca[up_idx, 0]},{X_pca[up_idx, 1]},{X_pca[up_idx, 2]},"
                f"{X_pca[down_idx, 0]},{X_pca[down_idx, 1]},{X_pca[down_idx, 2]}\n"
            )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D Projection onto First 2 PCs")
    plt.grid(True)
    plt.axhline(0, color="grey", lw=0.5)
    plt.axvline(0, color="grey", lw=0.5)
    plt.savefig("pca_projection_2d.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
