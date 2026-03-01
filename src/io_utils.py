#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
io_utils.py — Shared I/O utilities for CHGCAR / PARCHG format files.

Provides functions to read and write VASP volumetric data files, used by
both ``pca_orbital_analysis.py`` and ``compute_z_centroid.py``.
"""

import numpy as np


def read_chgcar(filename="CHGCAR"):
    """
    Read a VASP CHGCAR (or PARCHG / ``_r.vasp``) file.

    Parameters
    ----------
    filename : str
        Path to the CHGCAR-format file.

    Returns
    -------
    charge_density : ndarray, shape (ngz, ngy, ngx)
        Volumetric data normalised by cell volume.
    lattice : ndarray, shape (3, 3)
        Lattice vectors (Å), each row is a vector.
    atom_info : list[dict]
        List of ``{'symbol': str, 'coord': ndarray}`` with Cartesian coords.
    grid_dims : tuple (ngx, ngy, ngz)
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # --- header ---
    scale_factor = float(lines[1].strip())
    lattice = (
        np.array([list(map(float, line.split())) for line in lines[2:5]])
        * scale_factor
    )

    elements = lines[5].split()
    num_atoms = list(map(int, lines[6].split()))
    total_atoms = sum(num_atoms)

    current_line_idx = 7
    if "selective" in lines[current_line_idx].strip().lower():
        current_line_idx += 1

    coord_type = lines[current_line_idx].strip().lower()
    atom_coords_start_idx = current_line_idx + 1

    atom_coords_raw = [
        list(map(float, line.split()[:3]))
        for line in lines[atom_coords_start_idx : atom_coords_start_idx + total_atoms]
    ]

    atom_info = []
    atom_idx_counter = 0
    for elem, count in zip(elements, num_atoms):
        for _ in range(count):
            frac_coord = atom_coords_raw[atom_idx_counter]
            if coord_type == "direct":
                cart_coord = np.dot(frac_coord, lattice)
            else:
                cart_coord = np.array(frac_coord) * scale_factor
            atom_info.append({"symbol": elem, "coord": cart_coord})
            atom_idx_counter += 1

    # --- locate grid dimensions ---
    grid_dim_line_idx = atom_coords_start_idx + total_atoms
    while lines[grid_dim_line_idx].strip() != "":
        grid_dim_line_idx += 1
    grid_dim_line_idx += 1

    grid_dims = list(map(int, lines[grid_dim_line_idx].split()))
    ngx, ngy, ngz = grid_dims

    # --- volumetric data ---
    data_lines = lines[grid_dim_line_idx + 1 :]
    charge_data_flat = np.fromstring("".join(data_lines), sep=" ")
    charge_density_raw = charge_data_flat.reshape((ngz, ngy, ngx))

    volume = np.linalg.det(lattice)
    charge_density = charge_density_raw / volume

    return charge_density, lattice, atom_info, (ngx, ngy, ngz)


def create_subspace_header(
    original_lattice, original_atoms, slice_indices, grid_dims, center_atom_symbol="Fe"
):
    """
    Build a virtual cell header for a cropped sub-volume.

    The Fe (or other centre) atom is always included and clamped inside the
    sub-box so that VESTA can display it correctly.

    Parameters
    ----------
    original_lattice : ndarray (3, 3)
    original_atoms : list[dict]
    slice_indices : tuple
        ``(i_start, i_end, j_start, j_end, k_start, k_end)``
    grid_dims : tuple (ngx, ngy, ngz)
    center_atom_symbol : str

    Returns
    -------
    new_lattice : ndarray (3, 3)
    new_atoms : list[dict]
    """
    i_start, i_end, j_start, j_end, k_start, k_end = slice_indices
    ngx, ngy, ngz = grid_dims

    sub_dims_A = np.array(
        [
            np.linalg.norm(original_lattice[0]) / ngx * (i_end - i_start + 1),
            np.linalg.norm(original_lattice[1]) / ngy * (j_end - j_start + 1),
            np.linalg.norm(original_lattice[2]) / ngz * (k_end - k_start + 1),
        ]
    )
    new_lattice = np.diag(sub_dims_A)

    origin_frac = np.array([i_start / ngx, j_start / ngy, k_start / ngz])
    origin_cart = np.dot(origin_frac, original_lattice)

    new_atoms = []
    found_center_atom = False
    for atom in original_atoms:
        atom_symbol = atom["symbol"]
        atom_cart = atom["coord"]

        if atom_symbol == center_atom_symbol and not found_center_atom:
            center_atom_relative_coord = atom_cart - origin_cart
            center_atom_relative_coord = np.clip(
                center_atom_relative_coord, 0, sub_dims_A
            )
            new_atoms.append(
                {"symbol": atom_symbol, "coord": center_atom_relative_coord}
            )
            found_center_atom = True
            continue

        if (
            origin_cart[0] <= atom_cart[0] < origin_cart[0] + new_lattice[0, 0]
            and origin_cart[1] <= atom_cart[1] < origin_cart[1] + new_lattice[1, 1]
            and origin_cart[2] <= atom_cart[2] < origin_cart[2] + new_lattice[2, 2]
        ):
            new_atom_coord = atom_cart - origin_cart
            new_atoms.append({"symbol": atom_symbol, "coord": new_atom_coord})

    if not found_center_atom:
        print(
            f"Warning: centre atom '{center_atom_symbol}' not found for "
            f"sub-space {slice_indices}."
        )

    return new_lattice, new_atoms


def write_chgcar_like(filename, data_3d, lattice, atom_info):
    """
    Write a 3-D array to a CHGCAR-format file (viewable in VESTA).

    Parameters
    ----------
    filename : str
    data_3d : ndarray, shape (ngz, ngy, ngx)
    lattice : ndarray (3, 3)
    atom_info : list[dict]
    """
    header_lines = []
    header_lines.append("PCA component\n")
    header_lines.append("   1.0\n")
    for vec in lattice:
        header_lines.append(f" {vec[0]:11.6f} {vec[1]:11.6f} {vec[2]:11.6f}\n")

    elements = sorted(set(atom["symbol"] for atom in atom_info))
    counts = [
        len([atom for atom in atom_info if atom["symbol"] == e]) for e in elements
    ]
    header_lines.append(" ".join(elements) + "\n")
    header_lines.append(" ".join(map(str, counts)) + "\n")
    header_lines.append("Direct\n")

    for atom in atom_info:
        frac_coord = np.linalg.solve(lattice.T, atom["coord"].T)
        header_lines.append(
            f" {frac_coord[0]:9.6f} {frac_coord[1]:9.6f} {frac_coord[2]:9.6f}\n"
        )

    header_lines.append("\n")

    ngz, ngy, ngx = data_3d.shape
    header_lines.append(f" {ngx} {ngy} {ngz}\n")

    flat_data = data_3d.flatten()
    data_lines = []
    for i in range(0, len(flat_data), 5):
        line = " ".join(f"{x: .10E}" for x in flat_data[i : i + 5])
        data_lines.append(line + "\n")

    with open(filename, "w") as f:
        f.writelines(header_lines)
        f.writelines(data_lines)
