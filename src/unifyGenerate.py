import math
import numpy as np
import os
from ase import Atoms
from ase.io import write
from ase.neighborlist import neighbor_list
from ase.geometry import geometry
from ase.constraints import FixAtoms
from scipy.spatial.distance import cdist

# Planar graphene atomic coordinates (X, Y)
positions_xy = np.array(
    [
        [9.987992273170107, 5.000000520654201],
        [7.850281233292895, 6.234208451584836],
        [5.712570193415680, 7.468416382515471],
        [12.125703177954843, 6.234208621672997],
        [9.987992138077630, 7.468416552603633],
        [7.850281098200416, 8.702624483534269],
        [5.712570058323203, 9.936832414464904],
        [14.263414082739580, 7.468416722691794],
        [12.125703042862366, 8.702624653622429],
        [9.987992002985152, 9.936832584553065],
        [7.850280963107938, 11.171040515483702],
        [5.712569923230724, 12.405248446414337],
        [16.401124987524316, 8.702624823710591],
        [14.263413947647102, 9.936832754641227],
        [12.125702907769888, 11.171040685571862],
        [9.987991867892674, 12.405248616502497],
        [7.850280828015460, 13.639456547433133],
        [16.401124852431838, 11.171040855660022],
        [14.263413812554624, 12.405248786590660],
        [12.125702772677410, 13.639456717521295],
        [9.987991732800197, 14.873664648451928],
        [11.413133254724119, 5.000000000000001],
        [9.275422214846905, 6.234207930930636],
        [7.137711174969692, 7.468415861861272],
        [5.000000135092477, 8.702623792791908],
        [13.550844159508856, 6.234208101018798],
        [11.413133119631642, 7.468416031949433],
        [9.275422079754426, 8.702623962880068],
        [7.137711039877213, 9.936831893810703],
        [5.000000000000000, 11.171039824741339],
        [15.688555064293590, 7.468416202037595],
        [13.550844024416376, 8.702624132968230],
        [11.413132984539162, 9.936832063898866],
        [9.275421944661948, 11.171039994829501],
        [7.137710904784735, 12.405247925760136],
        [15.688554929201114, 9.936832233987028],
        [13.550843889323898, 11.171040164917663],
        [11.413132849446685, 12.405248095848297],
        [9.275421809569471, 13.639456026778934],
        [15.688554794108633, 12.405248265936459],
        [13.550843754231421, 13.639456196867094],
        [11.413132714354207, 14.873664127797730],
    ]
)

# Initial height and setup
z0 = 6.799058
z_flat = np.full(len(positions_xy), z0)


def get_center_and_size(xy):
    """
    Calculates the center (cx, cy) and dimensions (lx, ly)
    of a set of 2D points.
    """
    x, y = xy[:, 0], xy[:, 1]
    cx = np.mean(x)
    cy = np.mean(y)
    lx = np.max(x) - np.min(x)
    ly = np.max(y) - np.min(y)
    return cx, cy, lx, ly


# Calculate dimensions (keeping lx, ly for reference)
cx, cy, lx, ly = get_center_and_size(positions_xy)


def generate_surface_k1_k2_theta(x, y, k1, k2, theta_deg, cx, cy):
    """
    Generates the surface height 'z' based on principal curvatures k1, k2
    and the principal direction angle theta.
    """
    theta_rad = np.deg2rad(theta_deg)
    dx = x - cx
    dy = y - cy

    # Rotation transformation
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    # Quadratic deformation formula
    # [Note] A negative sign is added here so that positive curvature
    # visually appears convex (protruding towards -Z).
    z = -0.5 * (k1 * x_rot**2 + k2 * y_rot**2)

    return z


def normalize(k1, k2, theta):
    """
    Normalizes (k1, k2, theta) to avoid generating equivalent duplicate structures.

    Rules:
      1. Ensure |k1| >= |k2| (sort by magnitude).
      2. If swapped, adjust signs and angle.
      3. If k1 == k2 (isotropic), theta is irrelevant (set to 0).
    """
    # Case: Isotropic curvature -> theta is redundant, keep only one.
    if math.isclose(k1, k2):
        return round(k1, 2), round(k2, 2), 0

    kk1, kk2, ttheta = k1, k2, theta

    # Enforce magnitude sorting
    if abs(kk1) < abs(kk2):
        kk1, kk2 = kk2, kk1
        # Flip signs and rotate 90 degrees to maintain geometry
        kk1 = -kk1
        kk2 = -kk2
        ttheta = 90 - ttheta

    nk1 = round(kk1, 2) + 0.0
    nk2 = round(kk2, 2) + 0.0
    # Standardize precision to avoid floating point mismatch
    return nk1, nk2, ttheta


def generate_fix_skeleton(deformation_func):
    """
    Applies the deformation function to the flat sheet and saves as an XYZ file.
    """
    x, y = positions_xy[:, 0], positions_xy[:, 1]

    z_deformation = deformation_func(x, y)
    z = z_flat + z_deformation

    atoms = Atoms("C" * len(z), positions=np.column_stack((x, y, z)))
    return atoms


def fix_skeleton_to_SAC(name, atoms):
    # 2. Calculate center of mass and identify the 6 nearest Carbon atoms
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    c_indices = [i for i, s in enumerate(symbols) if s == "C"]
    c_positions = positions[c_indices]

    center = atoms.get_center_of_mass()
    distances = cdist([center], c_positions)[0]
    nearest_6_indices = np.array(c_indices)[np.argsort(distances)[:6]]

    # 4. Replace the remaining 4 atoms with Nitrogen (N)
    replace_indices = nearest_6_indices[2:]
    for i in replace_indices:
        atoms[i].symbol = "N"

    nitrogen_indices = [
        i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "N"
    ]
    center = atoms[nitrogen_indices].get_center_of_mass()

    # 5. Place a single Fe atom at the center of mass
    atoms += Atoms("Fe", positions=[center])

    # 3. Remove the 2 nearest Carbon atoms
    delete_indices = nearest_6_indices[:2]
    del atoms[delete_indices]

    # 6. Detect edges and passivate with Hydrogen (simplified approach)
    # Identify edge atoms: Carbon atoms with fewer than 3 neighbors
    cutoff = 2.1  # C–C bond cutoff
    i_list, j_list = neighbor_list("ij", atoms, cutoff)
    neighbor_counts = np.bincount(i_list, minlength=len(atoms))

    # Passivate edge C atoms (only apply to Carbon)
    H_positions = []
    for i in range(len(atoms)):
        if atoms[i].symbol == "C" and neighbor_counts[i] < 3:
            pos = atoms[i].position
            bonded = atoms.positions[j_list[i_list == i]]
            if len(bonded) > 0:
                normal = pos - bonded.mean(axis=0)
                normal /= np.linalg.norm(normal)
                H_positions.append(pos + 1.1 * normal)  # C–H bond length ~1.1 Å

    atoms += Atoms("H" * len(H_positions), positions=H_positions)

    # Align structure and configure the unit cell
    positions = atoms.get_positions()
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    size = max_pos - min_pos

    # Add vacuum padding: 5 Å each side for xy, and extra padding for z (total +15 Å here)
    cell = [size[0] + 10, size[1] + 10, size[2] + 15]
    atoms.set_cell(cell)
    atoms.center()

    # Constrain (fix) all Carbon atoms
    mask = [atom.symbol == "C" for atom in atoms]
    atoms.set_constraint(FixAtoms(mask=mask))

    fe_indices = [i for i, atom in enumerate(atoms) if atom.symbol == "Fe"]
    if not fe_indices:
        raise RuntimeError("Fe atom not found!")
    fe_index = fe_indices[0]

    # Calculate current fractional coordinates of Fe (original cell)
    old_cell = atoms.get_cell()
    inv_old_cell = np.linalg.inv(old_cell.T)
    frac_coords = np.dot(inv_old_cell, atoms[fe_index].position)

    # Define the new target cell
    new_cell = np.array([[24.0, 0.0, 0.0], [0.0, 22.0, 0.0], [0.0, 0.0, 16.0]])
    # Do not rescale atoms; preserve Cartesian coordinates
    atoms.set_cell(new_cell, scale_atoms=False)

    # Calculate fractional coordinates of Fe in the new cell
    inv_new_cell = np.linalg.inv(new_cell.T)
    frac_coords_new = np.dot(inv_new_cell, atoms[fe_index].position)

    # Calculate the displacement vector to center Fe
    target_frac = np.array([0.5, 0.5, 0.5])
    delta_frac = target_frac - frac_coords_new
    delta_cart = np.dot(new_cell.T, delta_frac)

    # Translate the entire structure
    atoms.translate(delta_cart)

    # Save the structure
    output_file = f"../data/fix_SAC/{name}.vasp"

    write(output_file, atoms, format="vasp", direct=True, vasp5=True, sort=True)


def main():
    """
    Main loop to generate unique structural parameters based on
    non-redundancy rules.
    """

    output_dir = "../data/fix_SAC"
    os.makedirs(output_dir, exist_ok=True)
    k_range = [i * 0.02 for i in range(-6, 7)]
    generated = set()  # To track unique keys

    generated_count = 0
    for k1 in k_range:
        for k2 in k_range:
            # Rule 1: Enforce curvature sorting convention (k1 >= k2)
            if k1 < k2:
                continue

            # Rule 2: Enforce sign convention (k1 >= 0) to exclude simple mirrors
            if k1 < 0:
                continue

            # Rule 3: Optimization for isotropic cases (k1 == k2)
            # If k1 == k2, rotation does not change the shape, so only run theta=0.
            if math.isclose(k1, k2):
                thetas_to_run = [0]
            else:
                thetas_to_run = range(0, 91, 15)

            for theta in thetas_to_run:
                nk1, nk2, ntheta = normalize(k1, k2, theta)

                key = (nk1, nk2, ntheta)
                if key in generated:
                    continue  # Skip duplicates

                generated.add(key)
                name = f"{nk1:.2f}_{nk2:.2f}_{ntheta}"

                # Create lambda to capture current loop parameters
                deformation_func = lambda x, y, k1=nk1, k2=nk2, theta=ntheta: generate_surface_k1_k2_theta(
                    x, y, k1, k2, theta, cx, cy
                )

                atoms = generate_fix_skeleton(deformation_func)
                fix_skeleton_to_SAC(name, atoms)
                generated_count += 1

    print(f"Task Complete! Generated {generated_count} unique structures.")


if __name__ == "__main__":
    main()
