import math
import numpy as np
import os
from ase import Atoms
from ase.io import write

# ==========================================
# Step 0: Utility Functions
# ==========================================

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

# ==========================================
# Step 1: Define Atomic Coordinates
# ==========================================

# Planar graphene atomic coordinates (X, Y)
positions_xy = np.array([
    [9.987992273170107, 5.000000520654201], [7.850281233292895, 6.234208451584836],
    [5.712570193415680, 7.468416382515471], [12.125703177954843, 6.234208621672997],
    [9.987992138077630, 7.468416552603633], [7.850281098200416, 8.702624483534269],
    [5.712570058323203, 9.936832414464904], [14.263414082739580, 7.468416722691794],
    [12.125703042862366, 8.702624653622429], [9.987992002985152, 9.936832584553065],
    [7.850280963107938, 11.171040515483702], [5.712569923230724, 12.405248446414337],
    [16.401124987524316, 8.702624823710591], [14.263413947647102, 9.936832754641227],
    [12.125702907769888, 11.171040685571862], [9.987991867892674, 12.405248616502497],
    [7.850280828015460, 13.639456547433133], [16.401124852431838, 11.171040855660022],
    [14.263413812554624, 12.405248786590660], [12.125702772677410, 13.639456717521295],
    [9.987991732800197, 14.873664648451928], [11.413133254724119, 5.000000000000001],
    [9.275422214846905, 6.234207930930636], [7.137711174969692, 7.468415861861272],
    [5.000000135092477, 8.702623792791908], [13.550844159508856, 6.234208101018798],
    [11.413133119631642, 7.468416031949433], [9.275422079754426, 8.702623962880068],
    [7.137711039877213, 9.936831893810703], [5.000000000000000, 11.171039824741339],
    [15.688555064293590, 7.468416202037595], [13.550844024416376, 8.702624132968230],
    [11.413132984539162, 9.936832063898866], [9.275421944661948, 11.171039994829501],
    [7.137710904784735, 12.405247925760136], [15.688554929201114, 9.936832233987028],
    [13.550843889323898, 11.171040164917663], [11.413132849446685, 12.405248095848297],
    [9.275421809569471, 13.639456026778934], [15.688554794108633, 12.405248265936459],
    [13.550843754231421, 13.639456196867094], [11.413132714354207, 14.873664127797730]
])

# Initial height and setup
z0 = 6.799058
z_flat = np.full(len(positions_xy), z0)

# Calculate dimensions (keeping lx, ly for reference)
cx, cy, lx, ly = get_center_and_size(positions_xy)


# ==========================================
# Step 2: Deformation Logic
# ==========================================

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
    z = -0.5 * (k1 * x_rot ** 2 + k2 * y_rot ** 2)
    
    return z

# ==========================================
# Step 3: IO and Normalization
# ==========================================

def create_and_save_structure(name, deformation_func):
    """
    Applies the deformation function to the flat sheet and saves as an XYZ file.
    """
    x, y = positions_xy[:, 0], positions_xy[:, 1]
    
    z_deformation = deformation_func(x, y)
    z = z_flat + z_deformation
    
    atoms = Atoms('C' * len(z), positions=np.column_stack((x, y, z)))
    
    output_dir = "../data/fix_C_skeleton"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{name}.xyz")
    write(filepath, atoms)
    print(f'Saved: {name}')

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

# ==========================================
# Main Execution Block
# ==========================================

def main():
    """
    Main loop to generate unique structural parameters based on 
    non-redundancy rules.
    """
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
                
                create_and_save_structure(name, deformation_func)
                generated_count += 1
    
    print(f"\nTask Complete! Generated {generated_count} unique structures.")

if __name__ == "__main__":
    main()