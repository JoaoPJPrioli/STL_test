import logging
import numpy as np
from typing import Optional

try:
    import trimesh
    TRIMESH_INSTALLED = True
except ImportError:
    TRIMESH_INSTALLED = False
    # Define dummy type for type hinting
    trimesh = type("trimesh", (), {"Trimesh": type("Trimesh", (), {})})
    print("WARNING: trimesh not found. Projector functionality will fail.")

# --- Logging Setup ---
proj_logger = logging.getLogger(__name__)
# Add a basic handler if none are configured
if not proj_logger.handlers:
    proj_logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler() # Output warnings to console
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    proj_logger.addHandler(ch)

# Default return value for centroid on failure
DEFAULT_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float64)

def calculate_geometric_center(mesh: 'trimesh.Trimesh') -> np.ndarray:
    """
    Calculates the geometric center (centroid) of a mesh.

    Args:
        mesh: The input trimesh.Trimesh object.

    Returns:
        A NumPy array representing the (x, y, z) coordinates of the centroid.
        Returns np.array([0.0, 0.0, 0.0]) and logs a warning if the
        calculation fails or the mesh is invalid/empty.
    """
    if not TRIMESH_INSTALLED:
        proj_logger.error("Trimesh library not installed, cannot calculate center.")
        return DEFAULT_CENTER.copy()

    if mesh is None or not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        proj_logger.warning("Cannot calculate center: Input mesh is None, invalid, or empty.")
        return DEFAULT_CENTER.copy()

    try:
        # Use mesh.center_mass assuming uniform density (geometric centroid)
        # If non-uniform density is needed, trimesh volume properties are more complex.
        center = mesh.center_mass
        # Check for NaN values which might indicate an issue (e.g., non-volume mesh)
        if np.isnan(center).any():
            proj_logger.warning(f"Centroid calculation resulted in NaN for mesh (is it watertight?). Returning default center {DEFAULT_CENTER}.")
            return DEFAULT_CENTER.copy()
        # Ensure it's a numpy array
        if not isinstance(center, np.ndarray):
             center = np.array(center, dtype=np.float64)

        return center

    except Exception as e:
        proj_logger.warning(f"Failed to calculate mesh centroid/center_mass. Error: {e}. Returning default center {DEFAULT_CENTER}.", exc_info=True)
        return DEFAULT_CENTER.copy()


def project_mesh_onto_plane(
    mesh: 'trimesh.Trimesh',
    direction: np.ndarray,
    plane_origin: np.ndarray
) -> np.ndarray:
    """
    Projects the vertices of a mesh onto a specified plane.

    The plane is defined by a point on the plane (plane_origin) and its
    normal vector (direction). The function calculates the 2D coordinates
    of the projected vertices relative to a basis defined on the plane.

    Args:
        mesh: The input trimesh.Trimesh object.
        direction: The normal vector of the projection plane (3D NumPy array).
                   Does not need to be normalized beforehand.
        plane_origin: A point lying on the projection plane (3D NumPy array).

    Returns:
        A NumPy array of shape (N, 2) containing the 2D coordinates of the
        projected vertices, where N is the number of vertices in the mesh.
        Returns an empty array np.empty((0, 2)) if the mesh is empty.

    Raises:
        RuntimeError: If the projection calculation fails unexpectedly.
        ValueError: If the direction vector is a zero vector or input shapes wrong.
        ImportError: If trimesh or numpy are not installed.
    """
    if not TRIMESH_INSTALLED:
         raise ImportError("Trimesh library not installed.")
    if not hasattr(np, 'ndarray'):
         raise ImportError("Numpy library not installed.")


    if mesh is None or not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        proj_logger.warning("Cannot project mesh: Input mesh is None, invalid, or empty.")
        return np.empty((0, 2), dtype=np.float64)

    try:
        # --- Validate and Normalize Direction Vector ---
        direction = np.asarray(direction, dtype=np.float64).flatten()
        if direction.shape != (3,):
            raise ValueError(f"Direction vector must have 3 elements, got shape {direction.shape}")

        norm = np.linalg.norm(direction)
        # Use a tolerance for zero check
        if np.isclose(norm, 0.0):
            raise ValueError("Direction vector cannot be a zero vector.")
        normal = direction / norm

        # --- Validate Plane Origin ---
        plane_origin = np.asarray(plane_origin, dtype=np.float64).flatten()
        if plane_origin.shape != (3,):
             raise ValueError(f"Plane origin vector must have 3 elements, got shape {plane_origin.shape}")


        # --- Define Plane Basis Vectors (U, V) ---
        # Choose axis A not parallel to the normal
        axis_x = np.array([1.0, 0.0, 0.0])
        # Check dot product magnitude against 1.0 within tolerance
        if np.isclose(np.abs(np.dot(normal, axis_x)), 1.0):
            # Normal is parallel to X-axis, choose Y-axis for A
            axis_a = np.array([0.0, 1.0, 0.0])
        else:
            # Normal is not parallel to X-axis, use X-axis for A
            axis_a = axis_x

        # U = normalize(cross(A, normal))
        u_vec = np.cross(axis_a, normal)
        u_norm = np.linalg.norm(u_vec)
        # Handle degenerate case where A and normal might be parallel (shouldn't happen with above check)
        if np.isclose(u_norm, 0.0):
             raise RuntimeError("Failed to create valid U basis vector (normal parallel to chosen A axis unexpectedly).")
        u_vec /= u_norm

        # V = cross(normal, U) - should already be normalized
        v_vec = np.cross(normal, u_vec)
        # Optional: Re-normalize V for numerical stability
        v_norm = np.linalg.norm(v_vec)
        if not np.isclose(v_norm, 1.0):
             # If v_norm is near zero, basis is degenerate
             if np.isclose(v_norm, 0.0):
                  raise RuntimeError("Failed to create valid V basis vector (U parallel to normal unexpectedly).")
             v_vec /= v_norm


        # --- Project Vertices (Vectorized) ---
        vertices = mesh.vertices # Shape (N, 3)

        # Vectors from plane origin to each vertex
        vecs_from_origin = vertices - plane_origin # Shape (N, 3)

        # Distances of vertices from the plane along the normal
        # distance = dot(vector_from_origin, normal)
        distances = np.dot(vecs_from_origin, normal) # Shape (N,)

        # Project points onto the 3D plane
        # P_proj = P - distance * normal
        # Need P relative to world origin, so use original vertices
        projections_3d = vertices - distances[:, np.newaxis] * normal # Shape (N, 3)

        # Calculate coordinates relative to the plane origin in 3D space
        # Coords = P_proj - plane_origin
        coords_on_plane_3d = projections_3d - plane_origin # Shape (N, 3)

        # Project onto the 2D basis vectors U and V using dot product
        # x_2d = dot(Coords, U), y_2d = dot(Coords, V)
        x_2d = np.dot(coords_on_plane_3d, u_vec) # Shape (N,)
        y_2d = np.dot(coords_on_plane_3d, v_vec) # Shape (N,)

        # Stack into the final (N, 2) array
        projected_coords = np.stack((x_2d, y_2d), axis=-1)

        return projected_coords

    except ValueError as ve:
        # Re-raise specific ValueErrors (like zero vector or shape mismatch)
        raise ve
    except Exception as e:
        # Treat other exceptions during calculation as runtime errors
        err_msg = f"Mesh projection failed unexpectedly. Error: {e}"
        # Log to the projector's logger
        proj_logger.error(err_msg, exc_info=True)
        # Raise RuntimeError to be caught by the main loop
        raise RuntimeError(err_msg) from e


# Example Usage (Optional)
if __name__ == '__main__':
    if not TRIMESH_INSTALLED:
        print("Error: trimesh is required to run this example.")
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        print("Running projector example...")
        # Create a simple mesh (e.g., a cube)
        try:
            # Define vertices for a cube centered roughly at origin
            vertices = np.array([
                [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
            ])
            # Define faces (triangles) for the cube
            faces = np.array([
                [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]
            ])
            cube_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False) # No processing needed here

            # Calculate center
            center = calculate_geometric_center(cube_mesh)
            print(f"\nCalculated Center: {center} (Expected close to [0, 0, 0])")

            # Project onto XY plane
            print("\nProjecting onto XY plane (normal=[0,0,1], origin=[0,0,0])...")
            origin_xy = np.array([0.0, 0.0, 0.0])
            normal_xy = np.array([0.0, 0.0, 1.0])
            projected_xy = project_mesh_onto_plane(cube_mesh, normal_xy, origin_xy)
            print("First 4 projected XY coordinates:")
            print(projected_xy[:4])
            print("Shape:", projected_xy.shape) # Expected (8, 2)
            # Expected output should roughly be the x,y coords of the vertices

            # Project onto YZ plane (shifted origin)
            print("\nProjecting onto YZ plane (normal=[1,0,0], origin=[0,1,1])...")
            origin_yz = np.array([0.0, 1.0, 1.0])
            normal_yz = np.array([1.0, 0.0, 0.0])
            projected_yz = project_mesh_onto_plane(cube_mesh, normal_yz, origin_yz)
            print("First 4 projected YZ coordinates (relative to shifted origin basis):")
            print(projected_yz[:4])
            print("Shape:", projected_yz.shape)

            # Project onto a tilted plane
            print("\nProjecting onto tilted plane (normal=[1,1,1], origin=[0,0,0])...")
            origin_tilted = np.array([0.0, 0.0, 0.0])
            normal_tilted = np.array([1.0, 1.0, 1.0]) # Will be normalized inside
            projected_tilted = project_mesh_onto_plane(cube_mesh, normal_tilted, origin_tilted)
            print("First 4 projected tilted coordinates:")
            print(projected_tilted[:4])
            print("Shape:", projected_tilted.shape)

            # Example: Empty mesh
            print("\nTesting empty mesh projection...")
            empty_m = trimesh.Trimesh()
            projected_empty = project_mesh_onto_plane(empty_m, normal_xy, origin_xy)
            print("Empty mesh projection result shape:", projected_empty.shape) # Expected (0, 2)

            # Example: Zero direction vector (should raise ValueError)
            try:
                print("\nTesting zero direction vector...")
                project_mesh_onto_plane(cube_mesh, np.array([0,0,0]), origin_xy)
            except ValueError as ve:
                print(f"Caught expected error: {ve}")

        except ImportError:
             print("Trimesh needed for example.")
        except Exception as e:
             print(f"An unexpected error occurred: {e}")
             proj_logger.error("Unexpected error in projector example", exc_info=True)
