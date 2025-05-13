# cad_collision_analyzer/projector.py

import logging
import numpy as np
from typing import Optional

# Attempt import
try:
    import trimesh
except ImportError as e:
    logging.getLogger("CADAnalyzer.projector").critical(f"Failed to import trimesh: {e}. Is trimesh installed?")
    raise ImportError(f"trimesh import failed in projector: {e}") from e

# Module-specific logger
logger = logging.getLogger("CADAnalyzer.projector")

# Configure logging for projection errors (optional separate file)
projection_error_logger = logging.getLogger("ProjectionErrors")
if not projection_error_logger.handlers:
    projection_error_handler = logging.FileHandler('projection_errors.log', mode='a')
    projection_error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - Component: %(component)s - %(message)s')
    projection_error_handler.setFormatter(projection_error_formatter)
    projection_error_logger.addHandler(projection_error_handler)
    projection_error_logger.setLevel(logging.WARNING)
    projection_error_logger.propagate = False


def calculate_geometric_center(mesh: trimesh.Trimesh, component_name: str = "unknown") -> Optional[np.ndarray]:
    """
    Calculates the geometric center (centroid) of a trimesh.Trimesh object.

    Args:
        mesh: The mesh object.
        component_name: Name of the component (for logging).

    Returns:
        A NumPy array representing the (x, y, z) coordinates of the centroid,
        or None if the calculation fails or the mesh is invalid.
    """
    if mesh is None or not isinstance(mesh, trimesh.Trimesh):
        logger.warning(f"Invalid mesh object provided for centroid calculation of component '{component_name}'.")
        return None
    if len(mesh.vertices) == 0:
         logger.warning(f"Mesh for component '{component_name}' has no vertices. Cannot calculate centroid.")
         return None

    try:
        # Trimesh centroid calculation is usually robust
        centroid = mesh.centroid
        logger.debug(f"Calculated centroid for component '{component_name}': {centroid}")
        return centroid
    except Exception as e:
        msg = f"Failed to calculate centroid for component '{component_name}': {e}"
        projection_error_logger.error(msg, extra={'component': component_name}, exc_info=True)
        logger.error(f"Centroid calculation failed for '{component_name}': {e}", exc_info=False)
        return None


def project_mesh_onto_plane(
    mesh: trimesh.Trimesh,
    plane_normal: np.ndarray,
    plane_origin: np.ndarray,
    component_name: str = "unknown"
) -> Optional[np.ndarray]:
    """
    Projects the vertices of a trimesh.Trimesh object onto a plane defined by
    its normal vector and a point on the plane (origin).

    Returns the 2D coordinates of the projected vertices in a coordinate system
    defined on the plane.

    Args:
        mesh: The mesh to project.
        plane_normal: The normal vector of the projection plane (should be a unit vector).
        plane_origin: A point (x, y, z) lying on the projection plane.
        component_name: Name of the component (for logging).

    Returns:
        A NumPy array of shape (N, 2) containing the 2D coordinates of the
        projected vertices relative to the plane_origin in the plane's local
        coordinate system (U, V axes). Returns None if projection fails or
        the input is invalid.

    Raises:
        ValueError: If the plane_normal vector is zero or near-zero.
    """
    logger.debug(f"Projecting component '{component_name}' onto plane (normal={plane_normal}, origin={plane_origin})")

    if mesh is None or not isinstance(mesh, trimesh.Trimesh):
        logger.warning(f"Invalid mesh object provided for projection of component '{component_name}'.")
        return None
    if len(mesh.vertices) == 0:
        logger.warning(f"Mesh for component '{component_name}' is empty. Cannot project.")
        return None

    # --- Validate Normal Vector ---
    try:
        normal = np.array(plane_normal, dtype=np.float64)
        norm_magnitude = np.linalg.norm(normal)
        if np.isclose(norm_magnitude, 0.0, atol=1e-9):
            raise ValueError("Plane normal vector cannot be zero or near-zero.")
        # Ensure normal is a unit vector
        unit_normal = normal / norm_magnitude
    except Exception as e:
         msg = f"Invalid plane normal vector {plane_normal} for projection of '{component_name}': {e}"
         projection_error_logger.error(msg, extra={'component': component_name})
         logger.error(msg, exc_info=False)
         raise ValueError(msg) from e # Re-raise as ValueError

    origin = np.array(plane_origin, dtype=np.float64)

    # --- Perform Projection ---
    try:
        # 1. Translate vertices so the plane origin is at (0,0,0)
        vertices_relative = mesh.vertices - origin

        # 2. Calculate the signed distance of each vertex to the plane along the normal
        # dot product (vertices_relative . unit_normal) gives this distance
        distances = np.dot(vertices_relative, unit_normal)

        # 3. Subtract the projection of each vertex onto the normal vector
        # This effectively moves each point onto the plane (z'=0 in plane coords)
        # projected_vertices_3d = vertices_relative - np.outer(distances, unit_normal)
        # Note: np.outer creates a (N, 3) array where each row is distances[i] * unit_normal
        projected_vertices_on_plane_3d = vertices_relative - distances[:, np.newaxis] * unit_normal


        # --- Define a 2D Coordinate System (U, V) on the Plane ---
        # We need two orthogonal unit vectors (u, v) that lie on the plane
        # (i.e., are orthogonal to the unit_normal).

        # Create a candidate 'up' vector (e.g., Z-axis)
        # Avoid using a vector that is parallel to the normal
        if np.abs(np.dot(unit_normal, [0, 0, 1])) < 0.99:
            ref_vector = np.array([0.0, 0.0, 1.0])
        else: # If normal is close to Z-axis, use Y-axis instead
            ref_vector = np.array([0.0, 1.0, 0.0])

        # Create the U vector (first basis vector on the plane) using cross product
        # u = normal x ref_vector (or ref_vector x normal)
        u_vec = np.cross(unit_normal, ref_vector)
        u_vec /= np.linalg.norm(u_vec) # Normalize U

        # Create the V vector (second basis vector on the plane)
        # v = normal x u
        v_vec = np.cross(unit_normal, u_vec)
        # v should already be normalized if u and normal are unit vectors and orthogonal

        # --- Calculate 2D Coordinates ---
        # Project the 3D points (which are now on the plane, relative to origin)
        # onto the U and V basis vectors.
        x_2d = np.dot(projected_vertices_on_plane_3d, u_vec)
        y_2d = np.dot(projected_vertices_on_plane_3d, v_vec)

        # Combine into an (N, 2) array
        projected_vertices_2d = np.column_stack((x_2d, y_2d))

        logger.debug(f"Successfully projected {len(mesh.vertices)} vertices of '{component_name}' to 2D.")
        return projected_vertices_2d

    except Exception as e:
        msg = f"Projection calculation failed for component '{component_name}': {e}"
        projection_error_logger.error(msg, extra={'component': component_name}, exc_info=True)
        logger.error(f"Projection failed for '{component_name}': {e}", exc_info=False)
        # Return None to indicate failure in this case, rather than raising RuntimeError
        # The calling function (process_direction_worker) should handle None return.
        return None

