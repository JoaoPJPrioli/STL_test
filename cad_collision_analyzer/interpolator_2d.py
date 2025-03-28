import numpy as np
from typing import Sequence # Sequence can be used for array-like inputs

# No logging needed in this module as errors are mainly handled by return values


def is_point_in_polygon(point: np.ndarray, polygon_vertices: np.ndarray) -> bool:
    """
    Checks if a 2D point is strictly inside a polygon using the Ray Casting algorithm.

    Points lying exactly on the boundary (edges or vertices) are considered outside.

    Args:
        point: A NumPy array representing the (x, y) point, shape (2,).
        polygon_vertices: A NumPy array of shape (N, 2) with polygon vertices
                          ordered sequentially (CW or CCW), N >= 3.

    Returns:
        True if the point is strictly inside the polygon, False otherwise.
    """
    # --- Input Validation ---
    if not isinstance(point, np.ndarray) or point.shape != (2,):
        # print("Warning: Invalid point shape in is_point_in_polygon.") # Optional console warning
        return False
    if not isinstance(polygon_vertices, np.ndarray) or polygon_vertices.ndim != 2 or polygon_vertices.shape[1] != 2:
        # print("Warning: Invalid polygon_vertices shape in is_point_in_polygon.")
        return False

    num_vertices = polygon_vertices.shape[0]
    if num_vertices < 3:
        return False # A polygon requires at least 3 vertices

    x, y = point[0], point[1]
    inside = False

    # Iterate through edges (p1 -> p2)
    p1 = polygon_vertices[0]
    for i in range(num_vertices):
        p2 = polygon_vertices[(i + 1) % num_vertices]
        p1x, p1y = p1[0], p1[1]
        p2x, p2y = p2[0], p2[1]

        # Ensure p1y <= p2y for consistent edge checking direction
        if p1y > p2y:
             p1x, p2x = p2x, p1x
             p1y, p2y = p2y, p1y

        # Check if the horizontal ray crosses the edge's Y-span
        # The ray crosses if y is strictly between p1y (inclusive) and p2y (exclusive)
        # This correctly handles horizontal edges (p1y==p2y -> condition is false)
        # and avoids double-counting at vertices shared by edges.
        if p1y <= y < p2y:
             # Check if edge is vertical
             if np.isclose(p1x, p2x):
                  # If point is strictly left of vertical edge, it crosses
                  if x < p1x:
                       inside = not inside
             else: # Non-vertical edge
                  # Calculate x-intersection of the ray with the non-vertical edge line
                  # Division by zero is avoided because p1y != p2y is guaranteed by y-span check
                  x_intersection = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x

                  # If point is strictly left of the intersection point, it crosses
                  # Using np.isclose helps avoid floating point issues near the boundary
                  if x < x_intersection and not np.isclose(x, x_intersection):
                        inside = not inside

        # Move to the next edge's starting point
        p1 = p2

    return inside


def check_2d_interpolation(
    sampled_points_i: np.ndarray,
    polygon_vertices_j: np.ndarray
) -> int:
    """
    Checks if any sampled point from component i lies strictly inside the
    projected polygon boundary of component j.

    Args:
        sampled_points_i: NumPy array of shape (M, 2) containing sampled points
                          from component i's projection (includes original vertices).
        polygon_vertices_j: NumPy array of shape (N, 2) containing the ordered
                            vertices of component j's projected polygon boundary.

    Returns:
        1: If *any* point in `sampled_points_i` is strictly inside the
           polygon defined by `polygon_vertices_j`.
        0: Otherwise (no points inside, empty inputs, or invalid polygon).
    """
    # --- Input Validation ---
    if not isinstance(sampled_points_i, np.ndarray) or sampled_points_i.ndim != 2 or sampled_points_i.shape[1] != 2:
        # Logger could be added here if needed, but main script checks validity mostly
        # print("Warning: Invalid sampled_points_i input to check_2d_interpolation")
        return 0 # Invalid points array
    if not isinstance(polygon_vertices_j, np.ndarray) or polygon_vertices_j.ndim != 2 or polygon_vertices_j.shape[1] != 2:
        # print("Warning: Invalid polygon_vertices_j input to check_2d_interpolation")
        return 0 # Invalid polygon array

    if sampled_points_i.shape[0] == 0:
        return 0 # No points to check
    if polygon_vertices_j.shape[0] < 3:
        return 0 # Not a valid polygon to check against

    # --- Check each point ---
    # Vectorization might be possible but complex due to polygon structure. Loop is clear.
    for i in range(sampled_points_i.shape[0]):
        point = sampled_points_i[i]
        # Call the robust point-in-polygon check
        if is_point_in_polygon(point, polygon_vertices_j):
            return 1 # Found a point inside, no need to check further

    # --- No points found inside ---
    return 0


# Example Usage (Optional)
if __name__ == '__main__':
    print("Running interpolator_2d example...")

    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    points_in = np.array([[0.5, 0.5], [0.1, 0.1]])
    points_out = np.array([[1.5, 0.5], [-0.1, -0.1]])
    points_boundary = np.array([[0.5, 0.0], [1.0, 0.5], [1.0, 1.0]])
    mixed_points = np.vstack((points_out, points_in))

    print("\nChecking points inside square:")
    print(f"Point [0.5, 0.5]: {is_point_in_polygon(np.array([0.5, 0.5]), square)}") # Expected: True
    print(f"Point [0, 0]: {is_point_in_polygon(np.array([0, 0]), square)}")       # Expected: False (vertex)
    print(f"Point [0.5, 0]: {is_point_in_polygon(np.array([0.5, 0]), square)}")     # Expected: False (edge)
    print(f"Point [1.1, 0.5]: {is_point_in_polygon(np.array([1.1, 0.5]), square)}")   # Expected: False

    print("\nRunning interpolation checks:")
    print(f"Check (points_in vs square): {check_2d_interpolation(points_in, square)}") # Expected: 1
    print(f"Check (points_out vs square): {check_2d_interpolation(points_out, square)}")# Expected: 0
    print(f"Check (points_boundary vs square): {check_2d_interpolation(points_boundary, square)}")# Expected: 0
    print(f"Check (mixed_points vs square): {check_2d_interpolation(mixed_points, square)}") # Expected: 1
    print(f"Check (empty_points vs square): {check_2d_interpolation(np.empty((0,2)), square)}") # Expected: 0
    print(f"Check (points_in vs line): {check_2d_interpolation(points_in, np.array([[0,0],[1,1]]))}") # Expected: 0
