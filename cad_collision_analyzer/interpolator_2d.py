# cad_collision_analyzer/interpolator_2d.py

import numpy as np
import logging
from typing import List, Tuple

# Module-specific logger
logger = logging.getLogger("CADAnalyzer.interpolator")

def is_point_in_polygon(point: np.ndarray, polygon_vertices: np.ndarray) -> bool:
    """
    Determines if a 2D point is inside a 2D polygon using the Ray Casting algorithm.

    Handles cases where the point lies exactly on a polygon edge.

    Args:
        point: A NumPy array of shape (2,) representing the (x, y) coordinates of the point.
        polygon_vertices: A NumPy array of shape (N, 2) representing the vertices
                          of the polygon (assumed to be ordered, either CW or CCW).

    Returns:
        True if the point is strictly inside or on the boundary of the polygon, False otherwise.
    """
    if point is None or polygon_vertices is None or polygon_vertices.shape[0] < 3:
        return False # Invalid input

    x, y = point
    n = len(polygon_vertices)
    inside = False

    # Iterate through polygon edges (p1 -> p2)
    p1x, p1y = polygon_vertices[0]
    for i in range(n + 1): # Go one extra step to close the loop
        p2x, p2y = polygon_vertices[i % n]

        # Check if point is on the current edge (handle vertical/horizontal lines)
        # Check if x is between edge x's, and y is between edge y's
        # Add tolerance for floating point comparisons? Maybe not needed here.
        if min(p1x, p2x) <= x <= max(p1x, p2x) and min(p1y, p2y) <= y <= max(p1y, p2y):
             # Check if point lies on the line segment using cross-product or slope
             # Cross product: (p2y - p1y) * (x - p1x) - (p2x - p1x) * (y - p1y)
             cross_product = (p2y - p1y) * (x - p1x) - (p2x - p1x) * (y - p1y)
             if np.isclose(cross_product, 0.0, atol=1e-9): # Point is on the line containing the segment
                 logger.debug(f"Point {point} is on edge ({p1x},{p1y})->({p2x},{p2y})")
                 return True # Point is on the boundary

        # --- Ray Casting Logic ---
        # Check if the horizontal ray starting at the point intersects the edge
        if y > min(p1y, p2y):            # Point y is above the lower vertex y
            if y <= max(p1y, p2y):       # Point y is not above the upper vertex y
                if x <= max(p1x, p2x):   # Point x is not to the right of the edge's rightmost x
                    # Calculate intersection x-coordinate of the ray with the line containing the edge
                    if not np.isclose(p1y, p2y): # Avoid division by zero for horizontal edges
                        x_intersection = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        x_intersection = p1x # Should handle horizontal lines correctly now

                    # If the edge is vertical OR the point's x is to the left of intersection x
                    if np.isclose(p1x, p2x) or x <= x_intersection:
                        # Check for vertex intersection: if ray passes through a vertex,
                        # count it only if the vertex is the upper endpoint of the edge
                        # to avoid double counting when the ray passes through a vertex shared by two edges.
                        # This is implicitly handled by the <= check in most cases, but being explicit can help.
                        # (Standard algorithm often handles this via strict inequalities or specific vertex checks)
                        # For simplicity, the current logic usually works.

                        inside = not inside # Flip the 'inside' status

        # Move to the next edge
        p1x, p1y = p2x, p2y

    return inside


def check_2d_interpolation(
    points_to_check: np.ndarray,
    polygon_vertices: np.ndarray
) -> int:
    """
    Checks if any of the provided points fall inside (or on the boundary of) the given polygon.

    Args:
        points_to_check: Array (M, 2) of (x, y) points to check.
        polygon_vertices: Array (N, 2) of vertices defining the polygon boundary.

    Returns:
        1 if any point in points_to_check is inside or on the boundary of the polygon.
        0 otherwise.
       -1 if input is invalid (e.g., not enough polygon vertices).
    """
    if points_to_check is None or polygon_vertices is None:
        logger.warning("Invalid input (None) to check_2d_interpolation.")
        return -1
    if points_to_check.ndim != 2 or points_to_check.shape[1] != 2:
         logger.warning(f"Invalid shape for points_to_check: {points_to_check.shape}. Expected (M, 2).")
         return -1
    if polygon_vertices.ndim != 2 or polygon_vertices.shape[1] != 2:
        logger.warning(f"Invalid shape for polygon_vertices: {polygon_vertices.shape}. Expected (N, 2).")
        return -1
    if polygon_vertices.shape[0] < 3:
        logger.warning(f"Polygon has fewer than 3 vertices ({polygon_vertices.shape[0]}). Cannot perform check.")
        return -1 # Cannot form a polygon

    num_points = points_to_check.shape[0]
    if num_points == 0:
        logger.debug("No points provided to check interpolation. Returning 0.")
        return 0 # No points to check means no interpolation

    logger.debug(f"Checking if any of {num_points} points are inside polygon with {polygon_vertices.shape[0]} vertices.")

    try:
        for i, point in enumerate(points_to_check):
            if is_point_in_polygon(point, polygon_vertices):
                logger.debug(f"Point {i} ({point}) found inside polygon. Interpolation detected.")
                return 1 # Found an interpolating point

        # If loop completes without finding any points inside
        logger.debug("No points found inside the polygon.")
        return 0

    except Exception as e:
        logger.error(f"Error during 2D interpolation check: {e}", exc_info=True)
        return -1 # Indicate failure

