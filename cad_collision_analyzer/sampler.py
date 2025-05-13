# cad_collision_analyzer/sampler.py

import logging
import numpy as np
import random
from typing import Optional

# Attempt import (needed for type hint and potentially internal use)
try:
    from cad_collision_analyzer.interpolator_2d import is_point_in_polygon
except ImportError as e:
    # This might indicate a setup issue or the interpolator module itself failed to load
    logging.getLogger("CADAnalyzer.sampler").critical(f"Failed to import 'is_point_in_polygon' from interpolator_2d: {e}. Check module integrity.")
    raise ImportError(f"Import failed in sampler: {e}") from e


# Module-specific logger
logger = logging.getLogger("CADAnalyzer.sampler")

# Configure logging for sampling errors (optional separate file)
sampling_error_logger = logging.getLogger("SamplingErrors")
if not sampling_error_logger.handlers:
    sampling_error_handler = logging.FileHandler('sampling_errors.log', mode='a')
    sampling_error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - Polygon Points: %(num_poly_pts)s - %(message)s')
    sampling_error_handler.setFormatter(sampling_error_formatter)
    sampling_error_logger.addHandler(sampling_error_handler)
    sampling_error_logger.setLevel(logging.WARNING)
    sampling_error_logger.propagate = False


def sample_polygon_points(
    polygon_vertices: np.ndarray,
    num_additional_samples: int
) -> np.ndarray:
    """
    Samples points uniformly from within a 2D polygon using rejection sampling.

    Adds the specified number of sampled points to the original polygon vertices.

    Args:
        polygon_vertices: A NumPy array of shape (N, 2) representing the
                          vertices of the 2D polygon (assumed to be ordered).
        num_additional_samples: The target number of *additional* points to
                                sample uniformly from within the polygon's area.

    Returns:
        A NumPy array of shape (N + M, 2) containing the original vertices
        plus M successfully sampled points (where M <= num_additional_samples).
        Returns the original polygon_vertices if num_additional_samples <= 0,
        if the polygon is invalid, or if sampling fails completely.
    """
    if polygon_vertices is None or not isinstance(polygon_vertices, np.ndarray) or polygon_vertices.ndim != 2 or polygon_vertices.shape[1] != 2:
         logger.warning("Invalid polygon_vertices input (must be Nx2 numpy array). Returning original vertices.")
         # Should ideally return None or raise error, but returning input matches original behavior
         return polygon_vertices if polygon_vertices is not None else np.empty((0, 2))


    num_poly_pts = polygon_vertices.shape[0]
    logger.debug(f"Starting sampling for polygon with {num_poly_pts} vertices. Target additional samples: {num_additional_samples}")

    if num_additional_samples <= 0:
        logger.debug("Number of additional samples requested is <= 0. Returning original vertices.")
        return polygon_vertices

    if num_poly_pts < 3:
        logger.warning(f"Polygon has fewer than 3 vertices ({num_poly_pts}). Cannot sample points. Returning original vertices.")
        return polygon_vertices

    # --- Perform Rejection Sampling ---
    sampled_points_list = []
    try:
        # Calculate bounding box
        min_xy = np.min(polygon_vertices, axis=0)
        max_xy = np.max(polygon_vertices, axis=0)
        min_x, min_y = min_xy
        max_x, max_y = max_xy

        # Check for degenerate bounding box (polygon might be a line)
        if np.isclose(min_x, max_x) or np.isclose(min_y, max_y):
            logger.warning(f"Polygon bounding box is degenerate (likely a line or point). Cannot sample area. Returning original vertices.")
            return polygon_vertices

        num_generated = 0
        # Set a reasonable maximum number of attempts to avoid infinite loops
        # Increase multiplier if sampling efficiency is very low (complex shapes)
        max_iterations = num_additional_samples * 20 + 100 # Base + factor

        while len(sampled_points_list) < num_additional_samples and num_generated < max_iterations:
            num_generated += 1
            # Generate a random point within the bounding box
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            point = np.array([x, y])

            # Check if the point is inside the polygon
            if is_point_in_polygon(point, polygon_vertices):
                sampled_points_list.append(point)

        # Log if the target number wasn't reached
        num_successfully_sampled = len(sampled_points_list)
        if num_successfully_sampled < num_additional_samples:
            msg = (f"Rejection sampling finished after {num_generated}/{max_iterations} iterations. "
                   f"Generated {num_successfully_sampled}/{num_additional_samples} additional points.")
            sampling_error_logger.warning(msg, extra={'num_poly_pts': num_poly_pts})
            logger.warning(msg)

    except Exception as e:
        # Catch potential errors during bounding box or point-in-polygon check
        msg = f"Error during rejection sampling: {e}"
        sampling_error_logger.error(msg, extra={'num_poly_pts': num_poly_pts}, exc_info=True)
        logger.error(f"Sampling failed: {e}", exc_info=False)
        # Return only original vertices on error
        return polygon_vertices

    # --- Combine original vertices and sampled points ---
    if not sampled_points_list:
        logger.debug("No additional points were successfully sampled. Returning original vertices.")
        return polygon_vertices
    else:
        sampled_points_array = np.array(sampled_points_list)
        combined_points = np.vstack([polygon_vertices, sampled_points_array])
        logger.debug(f"Successfully added {num_successfully_sampled} sampled points. Returning {combined_points.shape[0]} total points.")
        return combined_points

