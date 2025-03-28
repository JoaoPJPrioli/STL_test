import logging
import numpy as np
import random # Using standard random for simplicity, np.random is also fine

# --- Import Helper Function ---
# Attempt relative import first, fallback to direct for script execution
try:
    from .interpolator_2d import is_point_in_polygon
except ImportError:
    try:
        from interpolator_2d import is_point_in_polygon
    except ImportError:
        # Define a dummy function only if import fails completely
        logging.critical("CRITICAL: Could not import is_point_in_polygon from interpolator_2d.")
        def is_point_in_polygon(point: np.ndarray, polygon_vertices: np.ndarray) -> bool:
            logging.error("Using dummy is_point_in_polygon! Results will be incorrect.")
            return False # Fail safe

# --- Logging Setup ---
sampler_logger = logging.getLogger(__name__)
# Add a basic handler if none are configured
if not sampler_logger.handlers:
    sampler_logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler() # Output warnings to console
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    sampler_logger.addHandler(ch)


# --- Sampling Function ---
def sample_polygon_points(polygon_vertices: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Samples additional points uniformly from within a 2D polygon.

    Uses rejection sampling within the polygon's bounding box. Relies on the
    is_point_in_polygon function from the interpolator_2d module.

    Args:
        polygon_vertices: NumPy array of shape (N, 2) representing the
                          ordered vertices of the 2D polygon.
        num_samples: The number of *additional* points to sample uniformly
                     from within the polygon.

    Returns:
        A NumPy array of shape (N + S, 2) containing the original vertices
        plus the successfully sampled points (S <= num_samples).
        Returns np.empty((0, 2)) if the input vertices are invalid.
    """
    # --- Input Validation ---
    if not isinstance(polygon_vertices, np.ndarray) or polygon_vertices.ndim != 2 or polygon_vertices.shape[1] != 2:
        sampler_logger.error("Invalid input: polygon_vertices must be a NumPy array of shape (N, 2).")
        return np.empty((0, 2), dtype=np.float64)

    num_vertices = polygon_vertices.shape[0]

    if num_vertices < 3:
        sampler_logger.warning(f"Cannot sample polygon with fewer than 3 vertices ({num_vertices} provided). Returning original vertices.")
        return polygon_vertices # Return original points as is

    if num_samples <= 0:
        return polygon_vertices # Return original points if no sampling needed

    # --- Rejection Sampling ---
    try:
        min_xy = np.min(polygon_vertices, axis=0)
        max_xy = np.max(polygon_vertices, axis=0)
        min_x, min_y = min_xy[0], min_xy[1]
        max_x, max_y = max_xy[0], max_xy[1]

        # Check for degenerate bounding box (collinear points)
        if np.isclose(min_x, max_x) or np.isclose(min_y, max_y):
             sampler_logger.warning("Polygon bounding box has zero area (points may be collinear). Cannot sample points. Returning original vertices.")
             return polygon_vertices

        sampled_points_list = []
        attempts = 0
        # Set a generous limit based on expected acceptance rate, but capped
        # Increase multiplier for complex polygons or high sample counts
        max_total_attempts = max(num_samples * 200, 2000)

        while len(sampled_points_list) < num_samples and attempts < max_total_attempts:
            # Generate random point within the bounding box
            rand_x = random.uniform(min_x, max_x)
            rand_y = random.uniform(min_y, max_y)
            test_point = np.array([rand_x, rand_y])

            # Check if the point is inside the polygon using the imported function
            if is_point_in_polygon(test_point, polygon_vertices):
                sampled_points_list.append(test_point)

            attempts += 1

        if len(sampled_points_list) < num_samples:
            sampler_logger.warning(f"Reached maximum attempts ({max_total_attempts}) but only generated {len(sampled_points_list)} / {num_samples} samples. Polygon might be complex or have small area relative to bounding box.")

        # --- Combine and Return ---
        if not sampled_points_list:
            # No points sampled, return original vertices
            # Log this specific case as a warning as well
            if num_samples > 0: # Only warn if samples were requested
                 sampler_logger.warning(f"No points were successfully sampled within max attempts for polygon with {num_vertices} vertices.")
            return polygon_vertices
        else:
            sampled_array = np.array(sampled_points_list, dtype=np.float64)
            combined_points = np.vstack((polygon_vertices, sampled_array))
            return combined_points

    except Exception as e:
        sampler_logger.error(f"An unexpected error occurred during sampling: {e}", exc_info=True)
        # Fallback to returning original vertices on unexpected error
        return polygon_vertices

# Example Usage (Optional)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("Running sampler example...")

    square = np.array([[0,0], [1,0], [1,1], [0,1]])
    num_to_sample = 10

    print(f"\nSampling {num_to_sample} points from a square:")
    sampled_square = sample_polygon_points(square, num_to_sample)
    print("Shape of result:", sampled_square.shape) # Expected (4 + num_sampled, 2)
    if sampled_square.shape[0] > 4:
        print("Original vertices:")
        print(sampled_square[:4])
        print("First few sampled points:")
        print(sampled_square[4:min(8, sampled_square.shape[0])])

    print("\nSampling from a line (should return original):")
    line = np.array([[0,0], [1,1]])
    sampled_line = sample_polygon_points(line, 5)
    print("Shape of result:", sampled_line.shape) # Expected (2, 2)
    print("Result:", sampled_line)

    print("\nSampling 0 points:")
    sampled_zero = sample_polygon_points(square, 0)
    print("Shape of result:", sampled_zero.shape) # Expected (4, 2)
    print("Result matches input:", np.allclose(sampled_zero, square))

    print("\nSampling from empty input:")
    sampled_empty = sample_polygon_points(np.empty((0,2)), 5)
    print("Shape of result:", sampled_empty.shape) # Expected (0, 2)
