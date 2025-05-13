# cad_collision_analyzer/collision_detector_3d.py

import logging
from typing import Tuple

# Attempt import
try:
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape # type: ignore
    from OCC.Core.TopoDS import TopoDS_Shape # type: ignore
except ImportError as e:
    logging.getLogger("CADAnalyzer.collision_detector").critical(f"Failed to import pythonocc-core modules: {e}. Is pythonocc-core installed correctly?")
    raise ImportError(f"pythonocc-core import failed in collision_detector: {e}") from e

# Module-specific logger
logger = logging.getLogger("CADAnalyzer.collision_detector")

# Configure logging for collision detection errors to a separate file (optional)
collision_error_logger = logging.getLogger("CollisionDetectionErrors")
if not collision_error_logger.handlers:
    collision_error_handler = logging.FileHandler('cad_collision_errors.log', mode='a')
    collision_error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - Pair: %(pair)s - %(message)s')
    collision_error_handler.setFormatter(collision_error_formatter)
    collision_error_logger.addHandler(collision_error_handler)
    collision_error_logger.setLevel(logging.WARNING)
    collision_error_logger.propagate = False


def check_3d_collision(
    shape1: TopoDS_Shape,
    shape2: TopoDS_Shape,
    name1: str = "Shape1",
    name2: str = "Shape2",
    tolerance: float = 0.0001
) -> int:
    """
    Checks for 3D collision (contact) between two TopoDS_Shape objects based on minimum distance.

    Args:
        shape1: The geometry of the first component.
        shape2: The geometry of the second component.
        name1: Name of the first component (for logging).
        name2: Name of the second component (for logging).
        tolerance: The maximum distance (inclusive) to consider as contact.
                   Defaults to 0.0001. Must be non-negative.

    Returns:
        1 if the minimum distance between shape1 and shape2 is less than or equal to tolerance.
        0 if the minimum distance is greater than tolerance.
       -1 if the distance calculation fails.
    """
    pair_name = f"'{name1}' vs '{name2}'"
    logger.debug(f"Checking 3D collision for pair: {pair_name} with tolerance {tolerance}")

    if shape1.IsNull() or shape2.IsNull():
        logger.warning(f"One or both shapes are Null for pair {pair_name}. Cannot perform collision check.")
        return -1 # Indicate failure

    if tolerance < 0:
        logger.warning(f"Negative tolerance ({tolerance}) provided for pair {pair_name}. Using 0.0 instead.")
        tolerance = 0.0

    try:
        # Use BRepExtrema_DistShapeShape for accurate minimum distance calculation
        dist_tool = BRepExtrema_DistShapeShape(shape1, shape2)
        dist_tool.Perform() # Perform the calculation

        if not dist_tool.IsDone():
            # Log failure to the specific collision error log
            msg = f"BRepExtrema_DistShapeShape failed to compute distance."
            collision_error_logger.warning(msg, extra={'pair': pair_name})
            logger.warning(f"Distance calculation failed for pair {pair_name}.")
            return -1 # Indicate failure

        # Get the minimum distance value
        min_distance = dist_tool.Value()
        logger.debug(f"Minimum distance for pair {pair_name}: {min_distance}")

        # Check if the distance is within the tolerance
        if min_distance <= tolerance:
            logger.info(f"Contact detected for pair {pair_name} (Distance: {min_distance} <= Tolerance: {tolerance})")
            return 1 # Contact detected
        else:
            return 0 # No contact

    except Exception as e:
        # Log unexpected errors to the specific collision error log
        msg = f"Unexpected error during collision check: {e}"
        collision_error_logger.error(msg, extra={'pair': pair_name}, exc_info=True)
        logger.error(f"Unexpected error during collision check for pair {pair_name}: {e}", exc_info=False) # Don't flood main log with tracebacks
        return -1 # Indicate failure

