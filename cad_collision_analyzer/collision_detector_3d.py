import logging
from typing import Optional

# --- PythonOCC Core Imports ---
try:
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    PYTHONOCC_INSTALLED = True
except ImportError:
    PYTHONOCC_INSTALLED = False
    # Define dummy types for type hinting if import fails
    TopoDS_Shape = type("TopoDS_Shape", (), {"IsNull": lambda: True})
    print("WARNING: pythonocc-core not found. 3D collision detection functionality will fail.")

# --- Logging Setup ---
# Get logger for the module
col_logger = logging.getLogger(__name__)
# Add a basic handler if none are configured (e.g., by a main application)
# This ensures warnings are visible if the module is used standalone.
if not col_logger.handlers:
    col_logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler() # Output warnings to console
    # Use format that might be inherited from main logger if available
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    col_logger.addHandler(ch)


def check_3d_collision(
    shape1: TopoDS_Shape,
    name1: str = "Shape1",
    shape2: TopoDS_Shape,
    name2: str = "Shape2",
    tolerance: float = 0.0001
) -> int:
    """
    Checks if the minimum distance between two TopoDS_Shape objects is
    less than or equal to a specified tolerance.

    Uses BRepExtrema_DistShapeShape to compute the minimum distance.

    Args:
        shape1: The first CAD geometry shape.
        name1: Optional name for the first shape (used in logging).
        shape2: The second CAD geometry shape.
        name2: Optional name for the second shape (used in logging).
        tolerance: The maximum distance (inclusive) to be considered as
                   contact or collision. Defaults to 0.0001.

    Returns:
        1: If the minimum distance <= tolerance (collision/contact detected).
        0: If the minimum distance > tolerance, or if the distance
           calculation fails.

    Raises:
        ImportError: If pythonocc-core is not installed.
        ValueError: If either input shape is Null.
    """
    if not PYTHONOCC_INSTALLED:
        raise ImportError("pythonocc-core is not installed or could not be imported.")

    if shape1.IsNull():
        # Raise error for invalid input, should be caught by caller
        raise ValueError(f"Input shape '{name1}' is null.")
    if shape2.IsNull():
        # Raise error for invalid input
        raise ValueError(f"Input shape '{name2}' is null.")

    if tolerance < 0:
        col_logger.warning(f"Tolerance {tolerance} is negative. Using 0.0 instead for distance check between '{name1}' and '{name2}'.")
        tolerance = 0.0

    try:
        # Initialize the distance calculation tool
        # BRepExtrema_DistShapeShape(shape1, shape2, faceLimit=..., edgeLimit=...) # Can add limits
        dist_calculator = BRepExtrema_DistShapeShape(shape1, shape2)
        # Perform the calculation
        dist_calculator.Perform()

        # Check if the calculation was successful
        if dist_calculator.IsDone():
            distance = dist_calculator.Value()
            # Compare distance with tolerance
            if distance <= tolerance:
                # Collision or contact within tolerance
                return 1
            else:
                # Shapes are separated by more than the tolerance
                return 0
        else:
            # Calculation failed
            col_logger.warning(f"Distance calculation failed between '{name1}' and '{name2}'. Assuming no collision.")
            return 0

    except Exception as e:
        # Catch any other unexpected errors during OCC call (e.g., C++ exceptions)
        col_logger.error(f"Unexpected error during distance calculation between '{name1}' and '{name2}'. Error: {e}", exc_info=True)
        # Return 0 assuming no collision detected if calculation crashes
        return 0


# Example Usage (Optional)
if __name__ == '__main__':
    if not PYTHONOCC_INSTALLED:
        print("Error: pythonocc-core is required to run this example.")
    else:
        import sys
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.gp import gp_Pnt
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


        print("Running collision check example...")

        # Create two boxes
        box1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Shape()
        box2_touching = BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), gp_Pnt(2, 1, 1)).Shape()
        box3_separated = BRepPrimAPI_MakeBox(gp_Pnt(1.1, 0, 0), gp_Pnt(2.1, 1, 1)).Shape()
        box4_overlap = BRepPrimAPI_MakeBox(gp_Pnt(0.5, 0.5, 0.5), gp_Pnt(1.5, 1.5, 1.5)).Shape()

        tol = 0.01

        print(f"\nChecking Box1 vs Box2 (Touching) with tolerance {tol}:")
        result_touch = check_3d_collision(box1, "Box1", box2_touching, "Box2_Touch", tol)
        print(f"Result: {result_touch} (Expected 1)")

        print(f"\nChecking Box1 vs Box3 (Separated > tol) with tolerance {tol}:")
        result_sep = check_3d_collision(box1, "Box1", box3_separated, "Box3_Sep", tol)
        print(f"Result: {result_sep} (Expected 0)")

        print(f"\nChecking Box1 vs Box3 (Separated = 0.1) with tolerance 0.15:")
        result_sep_in_tol = check_3d_collision(box1, "Box1", box3_separated, "Box3_Sep", 0.15)
        print(f"Result: {result_sep_in_tol} (Expected 1, distance is 0.1)")


        print(f"\nChecking Box1 vs Box4 (Overlapping) with tolerance {tol}:")
        result_overlap = check_3d_collision(box1, "Box1", box4_overlap, "Box4_Overlap", tol)
        print(f"Result: {result_overlap} (Expected 1)")

        # Example with null shape (should raise ValueError)
        try:
            print("\nChecking Box1 vs Null Shape:")
            null_s = TopoDS_Shape()
            check_3d_collision(box1, "Box1", null_s, "Null")
        except ValueError as ve:
            print(f"Caught expected error: {ve}")
        except Exception as e:
            print(f"Caught unexpected error: {e}")
