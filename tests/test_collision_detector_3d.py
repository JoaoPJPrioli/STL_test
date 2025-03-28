import pytest
import logging

# --- Check for dependencies ---
try:
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    # Import BRepExtrema to potentially mock it later
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    PYTHONOCC_INSTALLED = True
except ImportError:
    PYTHONOCC_INSTALLED = False
    TopoDS_Shape = type("TopoDS_Shape", (), {"IsNull": lambda: True}) # Dummy

# --- Module Under Test ---
try:
    # Assuming tests run from root or PYTHONPATH includes project src
    from cad_collision_analyzer.collision_detector_3d import check_3d_collision
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from cad_collision_analyzer.collision_detector_3d import check_3d_collision


# --- Helper Functions ---

@pytest.fixture(scope="module")
def box_factory():
    """Factory fixture to create OCC boxes."""
    if not PYTHONOCC_INSTALLED:
        pytest.skip("pythonocc-core not installed")
    def _make_box(p1: tuple, p2: tuple) -> TopoDS_Shape:
        try:
            shape = BRepPrimAPI_MakeBox(gp_Pnt(*p1), gp_Pnt(*p2)).Shape()
            if shape.IsNull(): pytest.fail(f"Failed to create box ({p1}, {p2}): Null shape returned")
            return shape
        except Exception as e:
             pytest.fail(f"Failed to create box ({p1}, {p2}): {e}")
    return _make_box

@pytest.fixture(scope="module")
def sphere_factory():
    """Factory fixture to create OCC spheres."""
    if not PYTHONOCC_INSTALLED:
        pytest.skip("pythonocc-core not installed")
    def _make_sphere(center: tuple, radius: float) -> TopoDS_Shape:
         try:
            shape = BRepPrimAPI_MakeSphere(gp_Pnt(*center), radius).Shape()
            if shape.IsNull(): pytest.fail(f"Failed to create sphere ({center}, {radius}): Null shape returned")
            return shape
         except Exception as e:
             pytest.fail(f"Failed to create sphere ({center}, {radius}): {e}")
    return _make_sphere

def translate_shape(shape: TopoDS_Shape, vector: tuple) -> TopoDS_Shape:
    """Translates a TopoDS_Shape."""
    if not PYTHONOCC_INSTALLED:
        pytest.skip("pythonocc-core not installed")
    gp_vec = gp_Vec(*vector)
    transformation = gp_Trsf()
    transformation.SetTranslation(gp_vec)
    transformer = BRepBuilderAPI_Transform(shape, transformation, True) # True makes a copy
    new_shape = transformer.Shape()
    if new_shape.IsNull(): pytest.fail(f"Failed to translate shape: Null shape returned")
    return new_shape


# --- Test Cases ---

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="Requires pythonocc-core")
def test_separated_shapes(box_factory):
    """Test shapes clearly separated by more than tolerance."""
    box1 = box_factory((0, 0, 0), (1, 1, 1))
    box2 = box_factory((5, 0, 0), (6, 1, 1)) # Separated by 4 units along X
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=0.1) == 0
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=3.99) == 0
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=4.0) == 1 # Exactly distance
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=4.01) == 1

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="Requires pythonocc-core")
def test_touching_shapes(box_factory):
    """Test shapes touching exactly at a face (distance 0)."""
    box1 = box_factory((0, 0, 0), (1, 1, 1))
    box2 = box_factory((1, 0, 0), (2, 1, 1)) # Touch at x=1 face
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=0.0001) == 1
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=0.0) == 1
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=-0.1) == 1 # Negative tolerance treated as 0

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="Requires pythonocc-core")
def test_overlapping_shapes(box_factory):
    """Test shapes that clearly overlap (distance 0)."""
    box1 = box_factory((0, 0, 0), (1, 1, 1))
    box2 = box_factory((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=0.0001) == 1
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=0.0) == 1

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="Requires pythonocc-core")
def test_collision_within_tolerance(box_factory):
    """Test shapes separated by distance <= tolerance."""
    box1 = box_factory((0, 0, 0), (1, 1, 1))
    # Create box2 starting at x = 1 + 0.00005
    box2 = translate_shape(box1, (1.00005, 0, 0)) # Translate box1
    # box2 = box_factory((1.00005, 0, 0), (2.00005, 1, 1)) # Alternative way
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=0.0001) == 1
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=0.00005) == 1

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="Requires pythonocc-core")
def test_no_collision_outside_tolerance(box_factory):
    """Test shapes separated by distance > tolerance."""
    box1 = box_factory((0, 0, 0), (1, 1, 1))
    # Create box2 starting at x = 1 + 0.0002
    box2 = translate_shape(box1, (1.0002, 0, 0)) # Translate box1
    # box2 = box_factory((1.0002, 0, 0), (2.0002, 1, 1)) # Alternative way
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=0.0001) == 0
    # Test just below the actual distance
    assert check_3d_collision(box1, "B1", box2, "B2", tolerance=0.0001999) == 0

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="Requires pythonocc-core")
def test_sphere_box_collision(box_factory, sphere_factory):
    """Test collision between a sphere and a box."""
    box = box_factory((0, 0, 0), (1, 1, 1))
    # Sphere centered at (1.5, 0.5, 0.5) with radius 0.6 -> overlaps box
    sphere = sphere_factory((1.5, 0.5, 0.5), 0.6)
    assert check_3d_collision(box, "Box", sphere, "Sphere", tolerance=0.0001) == 1

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="Requires pythonocc-core")
def test_sphere_box_no_collision(box_factory, sphere_factory):
    """Test no collision between a sphere and a box (outside tolerance)."""
    box = box_factory((0, 0, 0), (1, 1, 1))
    # Sphere centered at (1.5, 0.5, 0.5) with radius 0.4 -> distance is 0.1
    sphere = sphere_factory((1.5, 0.5, 0.5), 0.4)
    assert check_3d_collision(box, "Box", sphere, "Sphere", tolerance=0.05) == 0
    # Check collision when tolerance includes the distance
    assert check_3d_collision(box, "Box", sphere, "Sphere", tolerance=0.1) == 1
    assert check_3d_collision(box, "Box", sphere, "Sphere", tolerance=0.1001) == 1


@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="Requires pythonocc-core")
def test_null_input_shape_raises_error(box_factory):
    """Test that ValueError is raised for null input shapes."""
    box1 = box_factory((0, 0, 0), (1, 1, 1))
    null_shape = TopoDS_Shape() # Default constructor creates a null shape

    with pytest.raises(ValueError, match=r"Input shape 'Shape2' is null"):
        check_3d_collision(box1, "Box1", null_shape, "Shape2")

    with pytest.raises(ValueError, match=r"Input shape 'Shape1' is null"):
        check_3d_collision(null_shape, "Shape1", box1, "Box1")


@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="Requires pythonocc-core")
def test_calculation_failure_logs_warning_returns_zero(box_factory, mocker, caplog):
    """Test behavior when BRepExtrema calculation fails (IsDone returns False)."""
    box1 = box_factory((0, 0, 0), (1, 1, 1))
    box2 = box_factory((5, 0, 0), (6, 1, 1))

    # Mock the IsDone method of the BRepExtrema_DistShapeShape class
    # This affects all instances created after the patch starts
    mock_isdone = mocker.patch.object(BRepExtrema_DistShapeShape, 'IsDone', return_value=False)
    # Optionally mock Perform just to ensure it doesn't raise unrelated error
    mock_perform = mocker.patch.object(BRepExtrema_DistShapeShape, 'Perform', return_value=None)

    with caplog.at_level(logging.WARNING):
        result = check_3d_collision(box1, "TestBox1", box2, "TestBox2")

    # Assertions
    assert result == 0, "Expected 0 (no collision) when calculation fails"
    assert mock_perform.called, "Perform should have been called"
    assert mock_isdone.called, "IsDone should have been called"

    # Check log message
    assert len(caplog.records) >= 1, "Expected at least one log record"
    # Find the relevant warning record
    warning_record = None
    for record in caplog.records:
        if record.levelname == "WARNING" and "Distance calculation failed" in record.message:
             warning_record = record
             break
    assert warning_record is not None, "Warning message about calculation failure not found"
    assert "Distance calculation failed between 'TestBox1' and 'TestBox2'" in warning_record.message
    assert "Assuming no collision" in warning_record.message
