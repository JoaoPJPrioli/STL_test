import pytest
import os
import logging
from pathlib import Path
import numpy as np
import shutil

# --- Check for dependencies ---
try:
    import trimesh
    TRIMESH_INSTALLED = True
except ImportError:
    TRIMESH_INSTALLED = False

try:
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    # Import BRep_Tool for mocking triangulation failure
    from OCC.Core.BRep import BRep_Tool
    PYTHONOCC_INSTALLED = True
except ImportError:
    PYTHONOCC_INSTALLED = False
    TopoDS_Shape = type("TopoDS_Shape", (), {"IsNull": lambda self: True}) # Dummy type
    BRep_Tool = None # Define dummy if needed for mocking check

# --- Module Under Test ---
try:
    # Assuming tests run from root or PYTHONPATH includes project src
    from cad_collision_analyzer.mesh_converter import convert_shape_to_mesh, MeshConversionError, MESH_LOG_FILE
    # Need reader to get shapes for some tests
    from cad_collision_analyzer.cad_reader import read_step_file
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from cad_collision_analyzer.mesh_converter import convert_shape_to_mesh, MeshConversionError, MESH_LOG_FILE
    from cad_collision_analyzer.cad_reader import read_step_file

# --- Constants ---
TEST_DATA_SUBDIR = "data"
# NOTE: Requires 'single_cube_named.step' from previous tests in tests/data/
VALID_CUBE_FILE = "single_cube_named.step"

# --- Fixtures ---

@pytest.fixture(scope="module")
def test_data_dir(request) -> Path:
    """Provides the path to the test data directory."""
    module_dir = Path(request.module.__file__).parent
    data_dir = module_dir / TEST_DATA_SUBDIR
    data_dir.mkdir(exist_ok=True) # Ensure it exists
    return data_dir

@pytest.fixture(scope="module")
def simple_shape_from_file(test_data_dir) -> TopoDS_Shape:
    """Reads a simple cube STEP file and returns the first shape."""
    if not PYTHONOCC_INSTALLED:
        pytest.skip("pythonocc-core not installed, cannot load shape from file.")

    file_path = test_data_dir / VALID_CUBE_FILE
    if not file_path.exists():
        pytest.skip(f"Test data file not found: {file_path}")

    components = read_step_file(str(file_path))
    if not components:
        pytest.fail(f"Failed to read any components from supposedly valid file: {file_path}")

    name, shape = components[0]
    if shape.IsNull():
         pytest.fail(f"Shape read from {file_path} is Null.")
    return shape

@pytest.fixture(scope="module")
def simple_generated_shape() -> TopoDS_Shape:
    """Generates a simple OCC box shape programmatically."""
    if not PYTHONOCC_INSTALLED:
        pytest.skip("pythonocc-core not installed, cannot generate shape.")
    # Create a simple 1x1x1 box
    box_shape = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    if box_shape.IsNull():
         pytest.fail("Failed to generate BRepPrimAPI_MakeBox shape.")
    return box_shape


@pytest.fixture(autouse=True)
def manage_mesh_log_file():
    """Ensure mesh log file is cleaned before tests that might write to it."""
    log_path = Path(MESH_LOG_FILE).resolve()
    if log_path.exists():
        try:
            log_path.unlink()
        except OSError:
            pass # Ignore if deletion fails
    yield # Run the test

# --- Test Cases ---

@pytest.mark.skipif(not PYTHONOCC_INSTALLED or not TRIMESH_INSTALLED, reason="Requires pythonocc-core and trimesh")
def test_successful_conversion(simple_generated_shape):
    """Tests successful conversion of a simple generated box."""
    shape = simple_generated_shape
    component_name = "GeneratedBox"
    mesh = convert_shape_to_mesh(shape, component_name=component_name, linear_deflection=0.1)

    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.vertices.shape[0] > 4 # Box should have more than 4 vertices after triangulation
    assert mesh.faces.shape[0] > 2    # Box should have more than 2 faces
    assert mesh.vertices.shape[1] == 3
    assert mesh.faces.shape[1] == 3
    assert np.issubdtype(mesh.vertices.dtype, np.floating)
    assert np.issubdtype(mesh.faces.dtype, np.integer)
    # A simple box should be watertight after meshing
    assert mesh.is_watertight, "Generated box mesh should be watertight"

@pytest.mark.skipif(not PYTHONOCC_INSTALLED or not TRIMESH_INSTALLED, reason="Requires pythonocc-core and trimesh")
def test_successful_conversion_from_file(simple_shape_from_file):
    """Tests successful conversion of a shape read from a STEP file."""
    shape = simple_shape_from_file
    component_name = "CubeFromFile"
    mesh = convert_shape_to_mesh(shape, component_name=component_name, linear_deflection=0.1)

    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.vertices.shape[0] > 4
    assert mesh.faces.shape[0] > 2
    # Check bounds roughly match a 1x1x1 cube if that's what the file contains
    # extent = mesh.bounding_box.extents
    # assert np.allclose(extent, [1.0, 1.0, 1.0], atol=0.1) # Adjust tolerance based on deflection

@pytest.mark.skipif(not PYTHONOCC_INSTALLED or not TRIMESH_INSTALLED, reason="Requires pythonocc-core and trimesh")
def test_deflection_parameters_effect(simple_generated_shape):
    """Tests that changing deflection parameters affects mesh density."""
    shape = simple_generated_shape
    component_name = "DeflectionTest"

    # Coarse mesh
    mesh_coarse = convert_shape_to_mesh(
        shape, component_name, linear_deflection=0.5, angular_deflection=0.8
    )
    num_verts_coarse = mesh_coarse.vertices.shape[0]
    num_faces_coarse = mesh_coarse.faces.shape[0]

    # Fine mesh
    mesh_fine = convert_shape_to_mesh(
        shape, component_name, linear_deflection=0.05, angular_deflection=0.1
    )
    num_verts_fine = mesh_fine.vertices.shape[0]
    num_faces_fine = mesh_fine.faces.shape[0]

    assert num_verts_fine > num_verts_coarse, "Finer mesh should have more vertices"
    assert num_faces_fine > num_faces_coarse, "Finer mesh should have more faces"


@pytest.mark.skipif(not PYTHONOCC_INSTALLED or not TRIMESH_INSTALLED, reason="Requires pythonocc-core and trimesh")
def test_meshing_failure_null_input_shape(caplog):
    """Tests that ValueError is raised for a null input shape."""
    null_shape = TopoDS_Shape() # Default constructor creates a null shape
    component_name = "NullShapeTest"
    log_path = Path(MESH_LOG_FILE).resolve()

    # Check ValueError is raised
    with pytest.raises(ValueError, match=r"Input TopoDS_Shape is null"):
        convert_shape_to_mesh(null_shape, component_name=component_name)

    # Check log file IS written to for null shape (logged as error before raise)
    assert log_path.exists(), f"{MESH_LOG_FILE} should exist at {log_path}"
    log_content = log_path.read_text()
    assert f"Component: {component_name}" in log_content
    assert "Input TopoDS_Shape is null" in log_content


@pytest.mark.skipif(not PYTHONOCC_INSTALLED or not TRIMESH_INSTALLED or BRep_Tool is None, reason="Requires pythonocc-core, trimesh and BRep_Tool for mocking")
def test_meshing_failure_logs_error_and_raises(mocker, simple_generated_shape, caplog):
    """Tests error logging and exception raising when triangulation fails."""
    shape = simple_generated_shape
    component_name = "TriangulationFailTest"
    log_path = Path(MESH_LOG_FILE).resolve()

    # Mock BRep_Tool.Triangulation to return None, simulating failure
    # Ensure the path matches where BRep_Tool is used in mesh_converter.py
    mocker.patch('cad_collision_analyzer.mesh_converter.BRep_Tool.Triangulation', return_value=None)

    # Capture logs at ERROR level from the mesh_logger
    logger_name = 'cad_collision_analyzer.mesh_converter' # Or __name__ if used in module
    # mesh_conv_logger = logging.getLogger(logger_name)
    # Use caplog which captures from all loggers by default
    with caplog.at_level(logging.ERROR, logger=logger_name):
         with pytest.raises(MeshConversionError, match=r"Failed to get valid triangulation"):
             convert_shape_to_mesh(shape, component_name=component_name)

    # Check logger output captured by caplog for the specific logger
    assert any(f"Component: {component_name}" in rec.message and rec.levelname == 'ERROR' for rec in caplog.records), "Component name not found in ERROR log"
    assert any("Failed to get valid triangulation" in rec.message for rec in caplog.records), "Triangulation error message not logged"

    # Check log file was created and contains the error
    assert log_path.exists(), f"{MESH_LOG_FILE} should exist at {log_path}"
    log_content = log_path.read_text()
    assert f"Component: {component_name}" in log_content
    assert "Failed to get valid triangulation" in log_content


@pytest.mark.skipif(not PYTHONOCC_INSTALLED or not TRIMESH_INSTALLED or BRep_Tool is None, reason="Requires pythonocc-core, trimesh and BRep_Tool for mocking")
def test_log_file_creation_on_mesh_conversion_error(mocker, simple_generated_shape):
    """Specifically verify log file creation on MeshConversionError."""
    shape = simple_generated_shape
    component_name = "LogFileCreationTest"
    log_path = Path(MESH_LOG_FILE).resolve()

    # Mock to force MeshConversionError
    mocker.patch('cad_collision_analyzer.mesh_converter.BRep_Tool.Triangulation', return_value=None)

    with pytest.raises(MeshConversionError):
         convert_shape_to_mesh(shape, component_name=component_name)

    assert log_path.exists(), f"{MESH_LOG_FILE} was not created after MeshConversionError at {log_path}."
    assert log_path.stat().st_size > 0, f"{MESH_LOG_FILE} was created but is empty."
