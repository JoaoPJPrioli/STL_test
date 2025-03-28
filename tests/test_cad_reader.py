import pytest
import os
import logging
from pathlib import Path
import shutil # For cleaning up log file

# --- Check if pythonocc is installed and import necessary types ---
try:
    from OCC.Core.TopoDS import TopoDS_Shape
    PYTHONOCC_INSTALLED = True
except ImportError:
    PYTHONOCC_INSTALLED = False
    TopoDS_Shape = type("TopoDS_Shape", (), {"IsNull": lambda: True}) # Dummy type

# --- Module Under Test ---
# Assuming tests are run from the root directory or PYTHONPATH is set
try:
    # Adjust path to LOG_FILE if it's defined differently or want absolute path
    from cad_collision_analyzer.cad_reader import read_step_file, LOG_FILE
except ImportError:
    # Adjust path if necessary based on how pytest is run
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from cad_collision_analyzer.cad_reader import read_step_file, LOG_FILE

# --- Constants ---
TEST_DATA_SUBDIR = "data"
# NOTE: You MUST create these STEP files manually in tests/data/
VALID_SINGLE_NAMED_FILE = "single_cube_named.step" # Expects one shape named "MyCube"
VALID_MULTI_UNNAMED_FILE = "two_blocks_unnamed.step" # Expects two shapes, names Component_Shape_0, Component_Shape_1
VALID_ASSEMBLY_NAMED_FILE = "assembly_named.step" # Expects >=2 shapes, e.g., "PartA", "PartB"
INVALID_TEXT_FILE = "not_a_step.txt"

# --- Fixtures ---

@pytest.fixture(scope="module")
def test_data_dir(request) -> Path:
    """Provides the path to the test data directory."""
    # Assumes 'tests/data' exists relative to the test file's location
    module_dir = Path(request.module.__file__).parent
    data_dir = module_dir / TEST_DATA_SUBDIR
    # Ensure directory exists for creating dummy file
    data_dir.mkdir(exist_ok=True)
    # Create dummy text file for one test
    (data_dir / INVALID_TEXT_FILE).write_text("This is not a STEP file.")
    return data_dir

@pytest.fixture(autouse=True)
def manage_log_file():
    """Ensure log file is cleaned before and after tests that might write to it."""
    # Use absolute path if LOG_FILE is relative
    log_path = Path(LOG_FILE).resolve()
    if log_path.exists():
        try:
            log_path.unlink()
        except OSError:
            pass # Ignore if deletion fails (e.g., permissions)
    yield # Run the test
    # Optional: Cleanup after test
    # if log_path.exists():
    #     try: log_path.unlink()
    #     except OSError: pass


# --- Test Cases ---

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="pythonocc-core not installed")
def test_read_valid_step_single_component(test_data_dir):
    """Tests loading a valid STEP file with one named component."""
    file_path = test_data_dir / VALID_SINGLE_NAMED_FILE
    if not file_path.exists():
        pytest.skip(f"Test data file not found: {file_path}")

    components = read_step_file(str(file_path))

    assert isinstance(components, list)
    assert len(components) == 1, f"Expected 1 component, found {len(components)}"
    name, shape = components[0]
    assert isinstance(name, str)
    # Exact name depends on the content of your file
    assert name == "MyCube", f"Expected name 'MyCube', found '{name}'"
    assert isinstance(shape, TopoDS_Shape)
    assert not shape.IsNull(), "Shape should not be null"

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="pythonocc-core not installed")
def test_read_valid_step_multiple_unnamed_components(test_data_dir):
    """Tests loading a valid STEP file with multiple components lacking names."""
    file_path = test_data_dir / VALID_MULTI_UNNAMED_FILE
    if not file_path.exists():
        pytest.skip(f"Test data file not found: {file_path}")

    components = read_step_file(str(file_path))

    assert isinstance(components, list)
    # Adjust expected count based on your file
    assert len(components) >= 2, f"Expected at least 2 components, found {len(components)}"

    expected_names = {f"Component_Shape_{i}" for i in range(len(components))}
    found_names = set()

    for i, (name, shape) in enumerate(components):
        assert isinstance(name, str)
        found_names.add(name)
        assert isinstance(shape, TopoDS_Shape)
        assert not shape.IsNull(), f"Shape at index {i} should not be null"

    assert found_names == expected_names, f"Expected names {expected_names}, found {found_names}"

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="pythonocc-core not installed")
def test_read_valid_assembly_with_names(test_data_dir):
    """Tests loading a valid assembly STEP file with named components."""
    file_path = test_data_dir / VALID_ASSEMBLY_NAMED_FILE
    if not file_path.exists():
        pytest.skip(f"Test data file not found: {file_path}")

    components = read_step_file(str(file_path))

    assert isinstance(components, list)
    # Adjust expected count based on your file
    assert len(components) >= 2, f"Expected at least 2 components, found {len(components)}"

    # Exact names depend on your file, check if expected names are present
    expected_names = {"PartA", "PartB"} # Example names
    found_names = set()

    for i, (name, shape) in enumerate(components):
        assert isinstance(name, str)
        found_names.add(name)
        assert isinstance(shape, TopoDS_Shape)
        assert not shape.IsNull(), f"Shape at index {i} should not be null"

    # Use issubset to allow for extra unnamed components if assembly structure is complex
    assert expected_names.issubset(found_names), \
        f"Expected names {expected_names} to be a subset of found names {found_names}"


def test_read_non_existent_file(tmp_path):
    """Tests behavior when the STEP file path does not exist."""
    non_existent_path = tmp_path / "i_do_not_exist.step"
    with pytest.raises(FileNotFoundError):
        read_step_file(str(non_existent_path))

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="pythonocc-core not installed")
def test_read_non_step_file_logs_error(test_data_dir, caplog):
    """Tests reading a non-STEP file, expecting an empty list and logged error."""
    file_path = test_data_dir / INVALID_TEXT_FILE
    log_path = Path(LOG_FILE).resolve()

    # Capture ERROR level messages logged via logging framework
    # caplog doesn't directly read file handlers, check file explicitly
    components = read_step_file(str(file_path))

    assert components == [], "Expected empty list for non-STEP file"

    # Check if the error was also written to the log file
    assert log_path.exists(), f"{LOG_FILE} should have been created at {log_path}"
    log_content = log_path.read_text()
    assert f"Failed to read STEP file: {file_path}" in log_content, f"Expected read error message in {LOG_FILE}"
    assert "Status Code:" in log_content # Check that status code info is logged

@pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="pythonocc-core not installed")
def test_log_file_creation(test_data_dir):
     """Specifically checks if the log file is created on error."""
     file_path = test_data_dir / INVALID_TEXT_FILE
     log_path = Path(LOG_FILE).resolve()

     # Ensure log doesn't exist before test (handled by fixture)
     read_step_file(str(file_path)) # This should log an error

     assert log_path.exists(), f"{LOG_FILE} was not created after a read error at {log_path}."
     assert log_path.stat().st_size > 0, f"{LOG_FILE} was created but is empty."

# Potential future test (requires specific file or mocking):
# @pytest.mark.skipif(not PYTHONOCC_INSTALLED, reason="pythonocc-core not installed")
# def test_read_step_with_null_shape_logs_warning(test_data_dir, caplog):
#     """Tests a STEP file containing a null shape, expecting a warning."""
#     # Requires a specific test file "null_shape.step" where one component
#     # results in a null shape after transfer/GetShape.
#     file_path = test_data_dir / "null_shape.step"
#     if not file_path.exists():
#         pytest.skip(f"Test data file not found: {file_path}")
#
#     # Capture WARNING level logs
#     # components = read_step_file(str(file_path))
#     # Check file content instead of caplog
#     log_path = Path(LOG_FILE).resolve()
#     assert log_path.exists()
#     log_content = log_path.read_text()
#
#     # Assertions depend on the file content (e.g., how many valid shapes remain)
#     # assert len(components) == 0 # Or N-1 if N components total, 1 null
#     assert "Skipping null or invalid shape" in log_content
