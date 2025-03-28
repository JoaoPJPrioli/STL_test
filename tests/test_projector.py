import pytest
import numpy as np
import logging

# --- Check for dependencies ---
try:
    import trimesh
    TRIMESH_INSTALLED = True
except ImportError:
    TRIMESH_INSTALLED = False

# --- Module Under Test ---
try:
    from cad_collision_analyzer.projector import calculate_geometric_center, project_mesh_onto_plane, DEFAULT_CENTER
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from cad_collision_analyzer.projector import calculate_geometric_center, project_mesh_onto_plane, DEFAULT_CENTER


# --- Fixtures ---

@pytest.fixture
def simple_cube_mesh() -> 'trimesh.Trimesh':
    """Creates a simple Trimesh cube centered at the origin."""
    if not TRIMESH_INSTALLED:
        pytest.skip("trimesh not installed")
    # Vertices of a cube with side length 1, centered at origin
    vertices = 0.5 * np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]
    ])
    # Use process=True to ensure center_mass is calculated correctly
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

@pytest.fixture
def offset_cube_mesh() -> 'trimesh.Trimesh':
    """Creates a simple Trimesh cube centered at (10, 20, 30)."""
    if not TRIMESH_INSTALLED:
        pytest.skip("trimesh not installed")
    center = np.array([10.0, 20.0, 30.0])
    vertices = 0.5 * np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
    ]) + center
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


@pytest.fixture
def empty_mesh() -> 'trimesh.Trimesh':
    """Creates an empty Trimesh object."""
    if not TRIMESH_INSTALLED:
        pytest.skip("trimesh not installed")
    return trimesh.Trimesh()


# --- Tests for calculate_geometric_center ---

@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_center_cube(simple_cube_mesh):
    """Test centroid/center_mass of a cube centered at origin."""
    center = calculate_geometric_center(simple_cube_mesh)
    assert np.allclose(center, [0.0, 0.0, 0.0], atol=1e-9)

@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_center_offset_cube(offset_cube_mesh):
    """Test centroid/center_mass of a cube offset from origin."""
    expected_center = np.array([10.0, 20.0, 30.0])
    center = calculate_geometric_center(offset_cube_mesh)
    assert np.allclose(center, expected_center, atol=1e-9)

@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_center_empty_mesh(empty_mesh, caplog):
    """Test centroid calculation on an empty mesh."""
    with caplog.at_level(logging.WARNING):
        center = calculate_geometric_center(empty_mesh)
    assert np.allclose(center, DEFAULT_CENTER)
    assert "mesh is None, invalid, or empty" in caplog.text

@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_center_invalid_input(caplog):
    """Test centroid calculation with None input."""
    with caplog.at_level(logging.WARNING):
        center = calculate_geometric_center(None) # type: ignore
    assert np.allclose(center, DEFAULT_CENTER)
    assert "mesh is None, invalid, or empty" in caplog.text

@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_center_calculation_failure(simple_cube_mesh, mocker, caplog):
    """Test fallback when mesh.center_mass access fails."""
    # Mock the center_mass property to raise an exception
    mocker.patch.object(trimesh.Trimesh, 'center_mass', property(lambda self: exec('raise ValueError("Mock center_mass error")')))

    with caplog.at_level(logging.WARNING):
        center = calculate_geometric_center(simple_cube_mesh)

    assert np.allclose(center, DEFAULT_CENTER)
    assert "Failed to calculate mesh centroid/center_mass" in caplog.text
    assert "Mock center_mass error" in caplog.text

# --- Tests for project_mesh_onto_plane ---

@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_project_xy_plane(simple_cube_mesh):
    """Project cube onto XY plane (normal Z)."""
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    projected = project_mesh_onto_plane(simple_cube_mesh, direction, origin)

    assert projected.shape == (8, 2)
    # For projection onto XY plane with origin 0,0,0 and normal 0,0,1
    # U vector is likely [1,0,0] or [-1,0,0]
    # V vector is likely [0,1,0] or [0,-1,0]
    # Projected 2D coords should match original XY coords (possibly flipped sign)
    expected_xy = simple_cube_mesh.vertices[:, :2] # Get original XY

    # Check if columns match, allowing for sign flips
    col0_match_x = np.allclose(projected[:, 0], expected_xy[:, 0]) or np.allclose(projected[:, 0], -expected_xy[:, 0])
    col1_match_y = np.allclose(projected[:, 1], expected_xy[:, 1]) or np.allclose(projected[:, 1], -expected_xy[:, 1])

    assert col0_match_x, "First projected column should match X or -X"
    assert col1_match_y, "Second projected column should match Y or -Y"


@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_project_yz_plane(simple_cube_mesh):
    """Project cube onto YZ plane (normal X)."""
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    projected = project_mesh_onto_plane(simple_cube_mesh, direction, origin)

    assert projected.shape == (8, 2)
    # Normal [1,0,0]. A can be [0,1,0]. U = cross(A, normal) = [0,0,1]. V = cross(normal, U) = [0,-1,0].
    # x_2d = dot(coords, U) = coords_z
    # y_2d = dot(coords, V) = -coords_y
    expected_z = simple_cube_mesh.vertices[:, 2]
    expected_neg_y = -simple_cube_mesh.vertices[:, 1]

    # Check if projected columns match expected results (or flipped versions)
    col0_match_z = np.allclose(projected[:, 0], expected_z) or np.allclose(projected[:, 0], -expected_z)
    col1_match_neg_y = np.allclose(projected[:, 1], expected_neg_y) or np.allclose(projected[:, 1], -expected_neg_y)

    assert col0_match_z, "First projected column should match Z or -Z"
    assert col1_match_neg_y, "Second projected column should match -Y or Y"


@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_project_offset_plane(simple_cube_mesh):
    """Project onto an offset XY plane."""
    origin = np.array([10.0, -5.0, 50.0]) # Offset origin
    direction = np.array([0.0, 0.0, 1.0]) # Still XY plane
    projected = project_mesh_onto_plane(simple_cube_mesh, direction, origin)

    # The resulting 2D coordinates should still represent the original XY coordinates
    # relative to the plane's U, V basis, independent of the plane's origin offset.
    assert projected.shape == (8, 2)
    expected_xy = simple_cube_mesh.vertices[:, :2]
    col0_match_x = np.allclose(projected[:, 0], expected_xy[:, 0]) or np.allclose(projected[:, 0], -expected_xy[:, 0])
    col1_match_y = np.allclose(projected[:, 1], expected_xy[:, 1]) or np.allclose(projected[:, 1], -expected_xy[:, 1])
    assert col0_match_x
    assert col1_match_y


@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_project_non_normalized_direction(simple_cube_mesh):
    """Test projection with a non-normalized direction vector."""
    origin = np.array([0.0, 0.0, 0.0])
    direction_norm = np.array([0.0, 0.0, 1.0])
    direction_non_norm = np.array([0.0, 0.0, 5.5]) # Non-normalized

    projected_norm = project_mesh_onto_plane(simple_cube_mesh, direction_norm, origin)
    projected_non_norm = project_mesh_onto_plane(simple_cube_mesh, direction_non_norm, origin)

    # Results should be identical as the function normalizes the direction
    assert np.allclose(projected_norm, projected_non_norm, atol=1e-9)


@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_project_empty_mesh(empty_mesh, caplog):
    """Test projection of an empty mesh."""
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    with caplog.at_level(logging.WARNING):
        projected = project_mesh_onto_plane(empty_mesh, direction, origin)

    assert isinstance(projected, np.ndarray)
    assert projected.shape == (0, 2)
    assert "mesh is None, invalid, or empty" in caplog.text


@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_project_zero_direction(simple_cube_mesh):
    """Test projection with a zero direction vector raises ValueError."""
    origin = np.array([0.0, 0.0, 0.0])
    zero_direction = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Direction vector cannot be a zero vector"):
        project_mesh_onto_plane(simple_cube_mesh, zero_direction, origin)


@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_project_invalid_direction_shape(simple_cube_mesh):
    """Test projection with incorrectly shaped direction vector."""
    origin = np.array([0.0, 0.0, 0.0])
    invalid_direction = np.array([1.0, 0.0]) # 2 elements instead of 3
    with pytest.raises(ValueError, match="Direction vector must have 3 elements"):
        project_mesh_onto_plane(simple_cube_mesh, invalid_direction, origin)

@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_project_invalid_origin_shape(simple_cube_mesh):
    """Test projection with incorrectly shaped origin vector."""
    direction = np.array([0.0, 0.0, 1.0])
    invalid_origin = np.array([0.0, 0.0, 0.0, 0.0]) # 4 elements instead of 3
    with pytest.raises(ValueError, match="Plane origin vector must have 3 elements"):
        project_mesh_onto_plane(simple_cube_mesh, direction, invalid_origin)


@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_project_failure_runtime_error(simple_cube_mesh, mocker, caplog):
    """Test that RuntimeError is raised on unexpected calculation failure."""
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])

    # Mock a numpy function used internally (e.g., linalg.norm called during basis calc)
    # to raise an error after the initial direction norm check passes.
    # Patching np.cross might be another option.
    mocker.patch('numpy.linalg.norm', side_effect=[1.0, ValueError("Mock numpy error")]) # First call works, second fails

    # Use the logger specifically associated with the projector module if needed
    logger_name = 'cad_collision_analyzer.projector'
    with caplog.at_level(logging.ERROR, logger=logger_name):
        with pytest.raises(RuntimeError, match="Mesh projection failed unexpectedly"):
            project_mesh_onto_plane(simple_cube_mesh, direction, origin)

    # Check that the error was logged by the projector logger
    assert any(rec.name == logger_name and rec.levelname == 'ERROR' for rec in caplog.records)
    assert "Mesh projection failed unexpectedly" in caplog.text
    assert "Mock numpy error" in caplog.text # Check original error is mentioned


@pytest.mark.skipif(not TRIMESH_INSTALLED, reason="trimesh not installed")
def test_project_tilted_plane_45deg(simple_cube_mesh):
    """ Project onto a plane tilted 45 degrees around Y axis """
    origin = np.array([0.0, 0.0, 0.0])
    # Normal vector for a plane tilted 45 deg around Y (points towards +X, +Z)
    direction = np.array([1.0, 0.0, 1.0]) # Will be normalized to [1/sqrt(2), 0, 1/sqrt(2)]
    projected = project_mesh_onto_plane(simple_cube_mesh, direction, origin)

    assert projected.shape == (8, 2)

    # Expected Basis (derived in thought process):
    # normal = [1/sqrt(2), 0, 1/sqrt(2)]
    # U = [1/sqrt(2), 0, -1/sqrt(2)] (or opposite sign)
    # V = [0, 1, 0] (or opposite sign)

    # Expected 2D coordinates:
    # x_2d = dot(vertex, U) = (vx - vz) / sqrt(2)
    # y_2d = dot(vertex, V) = vy
    sqrt2 = np.sqrt(2.0)
    expected_x_2d = (simple_cube_mesh.vertices[:, 0] - simple_cube_mesh.vertices[:, 2]) / sqrt2
    expected_y_2d = simple_cube_mesh.vertices[:, 1]

    # Check if projected columns match expected, allowing for potential sign flips or column swap
    col0_match_x = np.allclose(projected[:, 0], expected_x_2d) or np.allclose(projected[:, 0], -expected_x_2d)
    col1_match_y = np.allclose(projected[:, 1], expected_y_2d) or np.allclose(projected[:, 1], -expected_y_2d)
    col0_match_y = np.allclose(projected[:, 0], expected_y_2d) or np.allclose(projected[:, 0], -expected_y_2d)
    col1_match_x = np.allclose(projected[:, 1], expected_x_2d) or np.allclose(projected[:, 1], -expected_x_2d)

    assert (col0_match_x and col1_match_y) or \
           (col0_match_y and col1_match_x), \
           "Projected coordinates do not match expected values for 45-degree tilt"
