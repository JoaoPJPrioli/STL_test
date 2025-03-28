import pytest
import numpy as np

# --- Module Under Test ---
try:
    from cad_collision_analyzer.interpolator_2d import is_point_in_polygon, check_2d_interpolation
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from cad_collision_analyzer.interpolator_2d import is_point_in_polygon, check_2d_interpolation


# --- Fixtures ---

@pytest.fixture
def square_vertices() -> np.ndarray:
    """Unit square vertices."""
    return np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])

@pytest.fixture
def triangle_vertices() -> np.ndarray:
    """Simple triangle vertices."""
    return np.array([[0., 0.], [2., 0.], [1., 1.5]])

@pytest.fixture
def concave_vertices() -> np.ndarray:
    """A simple concave shape (like a 'U')."""
    return np.array([[0.,0.], [3.,0.], [3.,1.], [2.,1.], [2.,2.], [1.,2.], [1.,1.], [0.,1.]])

@pytest.fixture
def line_vertices() -> np.ndarray:
    """Two vertices forming a line (invalid polygon)."""
    return np.array([[0., 0.], [1., 1.]])

@pytest.fixture
def empty_vertices() -> np.ndarray:
    """Empty vertex array."""
    return np.empty((0, 2))

# --- Point Sets for check_2d_interpolation tests ---

@pytest.fixture
def points_inside_square() -> np.ndarray:
    return np.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])

@pytest.fixture
def points_outside_square() -> np.ndarray:
    return np.array([[1.5, 0.5], [-0.1, 0.8], [0.5, -0.2], [0.5, 1.1]])

@pytest.fixture
def points_on_boundary_square() -> np.ndarray:
    return np.array([[0.5, 0.0], [1.0, 0.5], [1.0, 1.0], [0.0, 0.5], [0.0, 0.0]])

@pytest.fixture
def empty_points() -> np.ndarray:
    return np.empty((0, 2))


# --- Tests for is_point_in_polygon ---

@pytest.mark.parametrize("point, expected", [
    (np.array([0.5, 0.5]), True),    # Clearly inside square
    (np.array([0.1, 0.9]), True),    # Inside near corner
    (np.array([2.0, 0.5]), False),   # Outside right
    (np.array([-0.1, 0.5]), False),  # Outside left
    (np.array([0.5, 1.5]), False),   # Outside top
    (np.array([0.5, -0.5]), False),  # Outside bottom
    # Boundary Cases (should be False due to strict check)
    (np.array([0.0, 0.0]), False),   # Vertex 0,0
    (np.array([1.0, 1.0]), False),   # Vertex 1,1
    (np.array([0.5, 0.0]), False),   # On bottom edge
    (np.array([1.0, 0.5]), False),   # On right edge
    (np.array([0.5, 1.0]), False),   # On top edge
    (np.array([0.0, 0.5]), False),   # On left edge
    # Points with same y as vertex but not on edge
    (np.array([-0.1, 1.0]), False), # Same y as top edge, but outside
    (np.array([0.5, 1.0]), False),  # On top edge (already tested, but good check)
    # Epsilon tests near boundary
    (np.array([0.0 + 1e-9, 0.5]), True), # Just inside left edge
    (np.array([1.0 - 1e-9, 0.5]), True), # Just inside right edge
    (np.array([0.5, 0.0 + 1e-9]), True), # Just inside bottom edge
    (np.array([0.5, 1.0 - 1e-9]), True), # Just inside top edge
])
def test_pip_square(square_vertices, point, expected):
    """Test point-in-polygon for a square, including boundaries."""
    assert is_point_in_polygon(point, square_vertices) == expected

@pytest.mark.parametrize("point, expected", [
    (np.array([1.0, 0.5]), True),   # Inside triangle
    (np.array([0.5, 0.2]), True),   # Inside
    (np.array([1.5, 0.2]), True),   # Inside
    (np.array([1.0, 1.49]), True),  # Inside near top vertex
    (np.array([1.0, 1.5]), False),  # Top vertex
    (np.array([0.0, 0.0]), False),  # Bottom-left vertex
    (np.array([1.0, 0.0]), False),  # On bottom edge
    (np.array([1.5, 0.75]), False), # On right edge (midpoint of (2,0) and (1,1.5))
    (np.array([1.0, 1.6]), False),  # Above triangle
    (np.array([-0.1, 0.0]), False), # Left of triangle
])
def test_pip_triangle(triangle_vertices, point, expected):
    """Test point-in-polygon for a triangle, including boundaries."""
    assert is_point_in_polygon(point, triangle_vertices) == expected

@pytest.mark.parametrize("point, expected", [
    (np.array([0.5, 0.5]), True),   # Inside lower body
    (np.array([1.5, 1.5]), True),   # Inside upper arm
    (np.array([1.5, 0.5]), True),   # Inside notch/indent
    (np.array([2.5, 0.5]), True),   # Inside lower body right
    (np.array([1.5, 2.0]), True),   # Inside upper arm middle
    (np.array([1.5, 2.1]), False),  # Above upper arm
    (np.array([4.0, 0.5]), False),  # Right of shape
    (np.array([-1.0, 0.5]), False), # Left of shape
    # Boundary Cases
    (np.array([0.0, 0.0]), False),  # Vertex
    (np.array([1.5, 0.0]), False),  # On bottom edge
    (np.array([3.0, 0.5]), False),  # On right edge
    (np.array([2.5, 1.0]), False),  # On horizontal edge inside indent
    (np.array([2.0, 1.5]), False),  # On vertical edge inside indent
    (np.array([1.5, 2.0]), True),   # Re-test known inside point
    (np.array([1.0, 1.0]), False),  # Vertex in indent corner
])
def test_pip_concave(concave_vertices, point, expected):
    """Test point-in-polygon for a concave shape, including boundaries."""
    assert is_point_in_polygon(point, concave_vertices) == expected

def test_pip_invalid_polygon(line_vertices):
    """Test point-in-polygon with fewer than 3 vertices."""
    point = np.array([0.5, 0.5])
    assert is_point_in_polygon(point, line_vertices) == False
    assert is_point_in_polygon(point, np.array([[0,0]])) == False
    assert is_point_in_polygon(point, np.empty((0,2))) == False

def test_pip_invalid_inputs():
    """Test point-in-polygon with invalid input types/shapes."""
    poly = np.array([[0., 0.], [1., 0.], [0., 1.]])
    assert is_point_in_polygon(np.array([0.1]), poly) == False # Wrong point shape
    assert is_point_in_polygon(np.array([0.1, 0.1, 0.1]), poly) == False # Wrong point shape
    assert is_point_in_polygon(np.array([0.1, 0.1]), np.array([1,2,3])) == False # Wrong poly shape
    assert is_point_in_polygon(np.array([0.1, 0.1]), np.array([[1],[2],[3]])) == False # Wrong poly shape (N,1)


# --- Tests for check_2d_interpolation ---

def test_check_one_point_inside(points_inside_square, square_vertices):
    """Test with points where at least one is inside."""
    # Modify fixture to ensure only one point is inside for this specific test
    points = np.array([[0.5, 0.5], [1.5, 1.5]]) # One in, one out
    assert check_2d_interpolation(points, square_vertices) == 1

def test_check_multiple_points_inside(points_inside_square, square_vertices):
    """Test with multiple points inside."""
    assert check_2d_interpolation(points_inside_square, square_vertices) == 1

def test_check_no_points_inside(points_outside_square, square_vertices):
    """Test with all points outside."""
    assert check_2d_interpolation(points_outside_square, square_vertices) == 0

def test_check_points_on_boundary(points_on_boundary_square, square_vertices):
    """Test with all points on the boundary (should return 0 due to strict check)."""
    assert check_2d_interpolation(points_on_boundary_square, square_vertices) == 0

def test_check_mixed_points(points_inside_square, points_outside_square, points_on_boundary_square, square_vertices):
    """Test with a mix of inside, outside, and boundary points."""
    mixed_points = np.vstack((points_outside_square, points_on_boundary_square, points_inside_square))
    # Shuffle points to ensure order doesn't matter and early exit works
    np.random.shuffle(mixed_points)
    assert check_2d_interpolation(mixed_points, square_vertices) == 1

def test_check_empty_points(empty_points, square_vertices):
    """Test with an empty list of points to check."""
    assert check_2d_interpolation(empty_points, square_vertices) == 0

def test_check_invalid_polygon(points_inside_square, line_vertices):
    """Test checking against an invalid polygon (fewer than 3 vertices)."""
    assert check_2d_interpolation(points_inside_square, line_vertices) == 0

def test_check_empty_polygon(points_inside_square, empty_vertices):
    """Test checking against an empty polygon vertex list."""
    assert check_2d_interpolation(points_inside_square, empty_vertices) == 0

def test_check_invalid_input_shapes():
    """Test check_2d_interpolation with invalid input shapes."""
    valid_points = np.array([[0.5, 0.5]])
    valid_poly = np.array([[0,0],[1,0],[0,1]])
    invalid_points_1d = np.array([1,2,3])
    invalid_poly_1d = np.array([1,2,3,4,5,6])
    invalid_points_3d = np.array([[[1,2],[3,4]]])

    assert check_2d_interpolation(invalid_points_1d, valid_poly) == 0
    assert check_2d_interpolation(valid_points, invalid_poly_1d) == 0
    assert check_2d_interpolation(invalid_points_3d, valid_poly) == 0
    # Test None inputs explicitly
    assert check_2d_interpolation(None, valid_poly) == 0 # type: ignore
    assert check_2d_interpolation(valid_points, None) == 0 # type: ignore
