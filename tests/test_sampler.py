import pytest
import numpy as np
import logging

# --- Module Under Test ---
try:
    # Assuming tests run from root or PYTHONPATH includes project src
    from cad_collision_analyzer.sampler import sample_polygon_points
    # Also need the function it depends on for rigorous checking
    from cad_collision_analyzer.interpolator_2d import is_point_in_polygon
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from cad_collision_analyzer.sampler import sample_polygon_points
    from cad_collision_analyzer.interpolator_2d import is_point_in_polygon


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
def line_vertices() -> np.ndarray:
    """Two vertices forming a line."""
    return np.array([[0., 0.], [1., 1.]])

@pytest.fixture
def point_vertices() -> np.ndarray:
    """Single vertex."""
    return np.array([[0., 0.]])


@pytest.fixture
def empty_vertices() -> np.ndarray:
    """Empty vertex array."""
    return np.empty((0, 2))

# --- Tests for sample_polygon_points ---

def test_sample_zero_samples(square_vertices):
    """Test sampling 0 points returns original vertices."""
    result = sample_polygon_points(square_vertices, 0)
    assert np.array_equal(result, square_vertices)
    assert result.shape == square_vertices.shape

def test_sample_negative_samples(square_vertices):
    """Test sampling negative points returns original vertices."""
    result = sample_polygon_points(square_vertices, -5)
    assert np.array_equal(result, square_vertices)
    assert result.shape == square_vertices.shape


def test_sample_positive_samples_square(square_vertices):
    """Test sampling positive number of points from a square."""
    num_samples = 10
    num_vertices = square_vertices.shape[0]
    result = sample_polygon_points(square_vertices, num_samples)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 2
    # Check if number of samples is approximately correct (allow for undershoot)
    assert result.shape[0] >= num_vertices
    assert result.shape[0] <= num_vertices + num_samples

    # Check original vertices are preserved at the start
    assert np.array_equal(result[:num_vertices, :], square_vertices)

    # Check sampled points are within bounding box AND strictly inside polygon
    if result.shape[0] > num_vertices:
        sampled = result[num_vertices:, :]
        min_xy = np.min(square_vertices, axis=0) - 1e-9 # Add tolerance for check
        max_xy = np.max(square_vertices, axis=0) + 1e-9 # Add tolerance for check
        assert np.all(sampled >= min_xy)
        assert np.all(sampled <= max_xy)
        # More rigorous check: are sampled points actually inside?
        for i in range(sampled.shape[0]):
            assert is_point_in_polygon(sampled[i,:], square_vertices), f"Sampled point {sampled[i,:]} is not inside square"


def test_sample_positive_samples_triangle(triangle_vertices):
    """Test sampling positive number of points from a triangle."""
    num_samples = 5
    num_vertices = triangle_vertices.shape[0]
    result = sample_polygon_points(triangle_vertices, num_samples)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 2
    assert result.shape[0] >= num_vertices
    assert result.shape[0] <= num_vertices + num_samples
    assert np.array_equal(result[:num_vertices, :], triangle_vertices)

    if result.shape[0] > num_vertices:
        sampled = result[num_vertices:, :]
        min_xy = np.min(triangle_vertices, axis=0) - 1e-9
        max_xy = np.max(triangle_vertices, axis=0) + 1e-9
        assert np.all(sampled[:, 0] >= min_xy[0])
        assert np.all(sampled[:, 1] >= min_xy[1])
        assert np.all(sampled[:, 0] <= max_xy[0])
        assert np.all(sampled[:, 1] <= max_xy[1])
        # More rigorous check
        for i in range(sampled.shape[0]):
             assert is_point_in_polygon(sampled[i,:], triangle_vertices), f"Sampled point {sampled[i,:]} is not inside triangle"


def test_sample_fewer_than_3_vertices(line_vertices, caplog):
    """Test sampling with fewer than 3 vertices logs warning and returns input."""
    num_samples = 5
    with caplog.at_level(logging.WARNING):
        result = sample_polygon_points(line_vertices, num_samples)
    assert np.array_equal(result, line_vertices)
    assert "Cannot sample polygon with fewer than 3 vertices" in caplog.text

def test_sample_point_input(point_vertices, caplog):
    """Test sampling with only 1 vertex."""
    num_samples = 5
    with caplog.at_level(logging.WARNING):
        result = sample_polygon_points(point_vertices, num_samples)
    assert np.array_equal(result, point_vertices)
    assert "Cannot sample polygon with fewer than 3 vertices" in caplog.text


def test_sample_empty_vertices(empty_vertices, caplog):
    """Test sampling with empty vertex input returns empty and logs."""
    num_samples = 5
    # Expect warning for < 3 vertices OR error for invalid shape depending on checks
    with caplog.at_level(logging.WARNING):
         result = sample_polygon_points(empty_vertices, num_samples)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 2)
    assert "Cannot sample polygon with fewer than 3 vertices" in caplog.text


def test_sample_invalid_input_shape(caplog):
    """Test sampling with invalid input shape (e.g., 1D array)."""
    invalid_input = np.array([1., 2., 3., 4.])
    num_samples = 5
    with caplog.at_level(logging.ERROR):
        result = sample_polygon_points(invalid_input, num_samples)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 2)
    assert "Invalid input: polygon_vertices must be a NumPy array" in caplog.text


def test_sample_failure_warning(square_vertices, mocker, caplog):
    """Test warning log if max attempts are reached before getting enough samples."""
    num_samples = 10
    # Mock is_point_in_polygon to always return False
    mocker.patch('cad_collision_analyzer.sampler.is_point_in_polygon', return_value=False)

    with caplog.at_level(logging.WARNING):
        result = sample_polygon_points(square_vertices, num_samples)

    # Should return only original vertices
    assert np.array_equal(result, square_vertices)
    assert result.shape == square_vertices.shape
    # Check for the specific warning about reaching max attempts
    assert "Reached maximum attempts" in caplog.text
    assert f"only generated 0 / {num_samples} samples" in caplog.text
    # Also check for the warning about no points being sampled
    assert "No points were successfully sampled" in caplog.text
