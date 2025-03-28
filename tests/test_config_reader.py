import pytest
import sys
import os
import json
import io
from typing import Dict, Any, Generator # For type hinting fixtures
from pathlib import Path # For tmp_path

# Assuming the tests directory is at the same level as the main package directory
# Adjust the path if your structure is different or use package installation
try:
    from cad_collision_analyzer.config_reader import load_config, DEFAULT_CONFIG
except ImportError:
    # If running pytest from the root directory, this might be needed
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from cad_collision_analyzer.config_reader import load_config, DEFAULT_CONFIG


# --- Fixtures ---

@pytest.fixture
def valid_config_data() -> Dict[str, Any]:
    """Provides valid configuration data."""
    return {
        "directions": [[1.0, 0.0, 0.0], [0, 1, 0], [0, 0, -1.5]],
        "num_samples": 50
    }

@pytest.fixture
def create_config_file(tmp_path: Path):
    """Factory fixture to create temporary config files."""
    def _create_file(filename: str, content: str) -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path
    return _create_file

# --- Test Cases ---

def test_load_valid_config(create_config_file, valid_config_data: Dict[str, Any]):
    """Tests loading a correctly formatted config file."""
    config_path = create_config_file("valid_config.json", json.dumps(valid_config_data))
    loaded_config = load_config(str(config_path))
    assert loaded_config == valid_config_data

def test_load_config_file_not_found(tmp_path: Path, capsys):
    """Tests behavior when the config file does not exist."""
    non_existent_path = tmp_path / "non_existent_config.json"
    loaded_config = load_config(str(non_existent_path))

    # Check if default config is returned
    assert loaded_config == DEFAULT_CONFIG

    # Check for warning message in stderr
    captured = capsys.readouterr()
    assert f"Warning: Configuration file '{non_existent_path}' not found." in captured.err

def test_load_malformed_json(create_config_file, capsys):
    """Tests behavior with an invalid JSON file."""
    malformed_content = '{"directions": [[1,0,0], "num_samples": 50,' # Missing closing brace
    config_path = create_config_file("malformed.json", malformed_content)
    loaded_config = load_config(str(config_path))

    # Check if default config is returned
    assert loaded_config == DEFAULT_CONFIG

    # Check for warning message in stderr
    captured = capsys.readouterr()
    assert f"Warning: Could not decode JSON from configuration file '{config_path}'." in captured.err

def test_missing_directions_key(create_config_file, capsys):
    """Tests exit behavior when 'directions' key is missing."""
    config_data = {"num_samples": 100}
    config_path = create_config_file("missing_directions.json", json.dumps(config_data))

    with pytest.raises(SystemExit) as e:
        load_config(str(config_path))

    assert e.type == SystemExit
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert f"Error: Missing 'directions' key in configuration file '{config_path}'." in captured.err

def test_missing_num_samples_key(create_config_file, capsys):
    """Tests exit behavior when 'num_samples' key is missing."""
    config_data = {"directions": [[1, 0, 0]]}
    config_path = create_config_file("missing_num_samples.json", json.dumps(config_data))

    with pytest.raises(SystemExit) as e:
        load_config(str(config_path))

    assert e.type == SystemExit
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert f"Error: Missing 'num_samples' key in configuration file '{config_path}'." in captured.err

def test_invalid_num_samples_type(create_config_file, capsys):
    """Tests exit behavior when 'num_samples' is not an integer."""
    config_data = {"directions": [[1, 0, 0]], "num_samples": "100"} # num_samples is string
    config_path = create_config_file("invalid_num_samples.json", json.dumps(config_data))

    with pytest.raises(SystemExit) as e:
        load_config(str(config_path))

    assert e.type == SystemExit
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "Error: 'num_samples' must be an integer" in captured.err
    assert f"Found type: <class 'str'>" in captured.err

def test_invalid_directions_type_not_list(create_config_file, capsys):
    """Tests exit behavior when 'directions' is not a list."""
    config_data = {"directions": "[[1, 0, 0]]", "num_samples": 100} # directions is string
    config_path = create_config_file("invalid_directions_type.json", json.dumps(config_data))

    with pytest.raises(SystemExit) as e:
        load_config(str(config_path))

    assert e.type == SystemExit
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "Error: 'directions' must be a list" in captured.err
    assert f"Found type: <class 'str'>" in captured.err

def test_invalid_direction_item_type(create_config_file, capsys):
    """Tests exit behavior when an item in 'directions' is not a list/tuple."""
    config_data = {"directions": [[1, 0, 0], "not a list"], "num_samples": 100}
    config_path = create_config_file("invalid_direction_item.json", json.dumps(config_data))

    with pytest.raises(SystemExit) as e:
        load_config(str(config_path))

    assert e.type == SystemExit
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "Error: Each item in 'directions' must be a list or tuple" in captured.err
    assert "Found type: <class 'str'> at index 1" in captured.err


def test_invalid_direction_item_length(create_config_file, capsys):
    """Tests exit behavior when a direction vector has incorrect length."""
    config_data = {"directions": [[1, 0, 0], [1, 2]], "num_samples": 100} # Second vector has length 2
    config_path = create_config_file("invalid_direction_length.json", json.dumps(config_data))

    with pytest.raises(SystemExit) as e:
        load_config(str(config_path))

    assert e.type == SystemExit
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "Error: Each direction vector in 'directions' must have exactly 3 elements" in captured.err
    assert "Found 2 elements at index 1" in captured.err


def test_invalid_direction_element_type(create_config_file, capsys):
    """Tests exit behavior when an element within a direction vector is not a number."""
    config_data = {"directions": [[1, 0, 0], [1, "two", 3]], "num_samples": 100} # "two" is string
    config_path = create_config_file("invalid_direction_element.json", json.dumps(config_data))

    with pytest.raises(SystemExit) as e:
        load_config(str(config_path))

    assert e.type == SystemExit
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "Error: Each element within a direction vector must be a number (int or float)" in captured.err
    assert "Found type: <class 'str'> at index 1, element 1" in captured.err

def test_default_config_is_returned_on_other_read_error(mocker, capsys):
    """Tests that default config is returned on unexpected file read errors."""
    # Mock os.path.exists to return True
    mocker.patch('os.path.exists', return_value=True)
    # Mock open to raise an unexpected error (e.g., PermissionError)
    mocker.patch('builtins.open', side_effect=PermissionError("Permission denied"))

    config_path = "some_protected_file.json"
    loaded_config = load_config(config_path)

    assert loaded_config == DEFAULT_CONFIG
    captured = capsys.readouterr()
    assert f"Warning: An error occurred while reading configuration file '{config_path}'." in captured.err
    assert "Permission denied" in captured.err # Check if the original error is mentioned
