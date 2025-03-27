import json
import os
import sys
from typing import Dict, List, Any, Tuple, Union

# Define default configuration values
DEFAULT_DIRECTIONS: List[List[int]] = [
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
]
DEFAULT_NUM_SAMPLES: int = 100
DEFAULT_CONFIG: Dict[str, Any] = {
    "directions": DEFAULT_DIRECTIONS,
    "num_samples": DEFAULT_NUM_SAMPLES
}

def _validate_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Validates the structure and types of the loaded configuration.

    Args:
        config: The configuration dictionary loaded from the JSON file.
        config_path: The path to the config file (used for error messages).

    Raises:
        SystemExit: If validation fails, prints an error and exits.
    """
    # --- Check for mandatory keys ---
    if "directions" not in config:
        print(f"Error: Missing 'directions' key in configuration file '{config_path}'.", file=sys.stderr)
        sys.exit(1)

    if "num_samples" not in config:
        print(f"Error: Missing 'num_samples' key in configuration file '{config_path}'.", file=sys.stderr)
        sys.exit(1)

    # --- Validate 'num_samples' type ---
    if not isinstance(config["num_samples"], int):
        print(f"Error: 'num_samples' must be an integer in configuration file '{config_path}'. "
              f"Found type: {type(config['num_samples'])}.", file=sys.stderr)
        sys.exit(1)

    # --- Validate 'directions' structure and types ---
    if not isinstance(config["directions"], list):
        print(f"Error: 'directions' must be a list in configuration file '{config_path}'. "
              f"Found type: {type(config['directions'])}.", file=sys.stderr)
        sys.exit(1)

    for i, direction in enumerate(config["directions"]):
        if not isinstance(direction, (list, tuple)):
            print(f"Error: Each item in 'directions' must be a list or tuple in configuration file '{config_path}'. "
                  f"Found type: {type(direction)} at index {i}.", file=sys.stderr)
            sys.exit(1)
        if len(direction) != 3:
            print(f"Error: Each direction vector in 'directions' must have exactly 3 elements in configuration file '{config_path}'. "
                  f"Found {len(direction)} elements at index {i}.", file=sys.stderr)
            sys.exit(1)
        for j, num in enumerate(direction):
            if not isinstance(num, (int, float)):
                print(f"Error: Each element within a direction vector must be a number (int or float) in configuration file '{config_path}'. "
                      f"Found type: {type(num)} at index {i}, element {j}.", file=sys.stderr)
                sys.exit(1)


def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Loads configuration settings from a JSON file.

    Reads the specified JSON file. If the file doesn't exist or is invalid JSON,
    it prints a warning and returns default configuration values.
    If the file is valid JSON but lacks required keys ('directions', 'num_samples')
    or has incorrect data types, it prints an error and exits the script.

    Args:
        config_path: The path to the JSON configuration file.
                     Defaults to 'config.json' in the current working directory.

    Returns:
        A dictionary containing the configuration settings. Returns default
        settings if the file is not found or cannot be parsed as JSON.

    Raises:
        SystemExit: If mandatory keys are missing or data types are invalid
                    after successfully parsing the JSON file.
    """
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file '{config_path}' not found. Using default configuration.", file=sys.stderr)
        return DEFAULT_CONFIG.copy() # Return a copy to prevent modification of defaults

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not decode JSON from configuration file '{config_path}'. Error: {e}. Using default configuration.", file=sys.stderr)
        return DEFAULT_CONFIG.copy() # Return a copy
    except Exception as e: # Catch other potential file reading errors
        print(f"Warning: An error occurred while reading configuration file '{config_path}'. Error: {e}. Using default configuration.", file=sys.stderr)
        return DEFAULT_CONFIG.copy() # Return a copy

    # Validate the loaded configuration
    _validate_config(config, config_path)

    return config

if __name__ == '__main__':
    # Example usage when running this script directly
    # In the actual application, 'main.py' would likely call load_config()
    print("Attempting to load default config.json...")
    cfg = load_config()
    print("Loaded configuration:")
    print(json.dumps(cfg, indent=2))

    # Example loading a specific file (if it exists)
    specific_path = 'my_specific_config.json'
    if os.path.exists(specific_path):
        print(f"\nAttempting to load {specific_path}...")
        cfg_specific = load_config(specific_path)
        print("Loaded configuration:")
        print(json.dumps(cfg_specific, indent=2))
    else:
        print(f"\n{specific_path} does not exist, skipping specific load example.")
