# cad_collision_analyzer/config_reader.py

import json
import os
import sys
import logging
from typing import Dict, List, Any, Union

# Module-specific logger
logger = logging.getLogger("CADAnalyzer.config_reader")

# --- Default Configuration Values ---
DEFAULT_DIRECTIONS: List[List[float]] = [
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]
]
DEFAULT_NUM_SAMPLES: int = 100
DEFAULT_CONTACT_TOLERANCE: float = 0.0001
DEFAULT_LINEAR_DEFLECTION: float = 0.1
DEFAULT_ANGULAR_DEFLECTION: float = 0.5

# Combine defaults into a dictionary
DEFAULT_CONFIG: Dict[str, Any] = {
    "directions": DEFAULT_DIRECTIONS,
    "num_samples": DEFAULT_NUM_SAMPLES,
    "contact_tolerance": DEFAULT_CONTACT_TOLERANCE,
    "linear_deflection": DEFAULT_LINEAR_DEFLECTION,
    "angular_deflection": DEFAULT_ANGULAR_DEFLECTION,
}

def _validate_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Validates the structure and types of the loaded configuration dictionary.

    Logs errors and returns False if validation fails.

    Args:
        config: The configuration dictionary loaded from the JSON file.
        config_path: The path to the config file (for error messages).

    Returns:
        True if the configuration is valid, False otherwise.
    """
    is_valid = True

    # --- Validate 'directions' ---
    if "directions" not in config:
        logger.error(f"Config Error: Missing 'directions' key in '{config_path}'. Using default.")
        config["directions"] = DEFAULT_DIRECTIONS # Use default
    elif not isinstance(config["directions"], list):
        logger.error(f"Config Error: 'directions' must be a list in '{config_path}'. Found type: {type(config['directions'])}. Using default.")
        config["directions"] = DEFAULT_DIRECTIONS # Use default
        is_valid = False # Mark as invalid even if using default
    else:
        for i, direction in enumerate(config["directions"]):
            if not isinstance(direction, (list, tuple)):
                logger.error(f"Config Error: Item at index {i} in 'directions' must be a list or tuple in '{config_path}'. Found type: {type(direction)}. Skipping this direction.")
                # Remove invalid direction? Or fail validation? Let's just log and potentially skip later.
                # For now, mark config as invalid.
                is_valid = False
                continue # Skip further checks for this item
            if len(direction) != 3:
                logger.error(f"Config Error: Direction vector at index {i} in 'directions' must have exactly 3 elements in '{config_path}'. Found {len(direction)}. Skipping this direction.")
                is_valid = False
                continue
            for j, num in enumerate(direction):
                if not isinstance(num, (int, float)):
                    logger.error(f"Config Error: Element {j} in direction vector at index {i} ('directions') must be a number (int/float) in '{config_path}'. Found type: {type(num)}. Skipping this direction.")
                    is_valid = False
                    break # Stop checking elements in this invalid direction


    # --- Validate 'num_samples' ---
    if "num_samples" not in config:
        logger.warning(f"Config Warning: Missing 'num_samples' key in '{config_path}'. Using default: {DEFAULT_NUM_SAMPLES}.")
        config["num_samples"] = DEFAULT_NUM_SAMPLES
    elif not isinstance(config["num_samples"], int):
        logger.error(f"Config Error: 'num_samples' must be an integer in '{config_path}'. Found type: {type(config['num_samples'])}. Using default: {DEFAULT_NUM_SAMPLES}.")
        config["num_samples"] = DEFAULT_NUM_SAMPLES
        is_valid = False
    elif config["num_samples"] < 0:
         logger.error(f"Config Error: 'num_samples' cannot be negative in '{config_path}'. Found: {config['num_samples']}. Using 0 instead.")
         config["num_samples"] = 0
         is_valid = False

    # --- Validate 'contact_tolerance' ---
    if "contact_tolerance" not in config:
        logger.warning(f"Config Warning: Missing 'contact_tolerance' key in '{config_path}'. Using default: {DEFAULT_CONTACT_TOLERANCE}.")
        config["contact_tolerance"] = DEFAULT_CONTACT_TOLERANCE
    elif not isinstance(config["contact_tolerance"], (int, float)):
        logger.error(f"Config Error: 'contact_tolerance' must be a number in '{config_path}'. Found type: {type(config['contact_tolerance'])}. Using default: {DEFAULT_CONTACT_TOLERANCE}.")
        config["contact_tolerance"] = DEFAULT_CONTACT_TOLERANCE
        is_valid = False
    elif config["contact_tolerance"] < 0:
        logger.error(f"Config Error: 'contact_tolerance' cannot be negative in '{config_path}'. Found: {config['contact_tolerance']}. Using default: {DEFAULT_CONTACT_TOLERANCE}.")
        config["contact_tolerance"] = DEFAULT_CONTACT_TOLERANCE
        is_valid = False

    # --- Validate 'linear_deflection' ---
    if "linear_deflection" not in config:
        logger.warning(f"Config Warning: Missing 'linear_deflection' key in '{config_path}'. Using default: {DEFAULT_LINEAR_DEFLECTION}.")
        config["linear_deflection"] = DEFAULT_LINEAR_DEFLECTION
    elif not isinstance(config["linear_deflection"], (int, float)):
        logger.error(f"Config Error: 'linear_deflection' must be a number in '{config_path}'. Found type: {type(config['linear_deflection'])}. Using default: {DEFAULT_LINEAR_DEFLECTION}.")
        config["linear_deflection"] = DEFAULT_LINEAR_DEFLECTION
        is_valid = False
    elif config["linear_deflection"] <= 0:
        logger.error(f"Config Error: 'linear_deflection' must be positive in '{config_path}'. Found: {config['linear_deflection']}. Using default: {DEFAULT_LINEAR_DEFLECTION}.")
        config["linear_deflection"] = DEFAULT_LINEAR_DEFLECTION
        is_valid = False

    # --- Validate 'angular_deflection' ---
    if "angular_deflection" not in config:
        logger.warning(f"Config Warning: Missing 'angular_deflection' key in '{config_path}'. Using default: {DEFAULT_ANGULAR_DEFLECTION}.")
        config["angular_deflection"] = DEFAULT_ANGULAR_DEFLECTION
    elif not isinstance(config["angular_deflection"], (int, float)):
        logger.error(f"Config Error: 'angular_deflection' must be a number in '{config_path}'. Found type: {type(config['angular_deflection'])}. Using default: {DEFAULT_ANGULAR_DEFLECTION}.")
        config["angular_deflection"] = DEFAULT_ANGULAR_DEFLECTION
        is_valid = False
    elif config["angular_deflection"] <= 0:
        logger.error(f"Config Error: 'angular_deflection' must be positive in '{config_path}'. Found: {config['angular_deflection']}. Using default: {DEFAULT_ANGULAR_DEFLECTION}.")
        config["angular_deflection"] = DEFAULT_ANGULAR_DEFLECTION
        is_valid = False

    # Add checks for any other expected keys here...

    # Filter out invalid directions before returning
    if isinstance(config.get("directions"), list):
        valid_directions = []
        for direction in config["directions"]:
             if isinstance(direction, (list, tuple)) and len(direction) == 3 and all(isinstance(n, (int, float)) for n in direction):
                 valid_directions.append(list(direction)) # Ensure it's a list of lists
        if len(valid_directions) != len(config["directions"]):
             logger.warning(f"Removed invalid direction entries from config '{config_path}'. Using {len(valid_directions)} valid directions.")
             config["directions"] = valid_directions


    return is_valid


def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Loads configuration settings from a JSON file.

    Reads the specified JSON file. If the file doesn't exist or is invalid JSON,
    it logs a warning and returns default configuration values.
    If the file is valid JSON but contains invalid values for keys, it logs errors,
    attempts to use default values for the invalid keys, and returns the resulting
    configuration dictionary.

    Args:
        config_path: The path to the JSON configuration file.
                     Defaults to 'config.json' in the current working directory.

    Returns:
        A dictionary containing the configuration settings, potentially mixing
        loaded values with defaults if errors were encountered.

    Raises:
        SystemExit: If the configuration file exists but cannot be read or parsed as JSON.
                    (This indicates a more severe problem than just invalid values).
    """
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file '{config_path}' not found. Using default configuration.")
        # Return a deep copy to prevent modification of global defaults
        return json.loads(json.dumps(DEFAULT_CONFIG))

    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"FATAL: Could not decode JSON from configuration file '{config_path}'. Error: {e}. Exiting.")
        print(f"\nFATAL ERROR: Invalid JSON in config file '{config_path}'. Please fix the JSON structure.", file=sys.stderr)
        sys.exit(1) # Exit if JSON is fundamentally broken
    except IOError as e:
        logger.error(f"FATAL: Could not read configuration file '{config_path}'. Error: {e}. Exiting.")
        print(f"\nFATAL ERROR: Cannot read config file '{config_path}'. Check permissions.", file=sys.stderr)
        sys.exit(1) # Exit if file cannot be read
    except Exception as e:
        logger.error(f"FATAL: An unexpected error occurred while reading configuration file '{config_path}'. Error: {e}. Exiting.", exc_info=True)
        sys.exit(1) # Exit on other unexpected errors

    # --- Create final config starting with defaults, then update with loaded values ---
    # This ensures all expected keys exist.
    final_config = json.loads(json.dumps(DEFAULT_CONFIG)) # Deep copy of defaults
    final_config.update(loaded_config) # Update with values from file

    # --- Validate the merged configuration ---
    if not _validate_config(final_config, config_path):
        logger.warning(f"Configuration file '{config_path}' contained errors. Defaults were used for invalid fields. Please review the configuration and logs.")
        # Do not exit here, allow running with defaults for problematic fields

    logger.info("Configuration loaded and validated (using defaults where necessary).")
    return final_config

# Example usage
if __name__ == '__main__':
    # Configure logging for standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("--- Loading default config (config.json) ---")
    cfg = load_config()
    print("Loaded configuration:")
    print(json.dumps(cfg, indent=2))

    # Create a dummy invalid config file for testing
    invalid_config_content = {
        "directions": [[1, 0, 0], [0, 1], [0, 0, 'a']], # Invalid entries
        "num_samples": "150", # Invalid type
        "contact_tolerance": -0.01, # Invalid value
        # linear_deflection missing -> warning, use default
        "angular_deflection": "high" # Invalid type
    }
    invalid_config_path = 'invalid_test_config.json'
    try:
        with open(invalid_config_path, 'w') as f:
            json.dump(invalid_config_content, f, indent=2)
        print(f"\n--- Loading invalid config ({invalid_config_path}) ---")
        cfg_invalid = load_config(invalid_config_path)
        print("Loaded configuration (with defaults applied for errors):")
        print(json.dumps(cfg_invalid, indent=2))
    except Exception as e:
        print(f"Error during invalid config test: {e}")
    finally:
        if os.path.exists(invalid_config_path):
            os.remove(invalid_config_path) # Clean up test file

    print("\n--- Loading non-existent config ---")
    cfg_nonexistent = load_config('non_existent_config.json')
    print("Loaded configuration:")
    print(json.dumps(cfg_nonexistent, indent=2))

