# main_refactored.py

import argparse
import os
import sys
import datetime
import logging
import numpy as np
import threading
import queue # Using queue for thread-safe result collection
import time
from typing import List, Dict, Tuple, Any, Optional

# --- Import project modules ---
try:
    from cad_collision_analyzer import config_reader
    from cad_collision_analyzer import cad_reader
    from cad_collision_analyzer import mesh_converter
    from cad_collision_analyzer import collision_detector_3d
    from cad_collision_analyzer import projector
    from cad_collision_analyzer import sampler
    from cad_collision_analyzer import interpolator_2d
    from cad_collision_analyzer import excel_writer
    # Import project-specific exception
    from cad_collision_analyzer.mesh_converter import MeshConversionError
    # Import OCC types if needed for type hinting (optional)
    from OCC.Core.TopoDS import TopoDS_Shape
    import trimesh

except ImportError as e:
    print(f"FATAL ERROR: Failed to import project modules. "
          f"Ensure the 'cad_collision_analyzer' package is in the Python path "
          f"and all dependencies (pythonocc-core, trimesh, numpy, openpyxl) are installed. Details: {e}", file=sys.stderr)
    sys.exit(1)

# --- Global Logger Setup ---
# Configure root logger
log_formatter = logging.Formatter('%(asctime)s - %(name)s [%(threadName)s] - %(levelname)s - %(message)s')
logger = logging.getLogger("CADAnalyzer") # Get a specific logger for the application
logger.setLevel(logging.INFO) # Set default level

# Console Handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# Optional File Handler (uncomment to enable file logging)
# file_handler = logging.FileHandler("cad_analyzer.log", mode='w') # Overwrite log file each run
# file_handler.setFormatter(log_formatter)
# logger.addHandler(file_handler)


# --- Worker Function for Parallel Direction Processing ---

def process_direction_worker(
    direction_vector: List[float],
    successful_components: List[Dict[str, Any]], # Contains dicts {name, mesh}
    component_names: List[str],
    num_successful: int,
    num_samples: int,
    results_queue: queue.Queue, # Thread-safe queue for results {sheet_name: matrix}
    error_event: threading.Event # Shared event to signal critical errors
):
    """
    Worker function executed by each thread to process one projection direction.
    Performs projection, sampling, and interpolation checks for all component pairs.
    Puts results (sheet_name, matrix) into the results_queue if successful.
    Sets error_event on critical failure.

    Args:
        direction_vector: The 3D vector for projection.
        successful_components: List of dictionaries, each containing 'name' and 'mesh' (trimesh.Trimesh).
        component_names: List of names corresponding to successful_components.
        num_successful: Number of successfully meshed components.
        num_samples: Number of points to sample inside each projected polygon.
        results_queue: Thread-safe queue to store results.
        error_event: Thread-safe event to signal critical errors across threads.
    """
    thread_name = threading.current_thread().name
    logger.info(f"Starting processing for direction: {direction_vector}")

    # --- 1. Normalize Direction ---
    try:
        dir_vec = np.array(direction_vector, dtype=np.float64)
        norm = np.linalg.norm(dir_vec)
        # Use a small tolerance for zero check
        if np.isclose(norm, 0.0, atol=1e-9):
            logger.warning(f"Skipping zero or near-zero vector direction: {direction_vector}")
            return # Exit thread for this direction
        direction_norm = dir_vec / norm
    except Exception as e:
        logger.error(f"Could not normalize direction {direction_vector}: {e}. Skipping.", exc_info=True)
        # This might not be critical enough to stop everything, depends on requirements
        # error_event.set() # Uncomment if any direction failure should stop all 2D analysis
        return # Exit thread for this direction

    # --- 2. Pre-calculate Projections and Samples for all components in this direction ---
    logger.info(f"Generating projections and samples for {num_successful} components...")
    all_projections = [None] * num_successful # Stores projected polygon vertices (np.ndarray N,2)
    all_samples = [None] * num_successful     # Stores sampled points (np.ndarray M,2)
    projection_valid = [False] * num_successful # Flags if projection was successful

    projection_plane_origin = None # Use the centroid of the first component as the common plane origin

    for k in range(num_successful):
        # Check if a critical error was signaled by another thread - exit early
        if error_event.is_set():
             logger.warning(f"Halting pre-calculation for direction {direction_vector} due to error signaled by another thread.")
             return # Exit thread

        comp_k = successful_components[k]
        mesh_k: trimesh.Trimesh = comp_k["mesh"]
        name_k: str = comp_k["name"]
        logger.debug(f"Processing component {k} ('{name_k}')...")

        try:
            # Calculate center only once for the first component
            if projection_plane_origin is None:
                 center_k = projector.calculate_geometric_center(mesh_k)
                 if center_k is None: # Handle potential failure in centroid calculation
                     logger.error(f"Failed to calculate geometric center for component {k} ('{name_k}'). Cannot define projection plane. Setting error flag.")
                     error_event.set()
                     return
                 projection_plane_origin = center_k
                 logger.info(f"Using center of '{name_k}' as projection plane origin: {projection_plane_origin}")


            # Project mesh onto the common plane
            poly_verts_k = projector.project_mesh_onto_plane(mesh_k, direction_norm, projection_plane_origin)

            # Check if projection resulted in a valid polygon (at least 3 vertices)
            if poly_verts_k is None or poly_verts_k.shape[0] < 3:
                logger.warning(f"Component {k} ('{name_k}') projection resulted in < 3 vertices for direction {direction_vector}. Skipping interpolation checks involving this component in this direction.")
                # projection_valid[k] remains False
                continue # Proceed to next component

            projection_valid[k] = True
            all_projections[k] = poly_verts_k

            # Sample points only if projection is valid and sampling is requested
            if num_samples > 0:
                # Pass only the polygon vertices to the sampler
                samples_k = sampler.sample_polygon_points(poly_verts_k, num_samples)
                # Sampler returns original vertices + sampled points, or just vertices if sampling fails/num_samples=0
                all_samples[k] = samples_k
                # Log if sampling didn't generate the expected number of *additional* points
                num_added_samples = samples_k.shape[0] - poly_verts_k.shape[0]
                if num_added_samples < num_samples:
                     logger.warning(f"Sampler generated {num_added_samples}/{num_samples} additional points for component {k} ('{name_k}'), direction {direction_vector}.")
            else:
                # If no sampling requested, use only the projected vertices for checks
                all_samples[k] = poly_verts_k


        except (ValueError, RuntimeError) as e: # Catch specific projection/sampling errors
            logger.error(f"CRITICAL Projection/Sampling failed for component {k} ('{name_k}') direction {direction_vector}: {e}", exc_info=True)
            # Signal the main thread and other threads about the critical error
            error_event.set()
            return # Exit this thread immediately
        except Exception as e:
             logger.error(f"CRITICAL Unexpected error during projection/sampling for component {k} ('{name_k}') direction {direction_vector}: {e}", exc_info=True)
             error_event.set()
             return # Exit this thread immediately

    # --- 3. Perform 2D Interpolation Checks ---
    # Proceed only if no critical error occurred during projection/sampling phase
    if error_event.is_set():
        logger.warning(f"Skipping interpolation checks for direction {direction_vector} due to earlier signaled error.")
        return

    logger.info(f"Performing 2D interpolation checks ({num_successful}x{num_successful})...")
    # Initialize matrix with -1 (or another indicator) for pairs that couldn't be checked
    interpolation_matrix = np.full((num_successful, num_successful), -1, dtype=int)

    try:
        for i in range(num_successful):
            # Skip row if component i's projection was invalid or sampling failed
            if not projection_valid[i] or all_samples[i] is None:
                # Mark entire row as unchecked (-1)
                interpolation_matrix[i, :] = -1
                continue

            samples_i = all_samples[i] # These are the points to check (vertices + sampled)

            for j in range(num_successful):
                if i == j:
                    interpolation_matrix[i, j] = 0 # No interpolation with self
                    continue

                # Skip column if component j's projection was invalid
                if not projection_valid[j] or all_projections[j] is None:
                    interpolation_matrix[i, j] = -1 # Cannot check this pair
                    continue

                poly_verts_j = all_projections[j] # This is the polygon to check against

                # Perform the check: are any points from i inside polygon j?
                result = interpolator_2d.check_2d_interpolation(samples_i, poly_verts_j)
                interpolation_matrix[i, j] = result

    except ImportError as e:
         logger.error(f"CRITICAL Import error during 2D interpolation check (missing dependency?): {e}")
         error_event.set()
         return
    except Exception as e:
         logger.error(f"CRITICAL Unexpected error during 2D interpolation checks for direction {direction_vector}: {e}", exc_info=True)
         error_event.set()
         return

    # --- 4. Store Result if Successful ---
    # Check event again before storing, as interpolation check might have failed critically
    if not error_event.is_set():
        try:
            # Sanitize sheet name more robustly
            # Format numbers, replace invalid chars, limit length
            dir_str = '_'.join(map(lambda x: f"{x:.3f}".replace('.', 'p').replace('-', 'm'), direction_vector))
            sheet_name = f"Dir_{dir_str}"
            # Replace any remaining potentially invalid Excel sheet chars
            invalid_chars = r'[]:*?/\\'
            for char in invalid_chars:
                sheet_name = sheet_name.replace(char, '_')
            # Enforce Excel sheet name length limit (31 chars)
            sheet_name = sheet_name[:31]

            # Put result into the thread-safe queue
            results_queue.put((sheet_name, interpolation_matrix))
            logger.info(f"Successfully processed direction {direction_vector}. Results queued for sheet '{sheet_name}'.")

        except Exception as e:
            logger.error(f"Failed to sanitize sheet name or queue results for direction {direction_vector}: {e}", exc_info=True)
            # Signal error as saving might be compromised if queueing fails
            error_event.set()


# --- Main Execution Function ---
def run_analysis(step_file_path: str, config_file_path: str):
    """Runs the complete CAD collision and interpolation analysis."""
    start_time = datetime.datetime.now()
    logger.info("========================================")
    logger.info("Starting CAD Collision Analyzer")
    logger.info(f"Timestamp: {start_time.isoformat()}")
    logger.info("========================================")

    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {config_file_path}")
    try:
        config = config_reader.load_config(config_file_path)
        num_samples: int = config.get("num_samples", config_reader.DEFAULT_NUM_SAMPLES)
        projection_directions: List[List[float]] = config.get("directions", config_reader.DEFAULT_DIRECTIONS)
        contact_tolerance: float = config.get("contact_tolerance", 0.0001)
        linear_deflection: float = config.get("linear_deflection", 0.1)
        angular_deflection: float = config.get("angular_deflection", 0.5)
        logger.info("Configuration loaded successfully.")
        logger.info(f"  Parameters: num_samples={num_samples}, contact_tolerance={contact_tolerance}, "
                    f"linear_deflection={linear_deflection}, angular_deflection={angular_deflection}")
        logger.info(f"  Projection Directions: {len(projection_directions)}")
    except SystemExit as e:
        logger.error("Exiting due to configuration error. See previous messages.")
        sys.exit(e.code) # Propagate exit code from config_reader
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading configuration: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. Read CAD File ---
    logger.info(f"Reading CAD file: {step_file_path}")
    try:
        # Returns List[Tuple[str, TopoDS_Shape]]
        components_data: List[Tuple[str, TopoDS_Shape]] = cad_reader.read_step_file(step_file_path)
    except FileNotFoundError:
        logger.error(f"Input STEP file not found: {step_file_path}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Import error during CAD reading (pythonocc-core installed and accessible?): {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to read or process STEP file '{step_file_path}'. Error: {e}", exc_info=True)
        sys.exit(1)

    if not components_data:
        logger.warning("No components found or extracted from the STEP file. Exiting.")
        sys.exit(0) # Graceful exit if no components
    num_components_read = len(components_data)
    logger.info(f"Successfully read {num_components_read} top-level shapes/components.")

    # --- 3. Prepare Output Excel File ---
    try:
        base_name = os.path.basename(step_file_path)
        name_part = os.path.splitext(base_name)[0]
        output_filename = f"output_{name_part}_{start_time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        logger.info(f"Output will be saved to: {output_filename}")
        writer = excel_writer.ExcelWriter(output_filename)
    except ImportError as e:
        logger.error(f"Failed to initialize ExcelWriter (openpyxl installed?): {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to prepare output file setup: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. Mesh Conversion ---
    logger.info("Starting mesh conversion for all components...")
    successful_components: List[Dict[str, Any]] = [] # List[Dict] -> {name, shape, mesh}
    failed_mesh_components: List[str] = []

    for idx, (name, shape) in enumerate(components_data):
        logger.info(f"  Meshing component {idx+1}/{num_components_read}: '{name}'...")
        try:
            mesh: trimesh.Trimesh = mesh_converter.convert_shape_to_mesh(
                shape,
                component_name=name,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection
            )
            successful_components.append({"name": name, "shape": shape, "mesh": mesh})
            logger.info(f"  Successfully meshed component '{name}'.")
        except MeshConversionError as e:
            # Log the specific error from mesh_converter (already logged there too)
            logger.error(f"Mesh conversion failed for component {idx+1} ('{name}'). See mesh_generation_errors.log. Skipping this component.")
            failed_mesh_components.append(name)
            # Decide whether to halt or continue: continue for now
            # print(f"\nFATAL ERROR: Mesh conversion failed for component {idx} ('{name}'). Check mesh_generation_errors.log. Exiting.", file=sys.stderr)
            # sys.exit(1) # Uncomment to make meshing errors fatal
        except ValueError as e: # Catch potential issues with shape data itself
            logger.warning(f"Skipping component {idx+1} ('{name}') due to invalid input shape data: {e}")
            failed_mesh_components.append(name)
        except ImportError as e:
             logger.error(f"Import error during mesh conversion (trimesh or numpy missing?): {e}")
             sys.exit(1) # Likely unrecoverable installation issue
        except Exception as e:
             logger.error(f"Unexpected error during mesh conversion for component {idx+1} ('{name}'): {e}", exc_info=True)
             # Decide whether to halt or continue: continue for now
             failed_mesh_components.append(name)
             # print(f"FATAL ERROR: Unexpected error during mesh conversion for '{name}'. Check logs. Exiting.", file=sys.stderr)
             # sys.exit(1) # Uncomment to make unexpected meshing errors fatal

    if not successful_components:
        logger.error("No components could be successfully meshed. Cannot proceed. Exiting.")
        print("\nFATAL ERROR: No components were successfully meshed. Analysis cannot continue.", file=sys.stderr)
        sys.exit(1)

    component_names = [comp["name"] for comp in successful_components]
    num_successful = len(successful_components)
    logger.info(f"Successfully meshed {num_successful} out of {num_components_read} components.")
    if failed_mesh_components:
        logger.warning(f"Failed to mesh {len(failed_mesh_components)} components: {', '.join(failed_mesh_components)}")

    # --- 5. Write Initial Excel Sheets ---
    logger.info("Writing metadata and component list to Excel...")
    try:
        metadata = {
            "Input File": step_file_path,
            "Analysis Timestamp": start_time.isoformat(),
            "Components Read": num_components_read,
            "Components Successfully Meshed": num_successful,
            "Failed Mesh Components": ", ".join(failed_mesh_components) if failed_mesh_components else "None",
            "Config File Used": config_file_path,
            "Sampling Parameter (num_samples)": num_samples,
            "3D Contact Tolerance": contact_tolerance,
            "Meshing Linear Deflection": linear_deflection,
            "Meshing Angular Deflection": angular_deflection,
            "Projection Directions Used": len(projection_directions),
            "Units": "Assumed consistent units from STEP file" # Placeholder
        }
        writer.add_metadata_sheet(metadata)
        writer.add_component_names_sheet(component_names) # Only includes successfully meshed components
    except Exception as e:
         logger.error(f"Failed during initial Excel sheet writing: {e}", exc_info=True)
         # Consider exiting if metadata fails, as output might be confusing
         sys.exit(1)

    # --- 6. 3D Collision Check ---
    logger.info("Performing 3D collision checks between successfully meshed components...")
    contact_matrix = np.zeros((num_successful, num_successful), dtype=int)
    try:
        checked_pairs = 0
        total_pairs = num_successful * (num_successful - 1) // 2
        for i in range(num_successful):
            for j in range(i + 1, num_successful): # Avoid self-collision and duplicates
                comp_i = successful_components[i]
                comp_j = successful_components[j]
                # Use original shapes for potentially more accurate collision check
                result = collision_detector_3d.check_3d_collision(
                    comp_i["shape"], comp_j["shape"], # Pass original TopoDS_Shapes
                    comp_i["name"], comp_j["name"],   # Pass names for logging within the function
                    tolerance=contact_tolerance
                )
                contact_matrix[i, j] = result
                contact_matrix[j, i] = result # Matrix is symmetric
                checked_pairs += 1
                # Log progress periodically
                if checked_pairs % 100 == 0 or checked_pairs == total_pairs:
                    logger.info(f"  Checked 3D collision pairs: {checked_pairs}/{total_pairs}")
        logger.info("3D collision checks completed.")
    except ImportError as e:
        logger.error(f"Import error during 3D collision check (pythonocc-core issue?): {e}")
        sys.exit(1)
    except Exception as e:
         logger.error(f"Unexpected error during 3D collision checks: {e}", exc_info=True)
         # Allow continuing to 2D analysis, but log the failure
         print("\nWARNING: 3D Collision checks failed unexpectedly. Contact matrix may be incomplete or inaccurate.", file=sys.stderr)


    # --- 7. Write 3D Collision Matrix ---
    logger.info("Writing 3D contact matrix to Excel...")
    try:
        writer.add_matrix_sheet("3D Contact Matrix", contact_matrix, component_names)
    except Exception as e:
        logger.error(f"Failed writing 3D contact matrix sheet: {e}", exc_info=True)
        # Continue to 2D analysis even if writing fails

    # --- 8. Parallel 2D Analysis (Projection, Sampling, Interpolation) ---
    logger.info(f"Starting parallel 2D analysis for {len(projection_directions)} directions...")
    results_queue = queue.Queue()       # Queue for results from threads
    projection_error_occurred = threading.Event() # Event to signal critical errors
    threads = []

    if not projection_directions:
        logger.warning("No projection directions specified in config. Skipping 2D analysis.")
    else:
        for direction_vector in projection_directions:
            # Create and start a thread for each direction
            thread = threading.Thread(
                target=process_direction_worker,
                name=f"Dir-{direction_vector}", # Meaningful thread name
                args=(
                    direction_vector,
                    successful_components, # Pass list of dicts {name, mesh}
                    component_names,       # Pass names list
                    num_successful,
                    num_samples,
                    results_queue,         # Shared queue
                    projection_error_occurred # Shared event
                ),
                daemon=True # Allows main thread to exit even if daemon threads are running (though we join)
            )
            threads.append(thread)
            thread.start()

        # Wait for all direction processing threads to complete
        logger.info(f"Waiting for {len(threads)} direction processing threads to finish...")
        for i, thread in enumerate(threads):
            thread.join() # Wait for this thread to finish
            logger.debug(f"Thread {thread.name} finished.")

        logger.info("All direction processing threads have completed.")

    # --- 9. Collect and Write 2D Interpolation Results ---
    interpolation_results: Dict[str, np.ndarray] = {}
    while not results_queue.empty():
        try:
            sheet_name, matrix = results_queue.get_nowait()
            interpolation_results[sheet_name] = matrix
        except queue.Empty:
            break # Should not happen with the check, but good practice
        except Exception as e:
            logger.error(f"Error retrieving result from queue: {e}", exc_info=True)
            projection_error_occurred.set() # Signal error if queue retrieval fails

    if projection_error_occurred.is_set():
        logger.error("CRITICAL errors occurred during 2D projection/sampling/interpolation in one or more threads. "
                     "Skipping writing of potentially incomplete/incorrect 2D interpolation results.")
        print("\nWARNING: Errors occurred during 2D analysis. 2D interpolation results will NOT be saved to Excel. Check logs for details.", file=sys.stderr)
    elif not interpolation_results:
         logger.warning("No 2D interpolation results were generated or collected (check config directions and logs).")
    else:
        logger.info(f"Writing {len(interpolation_results)} 2D interpolation matrices to Excel...")
        try:
            # Sort results by sheet name for consistent order in Excel file
            for sheet_name, matrix in sorted(interpolation_results.items()):
                # Add placeholder name for the matrix type
                descriptive_sheet_name = f"2D Interpolation {sheet_name}"[:31] # Ensure length limit
                writer.add_matrix_sheet(descriptive_sheet_name, matrix, component_names)
            logger.info("Finished writing 2D interpolation results.")
        except Exception as e:
             logger.error(f"Failed writing 2D interpolation results to Excel: {e}", exc_info=True)
             print("\nWARNING: Failed to write 2D interpolation results to Excel. File might be incomplete.", file=sys.stderr)
             # Continue to save attempt

    # --- 10. Save Excel File ---
    logger.info("Attempting to save final Excel file...")
    try:
        writer.save()
        logger.info(f"Successfully saved analysis results to {output_filename}")
        print(f"\nAnalysis complete. Results saved to: {output_filename}")
    except Exception as e:
        logger.error(f"CRITICAL error saving Excel file '{output_filename}': {e}", exc_info=True)
        print(f"\nFATAL ERROR: Could not save Excel file '{output_filename}'. Check permissions and logs.", file=sys.stderr)
        # No sys.exit here, analysis is done, just saving failed.

    # --- 11. Log Completion ---
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logger.info("========================================")
    logger.info(f"Analysis finished. Total execution time: {duration}")
    if projection_error_occurred.is_set():
         logger.warning("NOTE: 2D interpolation results were OMITTED from the Excel file due to processing errors.")
    elif not interpolation_results and projection_directions:
         logger.warning("NOTE: No 2D interpolation results were generated.")
    if failed_mesh_components:
        logger.warning(f"NOTE: {len(failed_mesh_components)} components failed meshing and were excluded from analysis.")
    logger.info("========================================")

# --- Entry Point ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Analyzes STEP files for 3D component collisions and 2D projected interpolations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )
    parser.add_argument(
        "step_file",
        help="Path to the input STEP file (.step or .stp)"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to the JSON configuration file."
    )
    parser.add_argument(
        "-log", "--logfile",
        default=None, # No log file by default
        help="Path to optional log file. If provided, logs will be written here in addition to the console."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG level logging to console and file (if specified)."
    )

    args = parser.parse_args()

    # --- Configure Logging Level and File Handler based on args ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level) # Set level on the main logger

    # Adjust console handler level if verbose
    if args.verbose:
        stream_handler.setLevel(logging.DEBUG)
    else:
        stream_handler.setLevel(logging.INFO) # Ensure console isn't flooded unless verbose

    # Add file handler if specified
    if args.logfile:
        try:
            file_handler = logging.FileHandler(args.logfile, mode='w') # Overwrite log
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(log_level) # Set level based on verbose flag
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {args.logfile}")
        except Exception as e:
            logger.error(f"Failed to configure log file handler for {args.logfile}: {e}", exc_info=True)
            # Continue without file logging

    # --- Run the main analysis ---
    try:
        run_analysis(args.step_file, args.config)
    except SystemExit as e:
        # Catch SystemExit calls (e.g., from config loading or early exits)
        logger.info(f"Analysis terminated with exit code {e.code}.")
        sys.exit(e.code)
    except Exception as e:
        # Catch any unexpected top-level errors
        logger.error(f"An unexpected critical error occurred during analysis: {e}", exc_info=True)
        print(f"\nFATAL ERROR: An unexpected error stopped the analysis. Check logs.", file=sys.stderr)
        sys.exit(1)
