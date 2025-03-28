# cad_collision_analyzer/main.py

import argparse
import os
import sys
import datetime
import logging
import numpy as np
import threading
import time # For potential sleep/debug

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
except ImportError as e:
    print(f"Error: Failed to import project modules. Ensure the package structure is correct and accessible. Details: {e}", file=sys.stderr)
    # Attempt relative imports if run as script within package for testing
    try:
        import config_reader
        import cad_reader
        import mesh_converter
        import collision_detector_3d
        import projector
        import sampler
        import interpolator_2d
        import excel_writer
        from mesh_converter import MeshConversionError
        print("Note: Using relative imports.", file=sys.stderr)
    except ImportError:
        sys.exit(1) # Exit if imports fail either way


# --- Basic Logging Configuration ---
# Configure logging early to capture messages from all modules
# Using a basic StreamHandler should be mostly thread-safe for messages,
# though order might be interleaved. FileHandler might require locks if strict ordering
# or atomicity per message is needed, but often works okay.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s [%(threadName)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
        # Consider adding a FileHandler here as well if persistent logs are desired
        # logging.FileHandler("analyzer.log")
    ]
)

# --- Worker Function for Parallel Direction Processing ---

def process_direction(
    direction_vector: list,
    successful_components: list, # Contains dicts {name, mesh}
    component_names: list,
    num_successful: int,
    num_samples: int,
    results_dict: dict, # Shared dictionary to store results {sheet_name: matrix}
    error_event: threading.Event # Shared event to signal critical errors
):
    """
    Worker function executed by each thread to process one projection direction.
    Performs projection, sampling, and interpolation checks.
    Stores results in results_dict if successful, sets error_event on failure.
    """
    thread_name = threading.current_thread().name
    logging.info(f"Starting processing for direction: {direction_vector}")

    # --- Normalize Direction ---
    try:
        dir_vec = np.array(direction_vector, dtype=np.float64)
        norm = np.linalg.norm(dir_vec)
        if np.isclose(norm, 0.0):
            logging.warning(f"[{thread_name}] Skipping zero vector direction: {direction_vector}")
            return # Exit thread for this direction
        direction_norm = dir_vec / norm
    except Exception as e:
        logging.warning(f"[{thread_name}] Could not normalize direction {direction_vector}: {e}. Skipping.")
        return # Exit thread for this direction

    # --- Pre-calculate Projections and Samples ---
    logging.info(f"[{thread_name}] Generating projections and samples...")
    all_projections_this_direction = [None] * num_successful
    all_samples_this_direction = [None] * num_successful
    projection_valid_flags = [False] * num_successful

    for k in range(num_successful):
        # Check if a critical error was signaled by another thread early exit
        if error_event.is_set():
             logging.warning(f"[{thread_name}] Halting pre-calculation for direction {direction_vector} due to error signaled by another thread.")
             return # Exit thread

        comp_k = successful_components[k]
        mesh_k = comp_k["mesh"]
        name_k = comp_k["name"]
        logging.debug(f"[{thread_name}]   Processing component {k} ('{name_k}')...")

        try:
            center_k = projector.calculate_geometric_center(mesh_k)
            poly_verts_k = projector.project_mesh_onto_plane(mesh_k, direction_norm, center_k)

            if poly_verts_k.shape[0] < 3:
                logging.warning(f"[{thread_name}]   Component {k} ('{name_k}') projection has < 3 vertices for direction {direction_vector}. Skipping interpolation checks involving this component.")
                continue # Keep flag as False, proceed to next component

            projection_valid_flags[k] = True
            all_projections_this_direction[k] = poly_verts_k

            # Sample points only if projection is valid
            samples_k = sampler.sample_polygon_points(poly_verts_k, num_samples)
            # Check if sampler returned only original vertices (sampling might have failed)
            if samples_k.shape[0] <= poly_verts_k.shape[0] and num_samples > 0:
                 logging.warning(f"[{thread_name}]   Sampler generated {samples_k.shape[0] - poly_verts_k.shape[0]} / {num_samples} points for component {k} ('{name_k}'), direction {direction_vector}.")
                 # Continue even if sampling didn't hit target number
            all_samples_this_direction[k] = samples_k

        except (ValueError, RuntimeError, ImportError) as e: # Catch projection/sampling errors
            logging.error(f"[{thread_name}] CRITICAL Projection/Sampling failed for component {k} ('{name_k}') direction {direction_vector}: {e}", exc_info=True)
            # Signal the main thread about the critical error
            error_event.set()
            return # Exit this thread immediately
        except Exception as e:
             logging.error(f"[{thread_name}] CRITICAL Unexpected error during projection/sampling for component {k} ('{name_k}') direction {direction_vector}: {e}", exc_info=True)
             error_event.set()
             return # Exit this thread immediately

    # --- Perform 2D Interpolation Checks ---
    # Proceed only if no critical error occurred during projection/sampling for this direction
    if error_event.is_set():
        logging.warning(f"[{thread_name}] Skipping interpolation checks for direction {direction_vector} due to earlier signaled error.")
        return

    logging.info(f"[{thread_name}] Performing 2D interpolation checks ({num_successful}x{num_successful})...")
    interpolation_matrix = np.zeros((num_successful, num_successful), dtype=int)
    try:
        for i in range(num_successful):
            if not projection_valid_flags[i]: continue # Skip row if i's projection failed
            samples_i = all_samples_this_direction[i]
            if samples_i is None: continue # Should not happen if flag is True

            for j in range(num_successful):
                if i == j: continue
                if not projection_valid_flags[j]: continue # Skip column if j's projection failed
                poly_verts_j = all_projections_this_direction[j]
                if poly_verts_j is None: continue

                # Perform the check
                result = interpolator_2d.check_2d_interpolation(samples_i, poly_verts_j)
                interpolation_matrix[i, j] = result

    except ImportError as e:
         logging.error(f"[{thread_name}] CRITICAL Import error during 2D interpolation check: {e}")
         error_event.set()
         return
    except Exception as e:
         logging.error(f"[{thread_name}] CRITICAL Unexpected error during 2D interpolation checks for direction {direction_vector}: {e}", exc_info=True)
         error_event.set()
         return

    # --- Store Result if Successful ---
    # Check event again before storing, as interpolation check might have failed
    if not error_event.is_set():
        try:
            # Sanitize sheet name more robustly
            dir_str = '_'.join(map(lambda x: f"{x:.3f}".replace('.', 'p').replace('-', 'm'), direction_vector))
            sheet_name = f"Dir_{dir_str}"
            # Replace any remaining potentially invalid chars (conservative approach)
            invalid_chars = r'[]:*?/\\'
            for char in invalid_chars:
                sheet_name = sheet_name.replace(char, '_')
            sheet_name = sheet_name[:31] # Enforce Excel sheet name length limit

            # Store result in the shared dictionary (atomic operation in CPython)
            results_dict[sheet_name] = interpolation_matrix
            logging.info(f"[{thread_name}] Successfully processed direction {direction_vector}. Results stored for sheet '{sheet_name}'.")
        except Exception as e:
            logging.error(f"[{thread_name}] Failed to sanitize sheet name or store results for direction {direction_vector}: {e}", exc_info=True)
            # Signal error as saving might be compromised
            error_event.set()


# --- Main Execution ---
def main():
    """Main function to orchestrate the CAD collision analysis process."""
    start_time = datetime.datetime.now()
    # Setup root logger - basicConfig does this if not already configured
    logging.info("========================================")
    logging.info("Starting CAD Collision Analyzer")
    logging.info(f"Timestamp: {start_time.isoformat()}")
    logging.info("========================================")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Analyzes STEP files for component collisions and interpolations.")
    parser.add_argument("step_file", help="Path to the input STEP file (.step or .stp)")
    parser.add_argument("-c", "--config", default="config.json", help="Path to configuration JSON file (default: config.json)")
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    logging.info(f"Loading configuration from: {args.config}")
    try:
        config = config_reader.load_config(args.config)
        num_samples = config.get("num_samples", config_reader.DEFAULT_NUM_SAMPLES)
        projection_directions = config.get("directions", config_reader.DEFAULT_DIRECTIONS)
        contact_tolerance = config.get("contact_tolerance", 0.0001)
        linear_deflection = config.get("linear_deflection", 0.1)
        angular_deflection = config.get("angular_deflection", 0.5)
        logging.info("Configuration loaded successfully.")
        logging.info(f"  Parameters: num_samples={num_samples}, contact_tolerance={contact_tolerance}, linear_deflection={linear_deflection}, angular_deflection={angular_deflection}")
        logging.info(f"  Projection Directions: {len(projection_directions)}")
    except SystemExit as e:
        logging.error("Exiting due to configuration error. See previous messages.")
        sys.exit(e.code)
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading configuration: {e}", exc_info=True)
        sys.exit(1)


    # --- 2. Read CAD File ---
    logging.info(f"Reading CAD file: {args.step_file}")
    try:
        components_data = cad_reader.read_step_file(args.step_file)
    except FileNotFoundError:
        logging.error(f"Input STEP file not found: {args.step_file}")
        sys.exit(1)
    except ImportError as e:
        logging.error(f"Import error during CAD reading (pythonocc-core installed?): {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to read or process STEP file '{args.step_file}'. Error: {e}", exc_info=True)
        sys.exit(1)

    if not components_data:
        logging.warning("No components found or extracted from the STEP file. Exiting.")
        sys.exit(0)
    num_components_read = len(components_data)
    logging.info(f"Found {num_components_read} top-level components.")

    # --- 3. Prepare Output ---
    try:
        base_name = os.path.basename(args.step_file)
        name_part = os.path.splitext(base_name)[0]
        output_filename = f"output_{name_part}.xlsx"
        logging.info(f"Output will be saved to: {output_filename}")
        writer = excel_writer.ExcelWriter(output_filename)
    except ImportError as e:
        logging.error(f"Failed to initialize ExcelWriter (openpyxl installed?): {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to prepare output file setup: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. Mesh Conversion ---
    logging.info("Starting mesh conversion for all components...")
    successful_components = [] # List[Dict] -> {name, shape, mesh}
    for idx, (name, shape) in enumerate(components_data):
        logging.info(f"  Meshing component {idx}/{num_components_read-1}: '{name}'...")
        try:
            mesh = mesh_converter.convert_shape_to_mesh(
                shape, component_name=name,
                linear_deflection=linear_deflection, angular_deflection=angular_deflection
            )
            successful_components.append({"name": name, "shape": shape, "mesh": mesh})
            # Don't log success here, mesh_converter does
        except MeshConversionError as e:
            logging.error(f"Stopping execution due to critical mesh conversion error for component '{name}'.")
            print(f"\nFATAL ERROR: Mesh conversion failed for component {idx} ('{name}'). Check mesh_generation_errors.log. Exiting.", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            logging.warning(f"Skipping component {idx} ('{name}') due to invalid input shape: {e}")
            print(f"Warning: Skipping component {idx} ('{name}') due to invalid shape.", file=sys.stderr)
        except ImportError as e:
             logging.error(f"Import error during mesh conversion (trimesh or numpy missing?): {e}")
             sys.exit(1)
        except Exception as e:
             logging.error(f"Unexpected error during mesh conversion for component {idx} ('{name}'): {e}", exc_info=True)
             print(f"FATAL ERROR: Unexpected error during mesh conversion for '{name}'. Check logs. Exiting.", file=sys.stderr)
             sys.exit(1)

    if not successful_components:
        logging.error("No components could be successfully meshed. Cannot proceed. Exiting.")
        sys.exit(1)
    component_names = [comp["name"] for comp in successful_components]
    num_successful = len(successful_components)
    logging.info(f"Successfully meshed {num_successful} out of {num_components_read} components.")

    # --- 5. Write Initial Excel Sheets ---
    logging.info("Writing metadata and component list to Excel...")
    try:
        metadata = {
            "Input File": args.step_file, "Analysis Timestamp": start_time.isoformat(),
            "Components Read": num_components_read, "Components Meshed": num_successful,
            "Config File Used": args.config, "Sampling Parameter (num_samples)": num_samples,
            "3D Contact Tolerance": contact_tolerance, "Meshing Linear Deflection": linear_deflection,
            "Meshing Angular Deflection": angular_deflection, "Units": "N/A"
        }
        writer.add_metadata_sheet(metadata)
        writer.add_component_names_sheet(component_names)
    except Exception as e:
         logging.error(f"Failed during initial Excel sheet writing: {e}", exc_info=True)
         sys.exit(1)

    # --- 6. 3D Collision Check ---
    logging.info("Performing 3D collision checks between successfully meshed components...")
    contact_matrix = np.zeros((num_successful, num_successful), dtype=int)
    try:
        checked_pairs = 0
        total_pairs = num_successful * (num_successful - 1) // 2
        for i in range(num_successful):
            for j in range(i + 1, num_successful):
                comp_i = successful_components[i]
                comp_j = successful_components[j]
                result = collision_detector_3d.check_3d_collision(
                    comp_i["shape"], comp_i["name"], comp_j["shape"], comp_j["name"],
                    tolerance=contact_tolerance
                )
                contact_matrix[i, j] = result
                contact_matrix[j, i] = result
                checked_pairs += 1
                if checked_pairs % 100 == 0 or checked_pairs == total_pairs:
                    logging.info(f"  Checked 3D pairs: {checked_pairs}/{total_pairs}")
        logging.info("3D collision checks completed.")
    except Exception as e:
         logging.error(f"Unexpected error during 3D collision checks: {e}", exc_info=True)
         sys.exit(1)

    # --- 7. Write 3D Collision Matrix ---
    logging.info("Writing contact matrix to Excel...")
    try:
        writer.add_matrix_sheet("contact matrix", contact_matrix, component_names)
    except Exception as e:
        logging.error(f"Failed writing contact matrix sheet: {e}", exc_info=True)
        # Continue...

    # --- 8. Parallel 2D Analysis Loop ---
    logging.info(f"Starting parallel 2D analysis for {len(projection_directions)} directions...")
    direction_results = {} # Shared dict for results {sheet_name: matrix}
    projection_error_occurred = threading.Event() # Shared event flag
    threads = []

    # Extract meshes once to pass to threads
    # component_meshes = [comp["mesh"] for comp in successful_components]

    for direction_vector in projection_directions:
        # Create and start a thread for each direction
        thread = threading.Thread(
            target=process_direction,
            name=f"Dir-{direction_vector}", # Give thread a meaningful name for logging
            args=(
                direction_vector,
                successful_components, # Pass the list of dicts
                component_names,
                num_successful,
                num_samples,
                direction_results,      # Shared dictionary
                projection_error_occurred # Shared event
            ),
            daemon=True # Allows main thread to exit if needed, though we join
        )
        threads.append(thread)
        thread.start()

    # Wait for all direction processing threads to complete
    logging.info(f"Waiting for {len(threads)} direction processing threads to finish...")
    for i, thread in enumerate(threads):
        thread.join() # Wait for this thread
        logging.debug(f"Thread {thread.name} finished.")

    logging.info("All direction processing threads have completed.")

    # --- 9. Write 2D Interpolation Results (Conditional) ---
    if projection_error_occurred.is_set():
        logging.error("CRITICAL errors occurred during 2D projection/sampling/interpolation in one or more threads. Skipping writing of 2D interpolation results.")
        print("\nWARNING: Errors occurred during 2D analysis. 2D interpolation results will NOT be saved.", file=sys.stderr)
    elif not direction_results:
         logging.warning("No 2D interpolation results were generated (all directions might have been skipped or invalid).")
    else:
        logging.info(f"Writing {len(direction_results)} 2D interpolation matrices to Excel...")
        try:
            # Sort results by sheet name for consistent order in Excel
            for sheet_name, matrix in sorted(direction_results.items()):
                writer.add_matrix_sheet(sheet_name, matrix, component_names)
            logging.info("Finished writing 2D interpolation results.")
        except Exception as e:
             logging.error(f"Failed writing 2D interpolation results to Excel: {e}", exc_info=True)
             # Continue to save attempt

    # --- 10. Save Excel File ---
    logging.info("Saving final Excel file...")
    writer.save()

    # --- 11. Log Completion ---
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info("========================================")
    logging.info(f"Analysis complete. Total execution time: {duration}")
    logging.info(f"Results potentially saved to {output_filename} (check logs for save/processing errors)")
    if projection_error_occurred.is_set():
         logging.warning("NOTE: 2D interpolation results were omitted due to processing errors.")
    logging.info("========================================")


if __name__ == "__main__":
    main()
