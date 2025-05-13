import argparse
import os
import sys
import datetime
import numpy as np
import logging
import threading
import queue

from cad_collision_analyzer import config_reader
from cad_collision_analyzer import cad_reader
from cad_collision_analyzer import mesh_converter
from cad_collision_analyzer import collision_detector_3d
from cad_collision_analyzer import projector
from cad_collision_analyzer import sampler
from cad_collision_analyzer import interpolator_2d
from cad_collision_analyzer import excel_writer


def process_direction(direction_vector, component_meshes, component_names, num_samples, components, original_component_indices, results_queue, projection_error_occurred):
    """
    Worker function to process a single direction vector for 2D interpolation.
    """
    try:
        logging.info(f"Processing direction: {direction_vector} in thread {threading.current_thread().name}")
        direction_vector = np.array(direction_vector)
        if np.allclose(direction_vector, np.zeros(3)):
            logging.warning(f"Zero direction vector encountered in thread {threading.current_thread().name}. Skipping.")
            return

        direction_norm = direction_vector / np.linalg.norm(direction_vector)

        interpolation_matrix = np.zeros((len(component_meshes), len(component_meshes)), dtype=int)
        N = len(component_meshes)

        # Loop Through Component Pairs (i, j)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                mesh_i = component_meshes[i]
                mesh_j = component_meshes[j]

                try:
                    # Projection
                    center_i = projector.calculate_geometric_center(mesh_i)
                    polygon_vertices_i = projector.project_mesh_onto_plane(mesh_i, direction_norm, center_i)
                    polygon_vertices_j = projector.project_mesh_onto_plane(mesh_j, direction_norm, center_i)

                    if polygon_vertices_i.shape[0] < 3 or polygon_vertices_j.shape[0] < 3:
                        logging.warning(f"Skipping interpolation check for components {component_names[i]} and {component_names[j]} in direction {direction_vector} due to invalid projections.")
                        continue # Skip current component pair

                    # Sampling (if projections valid)
                    sampled_points_i = sampler.sample_polygon_points(polygon_vertices_i, num_samples)

                    # 2D Interpolation Check (if projections valid)
                    result = interpolator_2d.check_2d_interpolation(sampled_points_i, polygon_vertices_j)
                    interpolation_matrix[i, j] = result

                except RuntimeError as e:
                    logging.error(f"Projection failed for components {component_names[i]} and {component_names[j]} in direction {direction_vector}: {e}")
                    projection_error_occurred.set() # Set the global error flag.
                    return # Exit the thread's work
        # Write Interpolation Matrix Sheet
        sheet_name = f"Direction_{direction_vector[0]}_{direction_vector[1]}_{direction_vector[2]}".replace('.', '_')
        results_queue.put((sheet_name, interpolation_matrix))

    except Exception as e:
        logging.error(f"An unexpected error occurred in thread {threading.current_thread().name}: {e}")


def main():
    """
    Main function to orchestrate the CAD collision analysis process.
    """
    # Record the start time
    start_time = datetime.datetime.now()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CAD Collision Analyzer")
    parser.add_argument("step_file", help="Path to the input STEP file")
    parser.add_argument(
        "-c", "--config", default="config.json", help="Path to the configuration JSON file"
    )
    args = parser.parse_args()

    # Configure basic logging
    logging.basicConfig(
        filename='cad_analyzer.log',  # Log to a file
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Starting CAD collision analysis...")

    # Load Configuration
    try:
        config = config_reader.load_config(args.config)
        num_samples = config.get('num_samples')
        directions = config.get('directions')
    except SystemExit:
        logging.error("Invalid configuration. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Read CAD File
    try:
        components = cad_reader.read_step_file(args.step_file)
        logging.info(f"Found {len(components)} components in STEP file.")
        if not components:
            logging.warning("No components found in STEP file. Exiting.")
            sys.exit(0)  # Graceful exit
    except FileNotFoundError:
        logging.error(f"STEP file not found: {args.step_file}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Invalid STEP file format: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading the STEP file: {e}")
        sys.exit(1)


    # Prepare Output
    output_filename = f"output_{os.path.basename(args.step_file).replace('.step', '').replace('.stp', '')}.xlsx"
    writer = excel_writer.ExcelWriter(output_filename)
    component_names = []
    component_meshes = []
    original_component_indices = []

    # Mesh Conversion
    num_components = len(components)
    logging.info(f"Starting mesh conversion for {num_components} components...")

    for i, (name, shape) in enumerate(components):
        try:
            mesh = mesh_converter.convert_shape_to_mesh(shape, component_name=name)
            component_meshes.append(mesh)
            component_names.append(name)
            original_component_indices.append(i)
            logging.info(f"Successfully meshed component {name} ({i+1}/{num_components})")
        except mesh_converter.MeshConversionError as e:
            logging.error(f"Mesh conversion failed for component {name}: {e}")
            print(f"Error: Mesh conversion failed for component {name}. Halting execution.")
            sys.exit(1)  # Exit on meshing error
        except Exception as e:
            logging.error(f"An unexpected error occurred during mesh conversion: {e}")
            print(f"Error: An unexpected error occurred during mesh conversion. Halting execution.")
            sys.exit(1)

    N = len(component_meshes)  # Number of successfully meshed components
    logging.info(f"Successfully meshed {N} components.")

    # Write Initial Excel Sheets
    metadata = {
        "Input File": os.path.basename(args.step_file),
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Number of Components": N,
        "Sampling Parameter": num_samples,
        "Directions": directions,
        "Units": "N/A",  # Placeholder
        "Contact Tolerance": 0.0001,  # Hardcoded for now
    }

    writer.add_metadata_sheet(metadata)
    writer.add_component_names_sheet(component_names)

    # 3D Collision Check
    contact_matrix = np.zeros((N, N), dtype=int)
    logging.info("Performing 3D collision checks...")

    for i in range(N):
        for j in range(i + 1, N):
            shape_i = components[original_component_indices[i]][1]  # Get original shape
            shape_j = components[original_component_indices[j]][1]  # Get original shape

            contact = collision_detector_3d.check_3d_collision(
                shape_i, shape_j, tolerance=0.0001
            )
            contact_matrix[i, j] = contact
            contact_matrix[j, i] = contact

    # Write 3D Collision Matrix
    writer.add_matrix_sheet("contact matrix", contact_matrix, component_names)

    # Parallel Processing of Directions
    logging.info("Starting parallel processing of directions...")
    direction_results = queue.Queue() # Thread-safe queue for results
    projection_error_occurred = threading.Event() # Thread-safe event for projection errors
    threads = []

    for direction_vector in directions:
        thread = threading.Thread(
            target=process_direction,
            args=(direction_vector, component_meshes, component_names, num_samples, components, original_component_indices, direction_results, projection_error_occurred)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if projection_error_occurred.is_set():
        logging.error("Projection errors occurred during 2D interpolation. Halting Excel sheet creation for these.")
        print("Error: Projection errors occurred. Halting Excel sheet creation for these.")
    else:
        logging.info("All directions processed successfully. Writing interpolation matrices...")
        while not direction_results.empty():
            sheet_name, interpolation_matrix = direction_results.get()
            writer.add_matrix_sheet(sheet_name, interpolation_matrix, component_names)


    # Save Excel File
    try:
        writer.save()
        logging.info(f"Successfully wrote output to {output_filename}")
    except IOError as e:
        logging.error(f"Error saving Excel file: {e}")
        sys.exit(1)

    # Log completion message and total execution time
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    logging.info(f"CAD collision analysis completed in {total_time}.")
    print(f"CAD collision analysis completed. Output written to {output_filename}")


if __name__ == "__main__":
    main()
</content>
  </change>
  <change>
    <file>requirements.txt</file>
    <description>add queue to requirements.txt</description>
    <content><![CDATA[
pythonocc-core
numpy
openpyxl
pytest
trimesh
