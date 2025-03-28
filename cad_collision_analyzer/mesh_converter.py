import logging
import numpy as np
from typing import Optional, Tuple

# --- PythonOCC Core Imports ---
try:
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Iterator
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.Poly import Poly_Triangulation
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.Poly import Poly_Array1OfTriangle
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TopLoc import TopLoc_Location
    PYTHONOCC_INSTALLED = True
except ImportError:
    PYTHONOCC_INSTALLED = False
    # Define dummy types for type hinting if import fails
    TopoDS_Shape = type("TopoDS_Shape", (), {"IsNull": lambda: True})
    print("WARNING: pythonocc-core not found. Mesh conversion functionality will fail.")

# --- Trimesh Import ---
try:
    import trimesh
    TRIMESH_INSTALLED = True
except ImportError:
    TRIMESH_INSTALLED = False
    # Define dummy type for type hinting
    trimesh = type("trimesh", (), {"Trimesh": type("Trimesh", (), {})})
    print("WARNING: trimesh not found. Mesh conversion functionality will fail.")


# --- Custom Exception ---
class MeshConversionError(Exception):
    """Custom exception for errors during CAD shape to mesh conversion."""
    pass

# --- Logging Setup ---
MESH_LOG_FILE = 'mesh_generation_errors.log'
mesh_logger = logging.getLogger(__name__)
mesh_logger.setLevel(logging.INFO) # Process messages at INFO level and above

# Prevent adding handlers multiple times
if not mesh_logger.handlers:
    # File handler for errors related to mesh generation
    try:
        fh = logging.FileHandler(MESH_LOG_FILE, mode='a') # Append mode
        fh.setLevel(logging.ERROR)
        # Include component name in the log format using a Filter
        log_format = '%(asctime)s - %(levelname)s - Component: %(component)s - %(message)s'
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)

        class ContextFilter(logging.Filter):
            """Injects context (like component name) into the LogRecord."""
            def filter(self, record):
                record.component = getattr(record, 'component', 'Unknown')
                return True

        # Add filter only to this handler, not the logger itself
        fh.addFilter(ContextFilter())
        mesh_logger.addHandler(fh)
    except Exception as e:
        # Fallback if file handler can't be created
        logging.basicConfig(level=logging.ERROR)
        mesh_logger = logging.getLogger(__name__) # Re-get logger
        mesh_logger.error(f"Could not configure file logging to {MESH_LOG_FILE}. Error: {e}", exc_info=True)


def convert_shape_to_mesh(
    shape: TopoDS_Shape,
    component_name: str = "Unknown",
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5
) -> 'trimesh.Trimesh':
    """
    Converts a TopoDS_Shape object to a trimesh.Trimesh object.

    Uses BRepMesh_IncrementalMesh to generate a mesh from the input shape,
    then extracts vertices and faces from the triangulation of each face.

    Args:
        shape: The input CAD geometry (pythonocc.Core.TopoDS.TopoDS_Shape).
        component_name: An identifier for the component (used in logging).
        linear_deflection: Linear deflection parameter for meshing accuracy.
                           Controls the maximum distance between a mesh edge
                           and the original geometry curve. Lower is finer.
        angular_deflection: Angular deflection parameter for meshing accuracy.
                            Controls the maximum angle between normals of
                            adjacent mesh facets on a curved surface. Lower is finer.

    Returns:
        A trimesh.Trimesh object representing the mesh.

    Raises:
        MeshConversionError: If meshing fails, no triangulation data is found,
                             or if pythonocc-core/trimesh are not installed.
                             Logs the error details to 'mesh_generation_errors.log'.
        ValueError: If the input shape is null.
    """
    if not PYTHONOCC_INSTALLED:
        raise MeshConversionError("pythonocc-core is not installed or could not be imported.")
    if not TRIMESH_INSTALLED:
        raise MeshConversionError("trimesh library is not installed.")

    log_extra = {'component': component_name} # Context for logger

    if shape.IsNull():
        err_msg = "Input TopoDS_Shape is null."
        # Log error to the specific file handler for mesh errors
        mesh_logger.error(err_msg, extra=log_extra)
        # Raise ValueError for invalid input, distinct from MeshConversionError
        raise ValueError(f"Component '{component_name}': {err_msg}")

    # --- Perform Meshing ---
    # The constructor performs the meshing. Using relative=True is common practice.
    # relative=True means deflections are relative to object size (0.1 = 10%)
    # relative=False (default) means absolute units. Let's use absolute for consistency.
    try:
        # BRepMesh_IncrementalMesh(shape, linear_deflection, is_relative=False, angular_deflection=..., parallel=...)
        mesh_util = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        # mesh_util.Perform() # Often implicit in constructor or specific OCC versions
        # Check if meshing algorithm reported success (optional check)
        # is_done = mesh_util.IsDone() # Might not be reliable indicator always
    except Exception as e:
        err_msg = f"BRepMesh_IncrementalMesh algorithm failed. Error: {e}"
        mesh_logger.error(err_msg, exc_info=True, extra=log_extra)
        raise MeshConversionError(f"Component '{component_name}': {err_msg}") from e


    # --- Extract Vertices and Faces ---
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    face_processed_count = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        # Use DownCast to get the specific TopoDS_Face type
        try:
             current_shape = explorer.Current()
             face: TopoDS_Face = TopoDS_Face().DownCast(current_shape)
        except Exception as cast_err:
             # This should ideally not happen if exploring faces, but handle defensively
             mesh_logger.warning(f"Could not downcast explored shape to TopoDS_Face. Skipping. Error: {cast_err}", extra=log_extra)
             explorer.Next()
             continue

        if face.IsNull():
             mesh_logger.warning("Encountered a null face while exploring shape, skipping.", extra=log_extra)
             explorer.Next()
             continue

        location = TopLoc_Location()
        triangulation: Optional[Poly_Triangulation] = None # Initialize
        try:
            # Get triangulation data for the face
            # BRep_Tool.Triangulation returns None if no triangulation exists
            triangulation = BRep_Tool.Triangulation(face, location)

            # Check if triangulation is valid
            if triangulation is None or triangulation.IsNull():
                # If *any* face fails triangulation, consider the whole shape meshing failed for robustness
                err_msg = f"Failed to get valid triangulation for a face (Face Index approx: {face_processed_count})."
                mesh_logger.error(err_msg, extra=log_extra)
                raise MeshConversionError(f"Component '{component_name}': {err_msg}")

            # Get transformation from the location
            trsf = location.Transformation()

            # Extract vertices for this face
            nodes: TColgp_Array1OfPnt = triangulation.Nodes()
            face_vertices_count = nodes.Length()
            if face_vertices_count == 0:
                 # Valid triangulation but no nodes? Log warning and skip face.
                 mesh_logger.warning(f"Face {face_processed_count} triangulation has 0 nodes. Skipping face.", extra=log_extra)
                 explorer.Next()
                 continue

            face_vertices = np.empty((face_vertices_count, 3), dtype=np.float64)
            for i in range(face_vertices_count):
                # Nodes are 1-based index in OCC
                pnt: gp_Pnt = nodes.Value(i + 1)
                # Apply location transformation IMPORTANT!
                pnt.Transform(trsf)
                face_vertices[i] = [pnt.X(), pnt.Y(), pnt.Z()]

            # Extract faces (triangles) for this face
            triangles: Poly_Array1OfTriangle = triangulation.Triangles()
            face_triangles_count = triangles.Length()
            if face_triangles_count == 0:
                 # Valid triangulation but no triangles? Log warning and skip face.
                 mesh_logger.warning(f"Face {face_processed_count} triangulation has 0 triangles. Skipping face.", extra=log_extra)
                 explorer.Next()
                 continue

            face_faces = np.empty((face_triangles_count, 3), dtype=np.int64)
            for i in range(face_triangles_count):
                # Triangles are 1-based index in OCC
                triangle = triangles.Value(i + 1)
                # Get vertex indices (1-based) and convert to 0-based for numpy/trimesh
                idx1, idx2, idx3 = triangle.Value(1), triangle.Value(2), triangle.Value(3)
                # Add current vertex_offset to make indices global for the shape
                face_faces[i] = [idx1 - 1 + vertex_offset,
                                 idx2 - 1 + vertex_offset,
                                 idx3 - 1 + vertex_offset]

            # Append face data to the main lists/arrays
            all_vertices.append(face_vertices)
            all_faces.append(face_faces)

            # Update the vertex offset for the next face's indices
            vertex_offset += face_vertices_count
            face_processed_count += 1

        except Exception as e:
             # Catch unexpected errors during processing of a specific face's triangulation
             err_msg = f"Error processing triangulation for face index approx {face_processed_count}. Error: {e}"
             mesh_logger.error(err_msg, exc_info=True, extra=log_extra)
             raise MeshConversionError(f"Component '{component_name}': {err_msg}") from e

        explorer.Next() # Move to the next face

    # --- Final Check and Trimesh Creation ---
    if not all_vertices or not all_faces:
        # This case occurs if shape had faces, but none yielded valid triangulation data
        err_msg = f"No mesh data (vertices or faces) could be extracted after processing {face_processed_count} faces."
        mesh_logger.error(err_msg, extra=log_extra)
        raise MeshConversionError(f"Component '{component_name}': {err_msg}")

    try:
        # Concatenate vertices and faces from all processed faces
        vertices_np = np.concatenate(all_vertices, axis=0)
        faces_np = np.concatenate(all_faces, axis=0)

        if vertices_np.shape[0] == 0 or faces_np.shape[0] == 0:
             err_msg = "Concatenated mesh data resulted in empty vertices or faces array."
             mesh_logger.error(err_msg, extra=log_extra)
             raise MeshConversionError(f"Component '{component_name}': {err_msg}")

        # Create the trimesh object
        # process=True performs basic processing like merging duplicate vertices
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=True)

        if not mesh.is_watertight:
            # Log as warning, not error, as non-watertight meshes might still be usable
            mesh_logger.warning(f"Resulting mesh is not watertight.", extra=log_extra)
        if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            # Post-processing might have removed everything if input was degenerate
            err_msg = "Mesh became empty after trimesh processing."
            mesh_logger.error(err_msg, extra=log_extra)
            raise MeshConversionError(f"Component '{component_name}': {err_msg}")


        console_logger = logging.getLogger('cad_collision_analyzer.main') # Log success to main logger
        console_logger.info(f"    -> Successfully converted shape to mesh ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces).")
        return mesh

    except Exception as e:
        # Catch errors during concatenation or Trimesh creation/processing
        err_msg = f"Failed during final mesh creation or processing using Trimesh. Error: {e}"
        mesh_logger.error(err_msg, exc_info=True, extra=log_extra)
        raise MeshConversionError(f"Component '{component_name}': {err_msg}") from e


# Example Usage (Optional)
if __name__ == '__main__':
    # Requires a valid TopoDS_Shape object.
    # This example needs the cad_reader and a test file to run.
    import sys
    import os
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(f"Mesh generation errors will be logged to: {MESH_LOG_FILE}")

    if not PYTHONOCC_INSTALLED or not TRIMESH_INSTALLED:
        print("Error: pythonocc-core and trimesh are required to run this example.", file=sys.stderr)
    else:
        try:
            # Need cad_reader to get a shape first
            from cad_collision_analyzer.cad_reader import read_step_file # Assuming relative import works
        except ImportError:
             try: # Fallback if run directly
                  from cad_reader import read_step_file
             except ImportError:
                  print("Could not import cad_reader to load a shape for the example.", file=sys.stderr)
                  sys.exit(1)

        # Example: Replace with a path to a real STEP file
        test_file = "example.step" # <<< PUT A REAL STEP FILE PATH HERE >>>

        if os.path.exists(test_file):
            print(f"Reading shape from: {test_file}")
            components = read_step_file(test_file)
            if components:
                name, shape_to_mesh = components[0] # Take the first component
                print(f"Attempting to mesh component: '{name}'")
                try:
                    mesh_result = convert_shape_to_mesh(
                        shape_to_mesh,
                        component_name=name,
                        linear_deflection=0.05, # Finer deflection for example
                        angular_deflection=0.2
                    )
                    print("Meshing successful!")
                    print(f"Trimesh object created with {len(mesh_result.vertices)} vertices and {len(mesh_result.faces)} faces.")
                    # print("Mesh bounding box:", mesh_result.bounding_box.bounds)
                    # mesh_result.show() # Requires additional dependencies (e.g., pyglet, matplotlib)
                except (MeshConversionError, ValueError) as mesh_err:
                    print(f"Meshing failed: {mesh_err}", file=sys.stderr)
                except Exception as ex:
                     print(f"An unexpected error occurred during meshing: {ex}", file=sys.stderr)
                     logging.getLogger(__name__).error("Unexpected error in mesh example", exc_info=True)

            else:
                print(f"Could not read any components from {test_file} (check cad_parsing_errors.log).")
        else:
            print(f"Test file '{test_file}' not found.", file=sys.stderr)
