# cad_collision_analyzer/mesh_converter.py

import logging
import numpy as np
from typing import Optional

# Attempt to import required modules using the correct OCC namespace
try:
    import trimesh
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh # type: ignore
    # Import specific classes needed
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face # type: ignore
    from OCC.Core.TopExp import TopExp_Explorer # type: ignore
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_ShapeEnum # Import shape types
    from OCC.Core.BRep import BRep_Tool # type: ignore
    from OCC.Core.Poly import Poly_Triangulation # type: ignore
    from OCC.Core.gp import gp_Pnt # type: ignore
    # For checking triangulation validity
    from OCC.Core.BRepTools import breptools_Clean # type: ignore

except ImportError as e:
    logging.getLogger("CADAnalyzer.mesh_converter").critical(f"Failed to import OCC.* or trimesh modules: {e}. Ensure they are installed correctly.")
    raise ImportError(f"Import failed in mesh_converter: {e}") from e

# Module-specific logger
logger = logging.getLogger("CADAnalyzer.mesh_converter")

# Configure logging for mesh generation errors to a separate file
mesh_error_logger = logging.getLogger("MeshGenerationErrors")
if not mesh_error_logger.handlers: # Avoid adding handlers multiple times
    mesh_error_handler = logging.FileHandler('mesh_generation_errors.log', mode='a') # Append errors
    mesh_error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - Component: %(component)s - %(message)s')
    mesh_error_handler.setFormatter(mesh_error_formatter)
    mesh_error_logger.addHandler(mesh_error_handler)
    mesh_error_logger.setLevel(logging.ERROR) # Log only errors to this file
    mesh_error_logger.propagate = False # Don't send mesh errors to the main CADAnalyzer logger stream/file


class MeshConversionError(Exception):
    """Custom exception raised when mesh conversion fails."""
    pass


# --- Refined version with IncrementalMesh called once ---

def convert_shape_to_mesh_optimized(
    shape: TopoDS_Shape,
    component_name: str = "unknown",
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5
) -> trimesh.Trimesh:
    """
    Converts a TopoDS_Shape to a single trimesh.Trimesh object. (Optimized)

    Runs BRepMesh_IncrementalMesh once on the entire shape, then extracts
    triangulation data from each face and combines them. This is generally
    more efficient than running IncrementalMesh per face.

    Args:
        shape: The CAD component geometry (TopoDS_Shape).
        component_name: Name of the component (for logging). Defaults to "unknown".
        linear_deflection: Controls the accuracy of the mesh generation. Defaults to 0.1.
        angular_deflection: Controls the angular deviation for curved surfaces. Defaults to 0.5.

    Returns:
        A trimesh.Trimesh object representing the combined mesh of all faces.

    Raises:
        MeshConversionError: If meshing fails or the shape is invalid.
        ValueError: If the input shape is Null.
    """
    if shape.IsNull():
        msg = f"Input shape for component '{component_name}' is Null."
        mesh_error_logger.error(msg, extra={'component': component_name})
        raise ValueError(msg)

    # --- Perform meshing on the entire shape once ---
    # This pre-computes the triangulation data needed later
    try:
        logger.debug(f"Running BRepMesh_IncrementalMesh for shape '{component_name}' (Deflections: lin={linear_deflection}, ang={angular_deflection})...")
        mesh_maker = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        mesh_maker.Perform()

        if not mesh_maker.IsDone():
            msg = f"BRepMesh_IncrementalMesh failed for the entire shape of component '{component_name}'. Check geometry or adjust deflection parameters."
            mesh_error_logger.error(msg, extra={'component': component_name})
            raise MeshConversionError(msg)
        logger.debug(f"BRepMesh_IncrementalMesh completed successfully for shape '{component_name}'.")

    except Exception as e:
         msg = f"Exception during BRepMesh_IncrementalMesh execution for component '{component_name}': {e}"
         mesh_error_logger.error(msg, extra={'component': component_name}, exc_info=True)
         raise MeshConversionError(msg) from e


    # --- Extract and combine triangulations from faces ---
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    # Explore only for faces explicitly
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    faces_processed_count = 0
    faces_failed_triangulation = 0

    while explorer.More():
        # Get the current shape from explorer. Since we initialized with TopAbs_FACE,
        # this should be a TopoDS_Face.
        current_face = explorer.Current()
        face_index += 1

        if current_face.IsNull():
            logger.warning(f"Component '{component_name}': Explorer returned a Null shape at index {face_index}. Skipping.")
            explorer.Next()
            continue

        # Use current_face directly, assuming it's already a TopoDS_Face
        # Add an assertion for safety during debugging
        assert isinstance(current_face, TopoDS_Face), \
            f"Explorer returned non-Face shape (Type: {type(current_face)}) when exploring for TopAbs_FACE."

        topo_face = current_face # Use the shape directly

        try:
            # Attempt to get triangulation for this face
            logger.debug(f"Component '{component_name}': Processing face {face_index}. Attempting triangulation...")
            face_location = topo_face.Location() # Get location

            # Get the triangulation data associated with this face
            logger.debug(f"Calling BRep_Tool.Triangulation for face {face_index}...")
            triangulation: Optional[Poly_Triangulation] = BRep_Tool.Triangulation(topo_face, face_location) # type: ignore
            logger.debug(f"BRep_Tool.Triangulation call completed for face {face_index}.")

            # *** CORRECTED Check: Rely on None check ***
            if triangulation is None:
                logger.warning(f"Component '{component_name}': BRep_Tool.Triangulation returned None for face {face_index}. Skipping this face.")
                faces_failed_triangulation += 1
                explorer.Next()
                continue

            # Check if triangulation has nodes and triangles
            num_nodes = triangulation.NbNodes()
            num_triangles = triangulation.NbTriangles()

            if num_nodes == 0 or num_triangles == 0:
                logger.debug(f"Component '{component_name}': Face {face_index} triangulation has no nodes ({num_nodes}) or triangles ({num_triangles}). Skipping.")
                explorer.Next()
                continue

            logger.debug(f"Component '{component_name}': Face {face_index} - Nodes: {num_nodes}, Triangles: {num_triangles}")

            # Extract nodes (vertices)
            nodes = triangulation.Nodes()
            vertices = np.array([
                (nodes.Value(i).X(), nodes.Value(i).Y(), nodes.Value(i).Z())
                for i in range(1, num_nodes + 1) # OCC indices are 1-based
            ])

            # Extract triangles (faces)
            triangles = triangulation.Triangles()
            faces = np.array([
                (triangles.Value(i).Value(1), triangles.Value(i).Value(2), triangles.Value(i).Value(3))
                for i in range(1, num_triangles + 1) # OCC indices are 1-based
            ]) - 1 # Convert to 0-based indices
            faces += vertex_offset # Add offset for combining with previous faces

            all_vertices.append(vertices)
            all_faces.append(faces)
            vertex_offset += num_nodes
            faces_processed_count += 1

        except AssertionError as ae:
             # Catch the assertion error if the explorer yields unexpected types
             msg = f"Assertion failed processing face {face_index} of component '{component_name}': {ae}"
             logger.error(msg, exc_info=True)
             mesh_error_logger.error(msg, extra={'component': component_name}, exc_info=True)
             faces_failed_triangulation += 1
             # Continue trying other faces? Or make it fatal? Let's continue for now.
        except Exception as e:
            # Catch errors during triangulation extraction for a specific face
            msg = f"Error processing face {face_index} of component '{component_name}': {e}"
            logger.error(msg, exc_info=True)
            # Log details about the face that failed
            face_details = f"Face Type: {type(topo_face)}, IsNull: {topo_face.IsNull() if topo_face else 'N/A'}" # IsNull might fail here too if topo_face is invalid
            mesh_error_logger.error(f"{msg} - Face Details: {face_details}", extra={'component': component_name}, exc_info=True)
            faces_failed_triangulation += 1
            # Continue trying other faces

        explorer.Next()
    # --- End face loop ---

    logger.info(f"Component '{component_name}': Processed {faces_processed_count} faces successfully. Failed to get triangulation for {faces_failed_triangulation} faces.")

    if not all_vertices:
        msg = f"No valid triangulations extracted for any face of component '{component_name}'. Cannot create mesh."
        mesh_error_logger.error(msg, extra={'component': component_name})
        raise MeshConversionError(msg)

    if not all_vertices or not all_faces:
         msg = f"Vertex or face list empty before stacking for component '{component_name}'. Cannot create mesh."
         mesh_error_logger.error(msg, extra={'component': component_name})
         raise MeshConversionError(msg)


    final_vertices = np.vstack(all_vertices)
    final_faces = np.vstack(all_faces)

    if final_vertices.shape[0] == 0 or final_faces.shape[0] == 0:
         msg = f"Resulting combined mesh for component '{component_name}' has no vertices or faces after extraction."
         mesh_error_logger.error(msg, extra={'component': component_name})
         raise MeshConversionError(msg)

    logger.debug(f"Component '{component_name}': Combined mesh data - Vertices: {final_vertices.shape[0]}, Faces: {final_faces.shape[0]}. Creating Trimesh object...")

    # --- Create Trimesh object ---
    try:
        mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces, process=True)

        if not mesh.is_volume:
             logger.warning(f"Mesh for component '{component_name}' has no volume (Volume: {mesh.volume}).")
        if not mesh.is_watertight:
            logger.warning(f"Mesh for component '{component_name}' is not watertight.")
        if not mesh.volume > 1e-9:
             logger.warning(f"Mesh for component '{component_name}' has near-zero or negative volume ({mesh.volume}).")

        logger.info(f"Successfully created combined trimesh for component '{component_name}' ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces after processing).")
        return mesh

    except Exception as e:
        msg = f"Failed to create or process Trimesh object for component '{component_name}': {e}"
        mesh_error_logger.error(msg, extra={'component': component_name}, exc_info=True)
        raise MeshConversionError(msg) from e

# Use the optimized version by default
convert_shape_to_mesh = convert_shape_to_mesh_optimized
