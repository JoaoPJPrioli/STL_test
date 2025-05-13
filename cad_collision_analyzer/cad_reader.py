# cad_collision_analyzer/cad_reader.py

import os
import sys # Import sys to use sys.maxsize (though not needed for Tag())
import logging
from typing import List, Tuple, Optional, Set

# Attempt to import pythonocc-core modules using the correct OCC namespace
try:
    from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_StepModelType # type: ignore
    from OCC.Core.IFSelect import IFSelect_ReturnStatus # type: ignore
    from OCC.Core.TopoDS import TopoDS_Shape # type: ignore
    from OCC.Core.TDocStd import TDocStd_Document # type: ignore
    # Import XCAFDoc directly for tools
    import OCC.Core.XCAFDoc as XCAFDoc # type: ignore
    from OCC.Core.TDF import TDF_Label, TDF_LabelSequence, TDF_Attribute # type: ignore
    # Removed Handle import
    from OCC.Core.TDataStd import TDataStd_Name # type: ignore
    from OCC.Core.Interface import Interface_Static # type: ignore
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader # type: ignore
    from OCC.Core.STEPConstruct import stepconstruct_FindEntity # type: ignore
    from OCC.Core.StepRepr import StepRepr_RepresentationItem # type: ignore
    from OCC.Core.TCollection import TCollection_HAsciiString, TCollection_ExtendedString # type: ignore
    # Import TopAbs for shape types
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_ShapeEnum # type: ignore
    # Import TopExp for exploring shapes
    from OCC.Core.TopExp import TopExp_Explorer # type: ignore


except ImportError as e:
    # This error will be caught by the main script, but logging it here helps diagnose
    logging.getLogger("CADAnalyzer.cad_reader").critical(f"Failed to import OCC.* modules: {e}. Is pythonocc-core installed correctly?")
    # Re-raise the error to ensure the main script knows about the failure
    raise ImportError(f"pythonocc-core import failed in cad_reader: {e}") from e


# Module-specific logger
logger = logging.getLogger("CADAnalyzer.cad_reader")

# --- Helper function to get shape name (Using NameTool Instance) ---
def _get_component_name(label: TDF_Label, doc: TDocStd_Document) -> Optional[str]:
    """
    Attempts to extract the name associated with a TDF_Label using XCAFDoc tools.
    Tries NameTool().GetName() first, then falls back to checking TDataStd_Name attribute.

    Args:
        label: The TDF_Label to get the name for.
        doc: The TDocStd_Document containing the XCAF data.

    Returns:
        The name as a string if found, otherwise None.
    """
    try:
        if label.IsNull():
            return None

        label_tag = label.Tag() # Use Tag() for logging identifier

        # --- Try using NameTool instance ---
        try:
            # Get the NameTool instance associated with the document's main label
            name_tool = XCAFDoc.XCAFDoc_DocumentTool.NameTool(doc.Main()) # Get NameTool instance
            if name_tool is not None:
                name_string = TCollection_ExtendedString()
                # Call GetName on the name_tool INSTANCE
                if name_tool.GetName(label, name_string):
                    name = name_string.ToExtString()
                    if name: # Ensure name is not empty
                        logger.debug(f"Found component name using NameTool.GetName(): '{name}' for label tag {label_tag}")
                        return name
                    else:
                        logger.debug(f"NameTool.GetName() returned True but name string was empty for label tag {label_tag}.")
                        # Continue to fallback...
                else:
                     logger.debug(f"NameTool.GetName() failed for label tag {label_tag}.")
                     # Continue to fallback...
            else:
                 logger.debug("Failed to get NameTool instance.")
                 # Continue to fallback...

        except AttributeError as ae:
             # Catch specific error if NameTool doesn't exist on XCAFDoc_DocumentTool
             if 'NameTool' in str(ae):
                 logger.debug("XCAFDoc_DocumentTool has no NameTool attribute in this version. Skipping NameTool check.")
             else:
                 logger.warning(f"AttributeError attempting to use NameTool.GetName() for label tag {label_tag}: {ae}", exc_info=False)
             # Continue to fallback...
        except Exception as nt_e:
             logger.warning(f"Error attempting to use NameTool.GetName() for label tag {label_tag}: {nt_e}", exc_info=False)
             # Continue to fallback...


        # --- Fallback: Try getting name via TDataStd_Name attribute ---
        # This might still fail if the previous AttributeError on TDataStd_Name.Get was correct
        logger.debug(f"Falling back to TDataStd_Name attribute check for label tag {label_tag}.")
        try:
            name_attrib_check = TDataStd_Name()
            # Use the static Get method for TDataStd_Name attribute retrieval
            if TDataStd_Name.Get(label, name_attrib_check):
                 if not name_attrib_check.IsNull():
                     name = name_attrib_check.Get().ToExtString()
                     if name:
                         logger.debug(f"Found component name via TDataStd_Name.Get(): '{name}' for label tag {label_tag}")
                         return name
                     else:
                         logger.debug(f"TDataStd_Name.Get() found attribute but name was empty for label tag {label_tag}.")
                         return None
            logger.debug(f"TDataStd_Name.Get() failed to find attribute for label tag {label_tag}.")
            return None # Name not found by either method
        except AttributeError as ae:
             # Catch specific error if TDataStd_Name.Get doesn't exist
             if 'Get' in str(ae) and 'TDataStd_Name' in str(ae):
                 logger.debug(f"TDataStd_Name.Get() static method not available for label tag {label_tag}. Cannot get name via attribute.")
                 return None
             else:
                 logger.warning(f"AttributeError during TDataStd_Name fallback for label tag {label_tag}: {ae}")
                 return None
        except Exception as fallback_e:
             logger.warning(f"Generic error during TDataStd_Name fallback for label tag {label_tag}: {fallback_e}", exc_info=True)
             return None

    except Exception as e:
        # Use Tag() in error message if available
        label_tag_str = str(label.Tag()) if not label.IsNull() else "Null Label"
        logger.warning(f"Generic error extracting component name for label tag {label_tag_str}: {e}", exc_info=True)
        return None


# --- Recursive function to extract solid components ---
def _extract_solid_components_recursive(
    label: TDF_Label,
    doc: TDocStd_Document, # Pass the document for name retrieval
    shape_tool: XCAFDoc.XCAFDoc_ShapeTool, # Pass tool as argument
    components_list: List[Tuple[str, TopoDS_Shape]],
    processed_labels: Set[int], # Use label tag for tracking
    level: int = 0 # Recursion depth for logging
):
    """
    Recursively traverses the XCAF structure starting from a label,
    extracting shapes that are Solids, CompSolids, or Compounds containing Solids.

    Args:
        label: The current TDF_Label to process.
        doc: The TDocStd_Document for accessing tools like NameTool.
        shape_tool: The XCAF ShapeTool instance.
        components_list: List to append found (name, shape) tuples to.
        processed_labels: Set to keep track of processed labels (using Tag()) to avoid cycles/duplicates.
        level: Current recursion depth (for logging indentation).
    """
    indent = "  " * level

    if label.IsNull():
        logger.debug(f"{indent}Skipping null label.")
        return

    # Use Tag() instead of HashCode() or Entry()
    label_tag = label.Tag() # Get the integer tag

    # Avoid infinite loops in case of cyclic references and re-processing
    # Use a copy of the set for the check to handle recursion correctly
    current_path_processed = processed_labels.copy()
    if label_tag in current_path_processed:
        logger.debug(f"{indent}Skipping already processed label tag in this path: {label_tag}")
        return
    current_path_processed.add(label_tag) # Mark this label as visited in this path


    # Get component name early for logging
    component_name = _get_component_name(label, doc) or f"Unnamed_Component_{label_tag}"
    logger.debug(f"{indent}Processing Label Tag: {label_tag} - Potential Name: '{component_name}'")

    # Check if this label represents an assembly or a simple shape reference
    is_assembly = shape_tool.IsAssembly(label)
    is_reference = shape_tool.IsReference(label)
    is_simple_shape = shape_tool.IsSimpleShape(label) # Explicitly check if it's marked as simple
    shape = shape_tool.GetShape(label) # Get shape associated with this label
    shape_type = shape.ShapeType() if not shape.IsNull() else TopAbs_ShapeEnum.TopAbs_UNKNOWN

    logger.debug(f"{indent}  Label Tag: {label_tag} - IsAssembly: {is_assembly}, IsReference: {is_reference}, IsSimpleShape: {is_simple_shape}, ShapeType: {shape_type}")

    # --- Case 1: It's an Assembly ---
    if is_assembly:
        logger.debug(f"{indent}  Label is Assembly. Exploring components...")
        component_labels = TDF_LabelSequence()
        shape_tool.GetComponents(label, component_labels)
        num_sub_components = component_labels.Length()
        logger.debug(f"{indent}  Found {num_sub_components} sub-component labels.")
        for i in range(1, num_sub_components + 1):
            sub_label = component_labels.Value(i)
            logger.debug(f"{indent}  Recursing into sub-component {i}/{num_sub_components} (Label Tag: {sub_label.Tag()})")
            # Pass the copied set for this recursion level
            _extract_solid_components_recursive(sub_label, doc, shape_tool, components_list, current_path_processed, level + 1)

    # --- Case 2: It's a Reference to another shape ---
    # This typically represents an instance of a part or sub-assembly
    elif is_reference:
        ref_label = TDF_Label()
        if shape_tool.GetReferredShape(label, ref_label):
            ref_label_tag = ref_label.Tag()
            logger.debug(f"{indent}  Label is Reference to Label Tag: {ref_label_tag}.")
            # Get the name from the *referencing* label (instance name)
            instance_name = component_name # Already retrieved above
            # Get the shape from the *referenced* label (definition geometry)
            ref_shape = shape_tool.GetShape(ref_label)
            ref_name = _get_component_name(ref_label, doc) # Get name of the definition
            ref_shape_type = ref_shape.ShapeType() if not ref_shape.IsNull() else TopAbs_ShapeEnum.TopAbs_UNKNOWN
            logger.debug(f"{indent}  Referenced Label Tag: {ref_label_tag}, Ref Name: '{ref_name}', Ref ShapeType: {ref_shape_type}")

            # If the referenced shape is a solid, add it using the instance name
            if not ref_shape.IsNull() and ref_shape_type in [TopAbs_SOLID, TopAbs_COMPSOLID]:
                 logger.info(f"{indent}  Adding Solid/CompSolid from Reference: Instance Name='{instance_name}', Definition Name='{ref_name}' (Ref Tag: {ref_label_tag})")
                 components_list.append((instance_name, ref_shape))
                 # Do NOT mark referenced label as processed here - allow multiple instances
            elif not ref_label.IsNull():
                 # If the referenced shape is not a solid itself (e.g., it's another assembly or compound),
                 # recurse into the *referenced* label to find solids within it.
                 logger.debug(f"{indent}  Referenced shape is not Solid/CompSolid. Recursing into referenced label tag {ref_label_tag}...")
                 # Pass the copied set for this recursion level
                 _extract_solid_components_recursive(ref_label, doc, shape_tool, components_list, current_path_processed, level + 1)
            else:
                 logger.warning(f"{indent}  Referenced label is Null, cannot recurse.")
        else:
             logger.warning(f"{indent}  Label is Reference, but GetReferredShape failed for label tag {label_tag}.")


    # --- Case 3: It's potentially a simple shape (or wasn't identified as Assembly/Reference) ---
    elif not shape.IsNull():
        # Check if the shape itself is a SOLID, COMPSOLID, or COMPOUND
        logger.debug(f"{indent}  Label is not Assembly/Reference. Checking shape type {shape_type}...")
        if shape_type in [TopAbs_SOLID, TopAbs_COMPSOLID]:
            logger.info(f"{indent}  Adding Solid/CompSolid: '{component_name}'")
            components_list.append((component_name, shape))
        elif shape_type == TopAbs_COMPOUND:
            # If it's a compound, explore its children to find solids
            logger.debug(f"{indent}  Shape is a Compound. Exploring for Solids...")
            solid_explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            compound_solid_index = 0
            solids_found_in_compound = 0
            while solid_explorer.More():
                compound_solid_index += 1
                solid_shape = solid_explorer.Current()
                if not solid_shape.IsNull():
                     solids_found_in_compound += 1
                     # Attempt to find a more specific name for the solid within the compound
                     solid_label = shape_tool.FindShape(solid_shape, False) # Try finding label for this specific solid
                     specific_name = _get_component_name(solid_label, doc) if not solid_label.IsNull() else None
                     solid_name = specific_name or f"{component_name}_Solid_{compound_solid_index}"
                     logger.info(f"{indent}    Adding Solid from Compound: '{solid_name}'")
                     components_list.append((solid_name, solid_shape))
                solid_explorer.Next()
            if solids_found_in_compound == 0:
                 logger.debug(f"{indent}  Compound shape did not contain any TopAbs_SOLID children.")
        else:
            logger.debug(f"{indent}  Shape type {shape_type} is not Solid, CompSolid, or Compound containing Solids. Skipping.")
    else:
        # This case might occur for labels that are purely structural without direct geometry
        logger.debug(f"{indent}  Label tag {label_tag} has no associated shape (IsNull). Skipping.")


# --- Main STEP Reading Function ---
def read_step_file(file_path: str) -> List[Tuple[str, TopoDS_Shape]]:
    """
    Reads a STEP file, extracts individual solid components and their names,
    traversing the assembly structure using XCAF.

    Args:
        file_path: Path to the STEP file (.step or .stp).

    Returns:
        A list of tuples, where each tuple contains the component name (str)
        and the TopoDS_Shape object for a solid component. Returns an empty list
        if the file cannot be read or no solid components are found.

    Raises:
        FileNotFoundError: If the file_path does not exist or is not a file.
        ImportError: If essential pythonocc-core modules cannot be imported.
        Exception: For other unexpected errors during file processing.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"STEP file not found or is not a file: {file_path}")

    logger.info(f"Attempting to read STEP file: {file_path}")

    # Initialize XCAF document
    try:
        doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    except Exception as doc_init_e:
         logger.error(f"Failed to initialize TDocStd_Document: {doc_init_e}", exc_info=True)
         return []

    # Initialize the XCAF reader
    reader = STEPCAFControl_Reader()

    # Read the file
    status = reader.ReadFile(file_path)
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        logger.error(f"STEPCAFControl_Reader failed to read file '{file_path}'. Status code: {status}")
        return []
    logger.info("STEP file read successfully. Transferring shapes...")

    # Transfer shapes to the XCAF document
    transfer_success = reader.Transfer(doc)
    if not transfer_success:
        logger.error(f"Failed to transfer shapes from reader to XCAF document for file '{file_path}'.")
        return []
    logger.info("Shapes transferred to XCAF document. Extracting components...")

    # Get tools
    shape_tool = XCAFDoc.XCAFDoc_DocumentTool.ShapeTool(doc.Main())

    # Get root labels (representing top-level items like assemblies or parts)
    root_labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(root_labels)
    num_roots = root_labels.Length()
    logger.info(f"Found {num_roots} root labels in XCAF structure.")

    # List to store final solid components (name, shape)
    solid_components: List[Tuple[str, TopoDS_Shape]] = []
    # Set to track processed labels during recursion using Tag()
    processed_labels_global: Set[int] = set() # Use a global set to avoid adding truly duplicate shapes

    # Process each root label recursively
    for i in range(1, num_roots + 1):
        root_label = root_labels.Value(i)
        logger.info(f"--- Processing Root Label {i}/{num_roots} (Tag: {root_label.Tag()}) ---")
        # Pass the global processed set to the recursive function
        _extract_solid_components_recursive(root_label, doc, shape_tool, solid_components, processed_labels_global)

    # --- Post-processing: Check for duplicates based on shape hash ---
    # This adds an extra layer of checking, although the label tracking should handle most cases.
    final_unique_components: List[Tuple[str, TopoDS_Shape]] = []
    processed_shape_hashes: Set[int] = set()
    duplicate_count = 0
    for name, shape in solid_components:
        # Use TShape hash for uniqueness check
        shape_hash = shape.__hash__()
        if shape_hash not in processed_shape_hashes:
            final_unique_components.append((name, shape))
            processed_shape_hashes.add(shape_hash)
        else:
            duplicate_count += 1
            logger.debug(f"Removed duplicate shape instance for component named '{name}' (Shape Hash: {shape_hash})")

    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate shape instances based on shape hash.")


    if not final_unique_components:
        logger.warning(f"No solid components extracted from '{file_path}' after traversing assembly structure.")

    logger.info(f"Finished component extraction. Found {len(final_unique_components)} unique solid components.")
    return final_unique_components


# Example usage (for testing purposes)
if __name__ == '__main__':
    # Configure logging for standalone testing
    # Set level to DEBUG to see the detailed traversal logs
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s [%(levelname)s] %(message)s')

    # Create a dummy STEP file path for testing (replace with a real path)
    test_file = "path/to/your/assembly_test_file.step" # <--- CHANGE THIS

    if not os.path.exists(test_file):
        logger.error(f"Test file '{test_file}' not found. Please update the path in the script.")
    else:
        try:
            extracted_components = read_step_file(test_file)
            if extracted_components:
                logger.info(f"\nSuccessfully extracted {len(extracted_components)} solid components:")
                for idx, (name, shape) in enumerate(extracted_components):
                    logger.info(f"  {idx+1}: Name='{name}', Shape Type={shape.ShapeType()}")
            else:
                logger.warning("No solid components were extracted.")
        except FileNotFoundError as fnf_error:
            logger.error(fnf_error)
        except ImportError as imp_error:
            logger.error(f"ImportError: {imp_error}. Ensure pythonocc-core is installed.")
        except Exception as ex:
            logger.error(f"An unexpected error occurred: {ex}", exc_info=True)

