import os
import logging
from typing import List, Tuple, Optional

# --- PythonOCC Core Imports ---
# Using try-except block for graceful failure if pythonocc-core is not installed
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ReturnStatus
    from OCC.Core.TopoDS import TopoDS_Shape
    # Imports needed for XDE (e.g., names, assembly structure)
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool_ShapeTool
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader # Use CAF reader for XDE features
    from OCC.Core.TDataStd import TDataStd_Name
    from OCC.Core.TCollection import TCollection_AsciiString
    from OCC.Core.TDF import TDF_Label, TDF_LabelSequence

    PYTHONOCC_INSTALLED = True
except ImportError:
    PYTHONOCC_INSTALLED = False
    # Define dummy types for type hinting if import fails
    TopoDS_Shape = type("TopoDS_Shape", (), {})
    IFSelect_ReturnStatus = type("IFSelect_ReturnStatus", (), {})
    TDF_Label = type("TDF_Label", (), {})


# --- Logging Setup ---
LOG_FILE = 'cad_parsing_errors.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Process messages at INFO level and above

# Prevent adding handlers multiple times if the module is reloaded
if not logger.handlers:
    # File handler for warnings and errors related to parsing
    try:
        # Ensure log directory exists if needed
        # log_dir = os.path.dirname(LOG_FILE)
        # if log_dir and not os.path.exists(log_dir):
        #    os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(LOG_FILE, mode='a') # Append mode
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        # Fallback if file handler can't be created
        # Use basicConfig which might go to stderr or overwrite existing handlers
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__) # Re-get logger after basicConfig
        logger.error(f"Could not configure file logging to {LOG_FILE}. Error: {e}", exc_info=True)

    # Optional: Add a console handler for INFO/DEBUG during development
    # If main.py also configures basicConfig, this might be redundant or conflict
    # Only add if logger has no handlers AND basicConfig hasn't been called elsewhere.
    # if not logging.getLogger().hasHandlers(): # Check root logger
    #    ch = logging.StreamHandler()
    #    ch.setLevel(logging.INFO)
    #    ch_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    #    ch.setFormatter(ch_formatter)
    #    logger.addHandler(ch)


def _get_component_name(label: 'TDF_Label', index: int) -> str:
    """
    Attempts to retrieve the name associated with a TDF_Label using XDE tools.
    Falls back to a default name if not found or if an error occurs.

    Args:
        label: The TDF_Label associated with the shape.
        index: An index used for generating a default name.

    Returns:
        The extracted or generated component name.
    """
    default_name = f"Component_Shape_{index}"
    if not PYTHONOCC_INSTALLED: return default_name # Should not happen if checked before

    name_attribute = TDataStd_Name()
    try:
        # Check if the label itself has the TDataStd_Name attribute
        if label.FindAttribute(TDataStd_Name.GetID(), name_attribute):
            t_name = name_attribute.Get()
            # Check if name is not empty
            if t_name and t_name.Length() > 0:
                name_str = t_name.ToCString()
                if name_str: # Ensure CString conversion didn't result in None/empty
                    return name_str
                else:
                    logger.debug(f"Found name attribute for label but ToCString() returned empty/None. Using default: {default_name}")
            else:
                 logger.debug(f"Found name attribute for label but it was empty. Using default: {default_name}")
        else:
             # If not on the label, check if it's associated via reference (common in assemblies)
             # This part might require more complex XCAF graph traversal depending on STEP structure.
             # For simplicity, we stick to the direct attribute for now.
             # logger.debug(f"No direct name attribute found for label. Using default: {default_name}")
             pass # Just use default if not found directly

    except Exception as e:
        # Log as warning as we fall back to default name
        logger.warning(f"Error retrieving name for label. Error: {e}. Using default: {default_name}", exc_info=True)

    return default_name


def read_step_file(file_path: str) -> List[Tuple[str, TopoDS_Shape]]:
    """
    Reads a STEP file and extracts top-level component shapes and their names.

    Uses pythonocc-core's XDE (Extended Data Exchange) capabilities via
    STEPCAFControl_Reader to handle assembly structures and associated names.

    Logs errors related to file reading or component processing to
    'cad_parsing_errors.log' instead of raising exceptions for those cases.

    Args:
        file_path: Path to the STEP (.stp or .step) file.

    Returns:
        A list of tuples, where each tuple contains:
        (component_name: str, component_shape: TopoDS_Shape).
        Returns an empty list if the file cannot be read or contains no valid shapes.

    Raises:
        FileNotFoundError: If the file_path does not exist or is not a file.
        ImportError: If pythonocc-core is not installed correctly.
    """
    if not PYTHONOCC_INSTALLED:
        raise ImportError("pythonocc-core is not installed or could not be imported.")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The specified STEP file does not exist or is not a file: {file_path}")

    # Use STEPCAFControl_Reader to work with XDE document structure
    reader = STEPCAFControl_Reader()

    # Read the STEP file
    status: IFSelect_ReturnStatus = reader.ReadFile(file_path)

    if status != IFSelect_RetDone:
        # Log error to the dedicated file handler
        logger.error(f"Failed to read STEP file: {file_path}. Status Code: {status}")
        return []

    # Transfer the STEP file structure to an XDE document
    doc = TDocStd_Document(TCollection_AsciiString("xde-doc"))
    try:
        if not reader.Transfer(doc):
            logger.error(f"Failed to transfer STEP data to XDE document for file: {file_path}")
            return []
    except Exception as e:
         logger.error(f"Exception during STEP data transfer for file: {file_path}. Error: {e}", exc_info=True)
         return []


    # Use XCAF tools to access shapes and names
    shape_tool: XCAFDoc_DocumentTool_ShapeTool = XCAFDoc_DocumentTool_ShapeTool(doc.Main())
    labels = TDF_LabelSequence()

    # Get labels of top-level ("free") shapes
    try:
        shape_tool.GetFreeShapes(labels)
    except Exception as e:
        logger.error(f"Exception while getting free shapes from XDE document for file: {file_path}. Error: {e}", exc_info=True)
        return []


    results: List[Tuple[str, TopoDS_Shape]] = []
    num_labels = labels.Length()

    if num_labels == 0:
        # Log as info/warning, not error, as file might be valid but empty
        logger.warning(f"No top-level shapes found in STEP file: {file_path}")
        return []

    # Log info message to console/main log if configured
    console_logger = logging.getLogger('cad_collision_analyzer.main') # Or appropriate logger name
    console_logger.info(f"Found {num_labels} top-level component candidate(s) in {file_path}. Processing...")


    for i in range(num_labels):
        # Check TDF_Label type if needed, assuming pythonocc handles this
        label: TDF_Label = labels.Value(i + 1) # Labels are 1-based index
        shape: Optional[TopoDS_Shape] = None
        component_name: str = ""

        try:
            # Retrieve the shape associated with the label
            # Check if the label corresponds to a shape first
            if not shape_tool.IsShape(label):
                logger.debug(f"Label at index {i} is not a shape. Skipping.")
                continue

            shape = shape_tool.GetShape(label)

            if shape is None or shape.IsNull():
                # Log to cad_parsing_errors.log
                logger.warning(f"Skipping null or invalid shape for label at index {i} in file: {file_path}")
                continue

            # Get the component name using the helper
            component_name = _get_component_name(label, i)

            results.append((component_name, shape))
            # logger.info(f"Successfully processed shape: '{component_name}'") # Optional: log success

        except Exception as e:
            # Catch unexpected errors during shape/name processing for a specific label
            # Log to cad_parsing_errors.log
            logger.error(f"Error processing label at index {i} in file {file_path}. Error: {e}", exc_info=True)
            continue # Skip this component and try the next one

    if not results:
         logger.warning(f"Finished processing {file_path}, but no valid components with shapes were extracted and added.")
    # else: # Logged by main script now
         # console_logger.info(f"Successfully extracted {len(results)} components from {file_path}.")


    return results

# Example Usage (Optional)
if __name__ == '__main__':
    # This block will only run if the script is executed directly
    # Requires pythonocc-core and a real STEP file
    import sys # Needed for stderr access here

    # Set up basic logging just for this example run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    test_file = "example.step" # <<< PUT A REAL STEP FILE PATH HERE FOR TESTING >>>

    print(f"Attempting to read STEP file: {test_file}")
    print(f"Parsing errors/warnings will be logged to: {LOG_FILE}")

    if not PYTHONOCC_INSTALLED:
        print("Error: pythonocc-core is required to run this example.", file=sys.stderr)
    elif os.path.exists(test_file):
        try:
            components = read_step_file(test_file)
            if components:
                print("\nExtracted Components:")
                for name, shape in components:
                    try:
                         shape_type = shape.ShapeType()
                         # Assuming ShapeType() returns an enum or similar representable value
                         type_str = str(shape_type)
                    except:
                         type_str = "Unknown"
                    print(f"- Name: '{name}', Shape Type: {type_str}")
            else:
                print("No components extracted or file read failed (check logs).")
        except FileNotFoundError as fnf_err:
            print(f"Error: {fnf_err}", file=sys.stderr)
        except ImportError as imp_err:
            print(f"Error: {imp_err}", file=sys.stderr)
        except Exception as ex:
            print(f"An unexpected error occurred: {ex}", file=sys.stderr)
            logging.getLogger(__name__).error("Unexpected error in main execution block", exc_info=True)
    else:
        print(f"Test file '{test_file}' not found. Please create it or modify the path.", file=sys.stderr)
