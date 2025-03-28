import logging
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import openpyxl
    from openpyxl.workbook import Workbook
    from openpyxl.worksheet.worksheet import Worksheet
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter
    OPENPYXL_INSTALLED = True
except ImportError:
    OPENPYXL_INSTALLED = False
    # Define dummy types for type hinting if import fails
    Workbook = type("Workbook", (), {})
    Worksheet = type("Worksheet", (), {})
    Font = type("Font", (), {}) # type: ignore
    print("WARNING: openpyxl not found. Excel writing functionality will fail.")

# --- Logging Setup ---
excel_logger = logging.getLogger(__name__)
# Add a basic handler if none are configured
if not excel_logger.handlers:
    excel_logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler() # Output warnings/errors to console
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    excel_logger.addHandler(ch)


class ExcelWriter:
    """
    Handles the creation and writing of analysis results to an Excel (.xlsx) file.

    Attributes:
        filename (str): The path to the Excel file to be created.
        wb (Workbook): The openpyxl Workbook object.
    """

    def __init__(self, filename: str):
        """
        Initializes the ExcelWriter.

        Args:
            filename: The name (including path) for the output Excel file.

        Raises:
            ImportError: If openpyxl is not installed.
        """
        if not OPENPYXL_INSTALLED:
            raise ImportError("openpyxl library is required but not installed.")

        self.filename: str = filename
        self.wb: Workbook = openpyxl.Workbook()

        # Remove the default sheet created by openpyxl
        if "Sheet" in self.wb.sheetnames:
            try:
                default_sheet = self.wb["Sheet"]
                self.wb.remove(default_sheet)
            except KeyError:
                pass # Ignore if somehow not found despite check
            except Exception as e:
                 excel_logger.warning(f"Could not remove default 'Sheet': {e}")


    def _apply_header_style(self, cell):
        """Applies bold font style to a cell."""
        if Font and cell: # Check if Font was imported and cell is valid
             cell.font = Font(bold=True)

    def _adjust_column_widths(self, ws: Worksheet):
         """Adjusts column widths based on content (simple approach)."""
         if not Worksheet: return # Skip if openpyxl types not available
         dims = {}
         for row in ws.rows:
             for cell in row:
                 if cell.value:
                     # Compare length of value to current max length for column
                     dims[cell.column_letter] = max((dims.get(cell.column_letter, 0), len(str(cell.value))))
         for col, value in dims.items():
             # Add a little padding (e.g., +2)
             ws.column_dimensions[col].width = value + 2


    def add_metadata_sheet(self, metadata: Dict[str, Any]):
        """
        Adds a sheet named "Metadata" containing key-value pairs.

        Args:
            metadata: A dictionary where keys are parameter names and
                      values are the corresponding metadata values.
        """
        if not Workbook or not Worksheet: return # Skip if openpyxl missing

        try:
            ws: Worksheet = self.wb.create_sheet(title="Metadata")

            # Write Headers
            header1 = ws.cell(row=1, column=1, value="Parameter")
            header2 = ws.cell(row=1, column=2, value="Value")
            self._apply_header_style(header1)
            self._apply_header_style(header2)

            # Write Data
            current_row = 2
            for key, value in metadata.items():
                ws.cell(row=current_row, column=1, value=str(key)) # Ensure key is string
                # Openpyxl handles basic types like str, int, float, datetime
                # For complex objects, convert to string
                if isinstance(value, (list, dict, tuple, set)):
                     val_to_write = str(value)
                else:
                     val_to_write = value
                ws.cell(row=current_row, column=2, value=val_to_write)
                current_row += 1

            self._adjust_column_widths(ws)

        except Exception as e:
            excel_logger.error(f"Failed to add Metadata sheet. Error: {e}", exc_info=True)


    def add_component_names_sheet(self, component_names: List[str]):
        """
        Adds a sheet named "Component Names" listing components and their indices.

        Args:
            component_names: A list of component name strings.
        """
        if not Workbook or not Worksheet: return # Skip if openpyxl missing

        try:
            ws: Worksheet = self.wb.create_sheet(title="Component Names")

            # Write Headers
            header1 = ws.cell(row=1, column=1, value="Index")
            header2 = ws.cell(row=1, column=2, value="Component Name")
            self._apply_header_style(header1)
            self._apply_header_style(header2)

            # Write Data
            for idx, name in enumerate(component_names):
                ws.cell(row=idx + 2, column=1, value=idx)
                ws.cell(row=idx + 2, column=2, value=name)

            self._adjust_column_widths(ws)

        except Exception as e:
            excel_logger.error(f"Failed to add Component Names sheet. Error: {e}", exc_info=True)


    def add_matrix_sheet(
        self,
        sheet_name: str,
        matrix: np.ndarray,
        component_names: List[str]
    ):
        """
        Adds a sheet containing an N x N matrix with component names as headers.

        Args:
            sheet_name: The desired name for the sheet (max 31 chars, avoid invalid chars).
            matrix: A NumPy array (N x N) containing the matrix data (e.g., 0s and 1s).
            component_names: A list of N component name strings for headers.
        """
        if not Workbook or not Worksheet: return # Skip if openpyxl missing

        # Basic sanitization and length check for sheet name
        if not isinstance(sheet_name, str) or not sheet_name:
             excel_logger.error("Invalid sheet_name provided for matrix sheet.")
             return
        # Replace common invalid characters and limit length
        safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in ('_', '-'))
        safe_sheet_name = safe_sheet_name[:31]
        if not safe_sheet_name: # Handle case where name becomes empty after sanitizing
             safe_sheet_name = "MatrixSheet"
        if safe_sheet_name != sheet_name:
             excel_logger.warning(f"Sanitized sheet name from '{sheet_name}' to '{safe_sheet_name}'.")


        try:
            # --- Input Validation ---
            if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                excel_logger.error(f"Invalid matrix provided for sheet '{safe_sheet_name}'. Matrix must be a square 2D NumPy array.")
                return
            n = matrix.shape[0]
            if len(component_names) != n:
                excel_logger.error(f"Dimension mismatch for sheet '{safe_sheet_name}'. Matrix size ({n}x{n}) does not match number of component names ({len(component_names)}).")
                return

            # --- Create Sheet and Headers ---
            ws: Worksheet = self.wb.create_sheet(title=safe_sheet_name)

            # Write Column Headers (Component Names starting B1)
            for j, name in enumerate(component_names):
                col_header_cell = ws.cell(row=1, column=j + 2, value=name)
                self._apply_header_style(col_header_cell)

            # Write Row Headers (Component Names starting A2)
            for i, name in enumerate(component_names):
                row_header_cell = ws.cell(row=i + 2, column=1, value=name)
                self._apply_header_style(row_header_cell)

            # --- Write Matrix Data ---
            for i in range(n):
                for j in range(n):
                    # Ensure value is a standard Python type if needed
                    value = matrix[i, j]
                    if isinstance(value, np.generic): # Convert numpy types
                         value = value.item()
                    # Write value - openpyxl handles int, float, bool, str
                    ws.cell(row=i + 2, column=j + 2, value=value)

            self._adjust_column_widths(ws)

        except Exception as e:
             excel_logger.error(f"Failed to add matrix sheet '{safe_sheet_name}'. Error: {e}", exc_info=True)


    def save(self):
        """
        Saves the Excel workbook to the specified filename.

        Logs an error if saving fails (e.g., due to permissions).
        """
        if not Workbook: return # Skip if openpyxl missing

        try:
            # Ensure directory exists before saving
            output_dir = os.path.dirname(self.filename)
            if output_dir and not os.path.exists(output_dir):
                 excel_logger.info(f"Creating output directory: {output_dir}")
                 os.makedirs(output_dir, exist_ok=True)

            self.wb.save(self.filename)
            # Use main logger if available, else use module logger
            main_logger = logging.getLogger('cad_collision_analyzer.main')
            if main_logger.hasHandlers():
                 main_logger.info(f"Excel file saved successfully: '{self.filename}'")
            else:
                 excel_logger.info(f"Excel file saved successfully: '{self.filename}'")
        except (IOError, PermissionError) as pe:
            excel_logger.error(f"Permission denied or I/O error saving Excel file '{self.filename}'. Error: {pe}", exc_info=False) # No need for full stack trace here
        except Exception as e:
            excel_logger.error(f"Failed to save Excel file '{self.filename}'. Unexpected error: {e}", exc_info=True)


# Example Usage (Optional)
if __name__ == '__main__':
    if not OPENPYXL_INSTALLED:
        print("Error: openpyxl is required to run this example.")
    else:
        import os
        import datetime
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        print("Running ExcelWriter example...")
        output_file = "example_output.xlsx"
        try:
             writer = ExcelWriter(output_file)

             # Add metadata
             meta = {
                 "Input File": "assembly_example.step",
                 "Timestamp": datetime.datetime.now(), # Use current time
                 "Number of Components": 3,
                 "Tolerance": 0.01
             }
             writer.add_metadata_sheet(meta)

             # Add component names
             names = ["WidgetA", "GadgetB", "ThingamajigC"]
             writer.add_component_names_sheet(names)

             # Add a contact matrix
             contact_mat = np.array([
                 [1, 0, 1],
                 [0, 1, 0],
                 [1, 0, 1]
             ])
             writer.add_matrix_sheet("Contact Matrix", contact_mat, names)

             # Add another matrix (e.g., interpolation results)
             interp_mat = np.array([
                 [0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]
             ])
             # Test sheet name sanitization
             writer.add_matrix_sheet("Interpolation Results [Test?/]", interp_mat, names)

             # Save the file
             writer.save()

             # Check if file exists
             if os.path.exists(output_file):
                 print(f"File '{output_file}' created successfully.")
             else:
                 print(f"File '{output_file}' was not created (check logs for errors).")

        except ImportError:
              print("openpyxl not installed.", file=sys.stderr)
        except Exception as ex:
              print(f"An error occurred: {ex}", file=sys.stderr)
