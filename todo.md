# TODO Checklist: CAD Assembly Collision Analysis Script

## Phase 1: Project Setup & Core Dependencies

-   [ ] **Project Structure:** Create the directory structure (`cad_collision_analyzer/`, `tests/`).
-   [ ] **Virtual Environment:** Set up a Python virtual environment (e.g., `venv` or `conda`).
-   [ ] **Initial Dependencies:** Install `pythonocc-core`, `numpy`, `openpyxl`, `pytest`, `trimesh` (add to `requirements.txt`).
-   [ ] **Basic Logging:** Set up basic logging configuration (e.g., in `main.py` or a dedicated utility).

## Phase 2: Configuration Module (`config_reader.py`)

-   [ ] **Implement `load_config` function:** Read JSON, handle file paths.
-   [ ] **Implement Default Values:** Provide defaults if file missing/malformed.
-   [ ] **Implement Key Checks:** Handle missing "directions" or "num_samples" keys (exit).
-   [ ] **Implement Type Validation:** Check data types for directions and num_samples (exit on error).
-   [ ] **Add Docstrings & Type Hints:** Document the function thoroughly.
-   [ ] **Write Unit Tests (`test_config_reader.py`):**
    -   [ ] Test loading valid config.
    -   [ ] Test non-existent config file (defaults).
    -   [ ] Test malformed JSON (defaults).
    -   [ ] Test missing "directions" key (exit).
    -   [ ] Test missing "num_samples" key (exit).
    -   [ ] Test invalid data types (exit).
-   [ ] **Run `pytest tests/test_config_reader.py`**

## Phase 3: CAD Reading Module (`cad_reader.py`)

-   [ ] **Implement `read_step_file` function:** Use `pythonocc-core` STEPControl_Reader.
-   [ ] **Implement Component Extraction:** Iterate shapes, get `TopoDS_Shape`.
-   [ ] **Implement Name Extraction:** Use XDE tools (`TDocStd_Document`, `XCAFDoc_DocumentTool`, etc.).
-   [ ] **Implement Fallback Naming:** Generate unique IDs if names are missing.
-   [ ] **Implement Error Handling:**
    -   [ ] Handle `FileNotFoundError`.
    -   [ ] Handle STEP read errors (`ValueError`).
    -   [ ] Log CAD parsing/naming errors to `cad_parsing_errors.log` and continue with successfully parsed components.
-   [ ] **Add Docstrings & Type Hints:** Document the function thoroughly.
-   [ ] **Write Unit Tests (`test_cad_reader.py`):**
    -   [ ] Test loading a valid simple STEP file.
    -   [ ] Test loading a valid assembly STEP file.
    -   [ ] Test non-existent file path (`FileNotFoundError`).
    -   [ ] Test invalid file format (`ValueError`).
    -   [ ] Test STEP file with missing names (check generated IDs).
    -   [ ] Test logging for parsing errors (mocking might be needed).
-   [ ] **Run `pytest tests/test_cad_reader.py`**

## Phase 4: Mesh Conversion Module (`mesh_converter.py`)

-   [ ] **Implement `convert_shape_to_mesh` function:** Use `BRepMesh_IncrementalMesh`.
-   [ ] **Implement Vertex/Face Extraction:** Get data from `BRep_Tool.Triangulation`.
-   [ ] **Implement `trimesh.Trimesh` Creation:** Construct the mesh object.
-   [ ] **Define `MeshConversionError` Exception:** Create custom exception class.
-   [ ] **Implement Error Handling:**
    -   [ ] Log mesh failures to `mesh_generation_errors.log`.
    -   [ ] Raise `MeshConversionError` on failure (to halt execution).
-   [ ] **Add Docstrings & Type Hints:** Document the function thoroughly.
-   [ ] **Write Unit Tests (`test_mesh_converter.py`):**
    -   [ ] Test successful conversion of a simple shape.
    -   [ ] Test correct `trimesh` object creation (vertices, faces).
    -   [ ] Test error handling (simulate failure, check log and exception).
-   [ ] **Run `pytest tests/test_mesh_converter.py`**

## Phase 5: 3D Collision Module (`collision_detector_3d.py`)

-   [ ] **Implement `check_3d_collision` function:** Use `BRepExtrema_DistShapeShape`.
-   [ ] **Implement Distance Check:** Compare result with tolerance (0.0001).
-   [ ] **Implement Error Handling:** Log warning and return 0 if distance calculation fails.
-   [ ] **Add Docstrings & Type Hints:** Document the function thoroughly.
-   [ ] **Write Unit Tests (`test_collision_detector_3d.py`):**
    -   [ ] Test separated shapes (> tolerance).
    -   [ ] Test touching/overlapping shapes (<= tolerance).
    -   [ ] Test shapes exactly at tolerance.
    -   [ ] Test shapes slightly above/below tolerance.
    -   [ ] (Optional) Test distance calculation failure.
-   [ ] **Run `pytest tests/test_collision_detector_3d.py`**

## Phase 6: Projection Module (`projector.py`)

-   [ ] **Implement `calculate_geometric_center` function:** Use `trimesh.Trimesh.centroid`. Handle errors.
-   [ ] **Implement `project_mesh_onto_plane` function:**
    -   [ ] Use `trimesh` projection method (e.g., `mesh.projected` or manual calculation if needed).
    -   [ ] Handle input validation (mesh not empty, direction vector not zero).
    -   [ ] Return `np.ndarray` of 2D vertices.
-   [ ] **Implement Error Handling:**
    -   [ ] Raise `ValueError` for zero direction vector.
    * [ ] Raise `RuntimeError` and log if projection fails (to halt execution).
    * [ ] Handle empty mesh input gracefully (return empty array, log warning).
-   [ ] **Add Docstrings & Type Hints:** Document functions thoroughly.
-   [ ] **Write Unit Tests (`test_projector.py`):**
    -   [ ] Test `calculate_geometric_center` with known mesh.
    -   [ ] Test `project_mesh_onto_plane` onto XY, YZ, and tilted planes.
    -   [ ] Test projection of simple known shapes.
    -   [ ] Test error handling (empty mesh, zero direction).
    -   [ ] Test projection failure raises `RuntimeError`.
-   [ ] **Run `pytest tests/test_projector.py`**

## Phase 7: Sampling Module (`sampler.py`)

-   [ ] **Implement `sample_polygon_points` function:**
    -   [ ] Handle `num_samples <= 0`.
    -   [ ] Handle input polygon with < 3 vertices.
    -   [ ] Implement sampling logic (e.g., Rejection Sampling using `is_point_in_polygon`).
    -   [ ] Combine original vertices with sampled points.
-   [ ] **Implement Error Handling:** Handle invalid input, log warnings if sampling yields too few points.
-   [ ] **Add Docstrings & Type Hints:** Document the function thoroughly.
-   [ ] **(Cleanup):** Remove temporary `is_point_in_polygon` if added earlier.
-   [ ] **Write Unit Tests (`test_sampler.py`):**
    -   [ ] Test with `num_samples = 0`.
    -   [ ] Test with positive `num_samples`. Check output shape.
    -   [ ] Test sampled points are within bounds (basic check).
    -   [ ] Test with invalid input polygon.
-   [ ] **Run `pytest tests/test_sampler.py`**

## Phase 8: 2D Interpolation Module (`interpolator_2d.py`)

-   [ ] **Implement `is_point_in_polygon` function:** Use Ray Casting algorithm. Handle edge/vertex cases.
-   [ ] **Implement `check_2d_interpolation` function:** Iterate sampled points, call `is_point_in_polygon`. Return 1 on first hit, 0 otherwise.
-   [ ] **Implement Error Handling:** Handle empty inputs, polygons < 3 vertices.
-   [ ] **Add Docstrings & Type Hints:** Document functions thoroughly.
-   [ ] **Write Unit Tests (`test_interpolator_2d.py`):**
    -   [ ] Test `is_point_in_polygon` (inside, outside, vertex, edge, concave).
    -   [ ] Test `check_2d_interpolation` (hit, no hit, empty inputs).
-   [ ] **Run `pytest tests/test_interpolator_2d.py`**

## Phase 9: Excel Output Module (`excel_writer.py`)

-   [ ] **Implement `ExcelWriter` class:** `__init__`, `save`.
-   [ ] **Implement `add_metadata_sheet` method:** Write key-value pairs.
-   [ ] **Implement `add_component_names_sheet` method:** Write Index, Component Name columns.
-   [ ] **Implement `add_matrix_sheet` method:** Write row/column headers (names) and N x N matrix data.
-   [ ] **Implement Error Handling:** Basic `try...except` around file saving.
-   [ ] **Add Docstrings & Type Hints:** Document the class and methods.
-   [ ] **Write Unit Tests (`test_excel_writer.py`):**
    -   [ ] Test workbook creation and saving.
    -   [ ] Test sheet creation (Metadata, Component Names, Matrix).
    -   [ ] Test content verification (read back data).
-   [ ] **Run `pytest tests/test_excel_writer.py`**

## Phase 10: Main Script Orchestration (`main.py`)

-   [ ] **Implement Argument Parsing:** Use `argparse` for `step_file` and `config`.
-   [ ] **Implement Initial Setup:** Timestamp, logging config.
-   [ ] **Integrate `config_reader`:** Load configuration, handle exit on error.
-   [ ] **Integrate `cad_reader`:** Read STEP file, handle `FileNotFoundError`, `ValueError`.
-   [ ] **Integrate `mesh_converter`:** Loop through components, mesh them, handle `MeshConversionError` (log, exit). Filter component list.
-   [ ] **Integrate `excel_writer` (Initial):** Instantiate writer, write Metadata, Component Names.
-   [ ] **Integrate `collision_detector_3d`:** Loop pairs, calculate 3D collisions, store in matrix.
-   [ ] **Integrate `excel_writer` (3D Matrix):** Add "contact matrix" sheet.
-   [ ] **Implement 2D Interpolation Loop (Sequential):**
    -   [ ] Loop through directions from config.
    -   [ ] Loop through component pairs (i, j).
    -   [ ] Integrate `projector` (center calculation, projection for i and j). Handle `RuntimeError` (log, exit).
    -   [ ] Integrate `sampler` (sample points for i).
    -   [ ] Integrate `interpolator_2d` (check interpolation).
    -   [ ] Store results in per-direction matrix.
    -   [ ] Integrate `excel_writer` (Direction Matrix): Add sheet for each direction.
-   [ ] **Implement Final Save:** Call `writer.save()` after all loops.
-   [ ] **Add Execution Time Logging.**

## Phase 11: Parallel Processing & Final Error Handling (`main.py`)

-   [ ] **Refactor Direction Loop for Parallelism:**
    -   [ ] Define `process_direction` worker function.
    -   [ ] Use `threading` module.
    -   [ ] Use shared dictionary/queue for results.
    -   [ ] Use `threading.Event` for critical projection error signaling.
    -   [ ] Modify worker error handling (log, set event, don't exit).
    -   [ ] Implement main thread logic (create/start/join threads).
    -   [ ] Check error event after joining.
    -   [ ] Write results to Excel from main thread *only if no critical error*.
-   [ ] **Review and Finalize Error Handling:** Ensure all specified error conditions (Spec 5) are handled correctly across all modules and the main script (logging, exiting, continuing as required).

## Phase 12: Testing & Refinement

-   [ ] **Run All Unit Tests:** `pytest tests/`. Ensure all pass.
-   [ ] **Write Integration Tests:** Create tests in `tests/` that run the main script (or parts of its flow) with simple test assemblies and configurations.
    -   [ ] Test the full pipeline for a small assembly, one direction.
    -   [ ] Test the pipeline with multiple directions (sequential first, then parallel).
-   [ ] **Write System Tests:**
    -   [ ] Test with various real-world STEP files (simple, complex, assemblies).
    -   [ ] Test with different `config.json` files (various directions, sample counts).
    -   [ ] Test edge cases (e.g., 2 components, many components, no directions in config handled by defaults).
-   [ ] **Test Error Handling Scenarios:** Manually trigger specified errors (invalid paths, bad files, bad config) and verify correct script behavior (logs, exit codes, default values).
-   [ ] **Verify Excel Output:** Manually inspect generated Excel files for several test cases. Check metadata, names, matrix dimensions, headers, and some sample values for correctness.
-   [ ] **Performance Testing:** (Optional but recommended) Run script on large assemblies, use `cProfile` or other profilers to identify bottlenecks. Apply optimizations if necessary.
-   [ ] **Code Review & Refactoring:**
    -   [ ] Ensure PEP 8 compliance (use linters like `flake8` or `pylint`).
    -   [ ] Ensure comprehensive docstrings for all modules, classes, functions.
    -   [ ] Ensure type hints are used consistently.
    -   [ ] Refactor for clarity, efficiency, and maintainability.
-   [ ] **Update `README.md`:** Add instructions on setup, usage, dependencies, configuration.

## Phase 13: Finalization

-   [ ] **Final Code Freeze.**
-   [ ] **Tag Release (e.g., using Git).**
-   [ ] **Package (Optional):** Prepare for distribution if needed.
