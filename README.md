# STL_test

# CAD Assembly Collision Analysis Tool

## Overview

This Python script analyzes potential collisions and line-of-sight intersections between components within a CAD assembly defined in a STEP file. It performs an initial 3D collision check and then, for a set of user-defined directions, projects components onto 2D planes to check for intersections based on sampled points. This type of analysis is crucial in manufacturing and design processes to verify assembly feasibility, check for potential interferences during operation or assembly sequences, and ensure clearances are met.

The results, including 3D contact information and 2D interpolation matrices for each specified direction, are compiled into a structured Excel file for easy review and documentation.

## Features

* **STEP File Input:** Reads CAD assembly geometry from `.step` or `.stp` files.
* **Component Identification:** Extracts top-level components and their names (or generates unique IDs).
* **Mesh Conversion:** Converts CAD geometry to mesh representations using `pythonocc-core` and `trimesh`.
* **3D Collision Detection:** Performs pairwise checks for direct contact or near-contact (tolerance definable, default 0.0001 units) between components in the original 3D assembly.
* **Configurable 2D Projection Analysis:**
    * Projects component meshes onto planes orthogonal to user-specified directions.
    * Samples points (vertices + uniform random) on one component's projection.
    * Checks if sampled points from one component fall within the projected area of another using the Ray Casting algorithm (2D interpolation check).
* **Parallel Processing:** Utilizes threading to process multiple projection directions concurrently for improved performance.
* **Detailed Excel Output:** Generates an `.xlsx` file containing:
    * Metadata (input file, timestamp, parameters).
    * List of component names and their indices.
    * N x N matrix detailing 3D collision results.
    * Separate N x N matrices detailing 2D interpolation results for each specified direction.
* **Error Handling & Logging:** Includes robust error handling for file I/O, configuration issues, CAD parsing, meshing failures, and projection problems. Logs critical errors to dedicated files (`cad_parsing_errors.log`, `mesh_generation_errors.log`).

## Prerequisites

* **Python:** Version 3.8 or higher recommended.
* **Operating System:** Tested primarily on Linux/macOS. Windows compatibility depends heavily on the `pythonocc-core` installation method. Using Conda is often recommended for easier cross-platform installation of `pythonocc-core`.
* **Python Libraries:**
    * `pythonocc-core`: For reading STEP files and performing core CAD operations.
    * `numpy`: For numerical operations, especially vector math and matrices.
    * `trimesh`: For mesh representation and geometric operations (projection, centroid).
    * `openpyxl`: For writing data to Excel `.xlsx` files.
    * `pytest` (for development/testing): For running unit and integration tests.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd cad-collision-analyzer
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    * **Using pip:**
        ```bash
        pip install -r requirements.txt
        ```
        *Note: Installing `pythonocc-core` via pip can sometimes be challenging depending on your OS and existing compilers. If you encounter issues, try the Conda method below.*

    * **Using Conda (Recommended for `pythonocc-core`):**
        If you use Conda, it's often easier to install `pythonocc-core` first:
        ```bash
        conda install -c conda-forge pythonocc-core
        ```
        Then, install the remaining dependencies using pip within your conda environment:
        ```bash
        pip install numpy trimesh openpyxl pytest
        # (Update requirements.txt accordingly if using this method)
        ```

## Configuration

The analysis behavior is controlled by a `config.json` file located in the project's root directory.

```json
{
  "directions": [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1]
  ],
  "num_samples": 100
}
