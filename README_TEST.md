# Surya C++ Application - Test Program Guide

This document provides instructions on how to compile and run the `SuryaCppApp` test program. This program is designed to process an image using ONNX models for layout detection and text recognition, producing a JSON output and a visualized image.

## Prerequisites

1.  **C++ Compiler**: A C++17 compliant compiler (e.g., GCC, Clang, MSVC).
2.  **CMake**: Version 3.10 or higher.
3.  **ONNX Runtime**:
    *   Download and extract the ONNX Runtime pre-built binaries for your platform (Windows, Linux, macOS) from the [ONNX Runtime GitHub Releases](https://github.com/microsoft/onnxruntime/releases).
    *   Set the environment variable `ONNXRUNTIME_ROOT_DIR` to the path of the extracted ONNX Runtime directory (e.g., `C:/onnxruntime-win-x64-1.16.0` or `/path/to/onnxruntime-linux-x64-1.16.0`). The CMake build process uses this variable to find the necessary headers and libraries.
4.  **OpenCV**:
    *   Install OpenCV (e.g., version 4.x) for your platform. You can download it from [OpenCV Releases](https://opencv.org/releases/) or install it via a package manager (e.g., `sudo apt-get install libopencv-dev` on Debian/Ubuntu).
    *   Ensure that OpenCV is findable by CMake. This might involve setting the `OpenCV_DIR` CMake variable to the directory containing `OpenCVConfig.cmake` (e.g., `/usr/local/lib/cmake/opencv4` or `C:/opencv/build/x64/vc15/lib`) or ensuring OpenCV's bin directory is in your system's PATH.
5.  **nlohmann/json**:
    *   This is a header-only library. The simplest way is to download `json.hpp` from the [nlohmann/json GitHub Releases](https://github.com/nlohmann/json/releases) and place it in a location accessible to your compiler's include paths.
    *   Alternatively, on systems like Debian/Ubuntu, you can install it via `sudo apt-get install nlohmann-json3-dev`. The `CMakeLists.txt` expects it to be findable. If you place `json.hpp` directly into the project (e.g., in a `lib/` or `external/` directory), you might need to adjust `target_include_directories` in `CMakeLists.txt` if it's not found.

## Compilation

1.  **Clone the Repository** (if you haven't already):
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Build Directory**:
    It's good practice to build out-of-source.
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure with CMake**:
    From the `build` directory:
    ```bash
    # For Windows (MSVC - Visual Studio)
    cmake .. -G "Visual Studio 16 2019" -A x64 
    # (Adjust generator and platform as needed for your VS version)
    # Ensure ONNXRUNTIME_ROOT_DIR is set as an environment variable.

    # For Linux/macOS (Makefiles)
    cmake .. 
    # (CMake will try to find compatible generators like Makefiles)
    # Ensure ONNXRUNTIME_ROOT_DIR is set.
    ```
    *   If CMake has trouble finding ONNX Runtime or OpenCV, you might need to specify their paths directly using `-D`:
        ```bash
        cmake .. -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime -DOpenCV_DIR=/path/to/opencv/build
        ```

4.  **Build the Executable**:
    After successful configuration, build the project:
    ```bash
    # For Makefiles (Linux/macOS)
    cmake --build .
    # or just 'make'

    # For Visual Studio
    cmake --build . --config Release
    # (Or open the generated .sln file in Visual Studio and build)
    ```
    The executable `SuryaCppApp` (or `SuryaCppApp.exe` on Windows) will be created in the `build/bin/` or `build/bin/Release/` directory.

## Running the Test Program

The program requires paths to an input image and two ONNX models (layout detection and text recognition).

**Command-Line Syntax:**

```bash
./SuryaCppApp <path_to_image> <path_to_layout_model.onnx> <path_to_recognition_model.onnx> [-v]
```
Or on Windows:
```cmd
SuryaCppApp.exe <path_to_image> <path_to_layout_model.onnx> <path_to_recognition_model.onnx> [-v]
```

**Arguments:**

*   `<path_to_image>`: Path to the input image file (e.g., `input.jpg`, `data/my_scan.png`).
*   `<path_to_layout_model.onnx>`: Path to the ONNX model file for layout detection.
*   `<path_to_recognition_model.onnx>`: Path to the ONNX model file for text recognition.
*   `[-v]` (Optional): Enable verbose logging to the console for more detailed output during processing.

**Example:**

```bash
./SuryaCppApp ../static/images/test_image.png ../models/dummy_layout_model.onnx ../models/dummy_recognition_model.onnx -v
```

### Note on ONNX Models:

*   The C++ application attempts to load and run these ONNX models. Therefore, the specified `.onnx` files **must exist** at the given paths, even if they are just placeholders.
*   **For initial testing without real models**: You can create dummy/empty files with the `.onnx` extension (e.g., `touch dummy_layout_model.onnx`). The program will likely fail during model loading or inference if the files are not valid ONNX models, but this can help test the file path handling and basic program flow.
*   The placeholder parsing logic in the C++ code is very simple and designed to make the program run end-to-end. It will produce some data for the output files, but this data won't be meaningful until actual model output parsing is implemented based on the true structure of your converted ONNX models.

## Expected Outputs

Upon successful execution, the program will generate two files in the same directory as the input image:

1.  **`*_output.json`**:
    *   A JSON file containing the (placeholder) detected layout polygons and (placeholder) recognized text lines.
    *   Example: If the input image is `path/to/my_image.jpg`, the output will be `path/to/my_image_output.json`.
    *   The structure includes:
        *   `image_path`, `image_width`, `image_height`
        *   `layout_polygons`: A list of polygons, where each polygon is a list of `[x, y]` coordinate pairs.
        *   `ocr_lines`: A list of text line objects, each containing `text`, `polygon` (coordinates), and `confidence`.

2.  **`*_visualization.png`**:
    *   An image file where the detected layout polygons are drawn in green, and recognized text (placeholder) is drawn in blue on top of the original image.
    *   Example: If the input image is `path/to/my_image.jpg`, the output will be `path/to/my_image_visualization.png`.

The program will also print the paths to these generated files to the console.

## Verifying Basic Success

1.  **Compilation**: The program compiles successfully without errors.
2.  **Execution**: The program runs without crashing when provided with valid (even if dummy) paths for the image and ONNX models.
3.  **Output Files**: The `_output.json` and `_visualization.png` files are created in the expected location.
4.  **Console Output**: The console shows messages indicating the steps being performed (especially with the `-v` flag) and confirms the paths to the output files.

This setup allows for iterative development and testing as the actual ONNX model conversion and output parsing logic are implemented.All requested tasks for this subtask have been completed.

**Summary of Actions:**

1.  **Enhanced `main.cpp` for Robustness and Clarity:**
    *   **Error Handling**:
        *   Added `fileExists` helper function.
        *   Implemented checks for the existence of the input image file and both ONNX model files. Errors are printed, and the program exits if files are not found.
        *   Try-catch blocks were already present and are deemed sufficient for this stage.
    *   **Command-Line Arguments**:
        *   Updated argument parsing to accept 3 required arguments (image path, layout model path, recognition model path) and one optional `-v` flag for verbosity.
        *   A detailed usage message is printed if arguments are incorrect.
    *   **Output**:
        *   The program prints paths to the generated JSON and visualization image files.
        *   A verbosity flag (`-v`) was added, and corresponding `if (verbose)` print statements were integrated into `main.cpp`, `ONNXInferenceManager`, `processLayout`, and `processOCR` for more detailed console output.

2.  **Placeholder Logic for ONNX Output Parsing:**
    *   **`layout_parser.cpp` (`processLayout`):**
        *   After `session.Run()`, the code now includes placeholder logic to interpret the output tensor's shape.
        *   If the shape vaguely resembles a segmentation mask or direct coordinates, it generates a few dummy polygons (e.g., dividing the image into blocks or fixed boxes). This ensures `LayoutData` is populated with some geometric data.
        *   Verbose logging indicates the assumed shape and actions taken.
    *   **`ocr_parser.cpp` (`processOCR`):**
        *   After `session.Run()` for each image patch:
        *   Placeholder logic "decodes" the output by generating a string like `"TxtRgnX"` based on the region index and assigns a random placeholder confidence.
        *   `OCRData` is populated with this placeholder text and the corresponding polygon from the layout stage.
        *   Verbose logging indicates the assumed output shape and generated text.
    *   The goal of making the test program runnable end-to-end with some data in output files has been achieved with these placeholders.

3.  **Documentation for Test Program Usage (`README_TEST.md`):**
    *   A new `README_TEST.md` file was created.
    *   It includes sections on:
        *   **Prerequisites**: Compiler, CMake, ONNX Runtime (with `ONNXRUNTIME_ROOT_DIR` setup), OpenCV (with `OpenCV_DIR` setup if needed), and nlohmann/json.
        *   **Compilation**: Step-by-step instructions for creating a build directory, configuring with CMake (for Windows, Linux/macOS, and troubleshooting common library path issues), and building the `SuryaCppApp` executable.
        *   **Running the Test Program**: Command-line syntax, argument descriptions (including the optional `-v` flag), and an example.
        *   **Note on ONNX Models**: Emphasizes that model files must exist and how to use dummy files for initial testing. Clarifies that current parsing logic is placeholder.
        *   **Expected Outputs**: Description of the `*_output.json` and `*_visualization.png` files.
        *   **Verifying Basic Success**: How to confirm the program is working at a basic level (compiles, runs, creates outputs).

All deliverables for this subtask are complete. The C++ application is now more robust as a test program, includes placeholder data generation from simulated ONNX outputs, and is documented for compilation and execution.
