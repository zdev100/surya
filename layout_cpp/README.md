# LayoutApp - C++ Layout Detection Application Guide

This document provides instructions on how to compile, run, and understand the `LayoutApp` C++ application. This application is designed **solely for layout detection** on an image using an ONNX model. It produces a JSON output detailing the detected layout polygons and a visualized image highlighting these regions.

It is intended for C++ developers who wish to compile and use this application, potentially with their own layout detection ONNX models converted from compatible PyTorch projects.

## Prerequisites

1.  **C++ Compiler**: A C++17 compliant compiler (e.g., GCC, Clang, MSVC).
2.  **CMake**: Version 3.10 or higher.
3.  **ONNX Runtime**:
    *   Download and extract the ONNX Runtime pre-built binaries for your platform (Windows, Linux, macOS) from the [ONNX Runtime GitHub Releases](https://github.com/microsoft/onnxruntime/releases).
    *   Set the environment variable `ONNXRUNTIME_ROOT_DIR` to the path of the extracted ONNX Runtime directory (e.g., `C:\onnxruntime-win-x64-1.16.0` or `/opt/onnxruntime-linux-x64-1.16.0`). The CMake build process uses this variable to find the necessary headers and libraries. This is the recommended way.
4.  **OpenCV**:
    *   Install OpenCV (e.g., version 4.x) for your platform. You can download it from [OpenCV Releases](https://opencv.org/releases/) or install it via a package manager (e.g., `sudo apt-get install libopencv-dev` on Debian/Ubuntu, `brew install opencv` on macOS).
    *   Ensure that OpenCV is findable by CMake. This might involve setting the `OpenCV_DIR` CMake variable (e.g., `-DOpenCV_DIR=/path/to/opencv/build` during CMake configuration) to the directory containing `OpenCVConfig.cmake` (e.g., `/usr/local/lib/cmake/opencv4` or `C:\opencv\build\x64\vc15\lib`) or ensuring OpenCV's bin directory is in your system's PATH.
5.  **nlohmann/json**:
    *   This is a header-only library. The simplest way is to download `json.hpp` from the [nlohmann/json GitHub Releases](https://github.com/nlohmann/json/releases) and place it in a location accessible to your compiler's include paths (e.g., within the `layout_cpp` directory or a system include path).
    *   Alternatively, on systems like Debian/Ubuntu, you can install it via `sudo apt-get install nlohmann-json3-dev`. The `CMakeLists.txt` expects it to be findable. If you place `json.hpp` directly into the project, you might need to adjust `target_include_directories` in `CMakeLists.txt` if it's not found by default.

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
    *   **Windows (MSVC - Visual Studio)**:
        ```bash
        cmake .. -G "Visual Studio 16 2019" -A x64 
        ```
        (Adjust the generator name `Visual Studio 16 2019` and platform `-A x64` as needed for your Visual Studio version and target architecture).
        Ensure `ONNXRUNTIME_ROOT_DIR` is set as an environment variable. If OpenCV is not found, you may need to add `-DOpenCV_DIR=C:/path/to/opencv/build`.
    *   **Linux/macOS (Makefiles)**:
        ```bash
        cmake .. 
        ```
        (CMake will try to find compatible generators like Makefiles).
        Ensure `ONNXRUNTIME_ROOT_DIR` is set as an environment variable. If OpenCV is not found, you may need to add `-DOpenCV_DIR=/path/to/opencv/build`.
    *   **General Tip**: If CMake has trouble finding dependencies, explicitly setting them via `-D<VAR_NAME>=/path/to/dependency` (e.g., `-DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime`, `-DOpenCV_DIR=/path/to/opencv/build`) during the CMake configuration step is the most direct solution.

4.  **Build the Executable**:
    After successful configuration, build the project from the `build` directory:
    ```bash
    cmake --build . --config Release
    ```
    (For single-configuration generators like Makefiles on Linux/macOS, `--config Release` might be ignored or you can simply run `make`).
    For Visual Studio, this builds the Release configuration. To build Debug, use `--config Debug`.

    The executable `LayoutApp` (or `LayoutApp.exe` on Windows) will be created in the `build/bin/` directory (or `build/bin/Release/` for Visual Studio Release builds).

## Running the Application

The program requires paths to an input image and an ONNX layout detection model.

**Command-Line Syntax:**

```bash
./LayoutApp <path_to_image> <path_to_layout_model.onnx> [-v]
```
Or on Windows (assuming the executable is in `bin/Release` relative to the `build` directory):
```cmd
.\bin\Release\LayoutApp.exe <path_to_image> <path_to_layout_model.onnx> [-v]
```

**Arguments:**

*   `<path_to_image>`: Path to the input image file (e.g., `../static/images/sample.jpg`).
*   `<path_to_layout_model.onnx>`: Path to the ONNX model file for layout detection.
*   `[-v]` (Optional): Enable verbose logging to the console for more detailed output during processing.

**Example:**

```bash
# From the 'build' directory on Linux/macOS
./bin/LayoutApp ../static/images/test_image.png ../models/dummy_layout_model.onnx -v
```

### Note on the ONNX Layout Model:

*   The C++ application attempts to load and run the specified ONNX layout model. Therefore, the `.onnx` file **must exist** at the given path.
*   **For initial testing without a real model**: You can create a dummy/empty file with the `.onnx` extension (e.g., `touch dummy_layout_model.onnx`). The program will likely fail during model loading or inference if the file is not a valid ONNX model, but this can help test file path handling and basic program flow.
*   **Placeholder Parsing Logic**: The C++ code in `layout_parser.cpp` for parsing the ONNX model's output is currently **placeholder logic**. It's designed to make the program run end-to-end and produce some geometric data (dummy polygons). **This data will not be meaningful until you adapt the parsing logic to the specific output structure of your converted ONNX layout model.** Refer to `pytorch_to_onnx_guide.md` for general model conversion advice.

## Expected Outputs

Upon successful execution, the program will generate two files in the same directory as the input image:

1.  **`*_output.json`**:
    *   A JSON file containing the detected layout polygons and image metadata.
    *   Example: If the input image is `path/to/my_image.jpg`, the output will be `path/to/my_image_output.json`.
    *   The structure includes:
        *   `image_path`, `image_width`, `image_height`
        *   `layout_polygons`: A list of polygons, where each polygon is a list of `[x, y]` coordinate pairs (currently generated by placeholder logic).
        *   `version`: Application version string.
        *   `raw_model_output_info` (if verbose mode was active and output tensor info was captured).

2.  **`*_visualization.png`**:
    *   An image file where the detected layout polygons (placeholder) are drawn in green on top of the original image.
    *   Example: If the input image is `path/to/my_image.jpg`, the output will be `path/to/my_image_visualization.png`.

The program will also print the paths to these generated files to the console.

## Verifying Basic Success

1.  **Compilation**: The program compiles successfully without errors.
2.  **Execution**: The program runs without crashing when provided with valid paths for the image and the ONNX model file (even if the ONNX file is a dummy placeholder for initial tests).
3.  **Output Files**: The `*_output.json` and `*_visualization.png` files are created in the expected location.
4.  **Console Output**: The console shows messages indicating the steps being performed (especially with the `-v` flag) and confirms the paths to the output files.

This setup allows for iterative development and testing. The most crucial next step for meaningful output is to adapt the ONNX model output parsing in `layout_cpp/layout_parser.cpp` to your specific layout detection model.

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
