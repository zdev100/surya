# Surya C++ Application Guide

This document provides instructions on how to compile, run, and understand the `SuryaCppApp` C++ application. This application processes an image using ONNX models for layout detection and text recognition, producing a JSON output detailing the findings and a visualized image.

It is intended for C++ developers who wish to compile and use this application, potentially with their own ONNX models converted from the original Surya PyTorch project.

## 1. Prerequisites

Before compiling, ensure the following dependencies are installed and correctly configured:

1.  **C++ Compiler**: A C++17 compliant compiler is required.
    *   **Linux**: GCC (g++) or Clang.
    *   **Windows**: Microsoft Visual Studio (MSVC) 2019 or later.
    *   **macOS**: Clang (Xcode).

2.  **CMake**: Version 3.10 or higher. CMake is used to manage the build process.
    *   Download from [cmake.org](https://cmake.org/download/).

3.  **ONNX Runtime**: This library is essential for running the ONNX models.
    *   **Download**: Obtain the pre-built binaries for your specific OS and architecture (e.g., Windows x64, Linux x64) from the [ONNX Runtime GitHub Releases](https://github.com/microsoft/onnxruntime/releases). Choose a version compatible with your models (e.g., 1.10.0 or newer).
    *   **Environment Variable**: After extracting the downloaded archive, you **must** set the `ONNXRUNTIME_ROOT_DIR` environment variable to point to the root directory of the extracted ONNX Runtime.
        *   **Windows Example**: If you extracted to `C:\lib\onnxruntime-win-x64-1.16.0`, set `ONNXRUNTIME_ROOT_DIR=C:\lib\onnxruntime-win-x64-1.16.0`.
        *   **Linux/macOS Example**: If you extracted to `/opt/onnxruntime-linux-x64-1.16.0`, set `export ONNXRUNTIME_ROOT_DIR=/opt/onnxruntime-linux-x64-1.16.0` (add this to your `.bashrc` or `.zshrc` for persistence).
        *   This variable allows CMake's `find_package(ONNXRuntime REQUIRED)` command to locate the necessary header files and libraries.

4.  **OpenCV**: Used for image loading, preprocessing, and visualization. Version 4.x is recommended.
    *   **Download/Install**:
        *   From [OpenCV Releases](https://opencv.org/releases/).
        *   Via a package manager (e.g., `sudo apt-get install libopencv-dev python3-opencv` on Debian/Ubuntu, `brew install opencv` on macOS).
    *   **CMake Configuration**: CMake needs to find your OpenCV installation.
        *   Often, package manager installations are found automatically.
        *   If you built from source or downloaded pre-built binaries, you might need to set the `OpenCV_DIR` environment variable or CMake variable to the directory containing `OpenCVConfig.cmake`. This directory is usually found within your OpenCV installation (e.g., `C:\opencv\build`, `/usr/local/lib/cmake/opencv4`).
        *   **Windows Example**: If OpenCV is installed in `C:\opencv\build`, `OpenCV_DIR` might be `C:\opencv\build`.
        *   **Linux Example**: If installed in `/usr/local`, `OpenCV_DIR` might be `/usr/local/lib/cmake/opencv4`.

5.  **nlohmann/json**: A header-only library for JSON manipulation.
    *   **Download**: Get the `json.hpp` file from the [nlohmann/json GitHub Releases](https://github.com/nlohmann/json/releases) (usually included as a single header file).
    *   **Installation**:
        *   Place `json.hpp` in a location where your compiler can find it (e.g., a project `external/` directory that you add to your include paths, or a system-wide include directory).
        *   Alternatively, on Debian/Ubuntu, you can install it via: `sudo apt-get install nlohmann-json3-dev`. The `CMakeLists.txt` expects the header to be discoverable via standard include paths.

## 2. Building the Application

1.  **Navigate to the C++ Application Directory**:
    Assuming the C++ source code and `CMakeLists.txt` are in a subdirectory (e.g., `cpp_app` within the main project):
    ```bash
    cd path/to/your_project/cpp_app 
    ```
    If the C++ files are at the root of the project, navigate there.

2.  **Create and Navigate to a Build Directory**:
    It is standard practice to perform an out-of-source build.
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure the Build with CMake**:
    From within the `build` directory:

    *   **Linux / macOS (Makefile Generator - Common Case)**:
        ```bash
        # For a Release build (optimized):
        cmake .. -DCMAKE_BUILD_TYPE=Release

        # For a Debug build:
        cmake .. -DCMAKE_BUILD_TYPE=Debug
        ```

    *   **Windows (Visual Studio Generator)**:
        You need to specify the generator for your Visual Studio version and the platform.
        ```bash
        # Example for Visual Studio 2019 (VS 16), 64-bit, Release build:
        cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release

        # Example for Visual Studio 2019 (VS 16), 64-bit, Debug build:
        cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Debug
        ```
        Adjust `"Visual Studio 16 2019"` and `-A x64` according to your Visual Studio version and target architecture.

    *   **Troubleshooting CMake Configuration**:
        If CMake cannot find ONNX Runtime or OpenCV, ensure `ONNXRUNTIME_ROOT_DIR` is correctly set as an environment variable. For OpenCV, you might need to explicitly tell CMake where to find it if it's not in a standard location:
        ```bash
        cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/path/to/opencv_install_dir/share/OpenCV 
        # (The exact path for OpenCV_DIR depends on your OpenCV installation structure)
        ```

4.  **Compile the Application**:
    After CMake has successfully configured the project:
    ```bash
    cmake --build . --config <BUILD_TYPE>
    ```
    Replace `<BUILD_TYPE>` with `Release` or `Debug`, matching what you used during configuration.
    *   **Linux/macOS (Makefiles)**: You can often just run `make` after configuration.
    *   **Windows (Visual Studio)**: This command will invoke MSBuild. Alternatively, you can open the `.sln` file generated in the `build` directory with Visual Studio and build from there.

5.  **Locate the Executable**:
    The compiled executable `SuryaCppApp` (or `SuryaCppApp.exe` on Windows) will typically be found in:
    *   `build/bin/`
    *   Or, for multi-config generators like Visual Studio: `build/bin/Release/` or `build/bin/Debug/`.

## 3. Running the Application

Execute the application from the command line, providing paths to the input image and the two required ONNX models.

**Command-Line Syntax:**
```bash
<path_to_executable>/SuryaCppApp <path_to_image> <path_to_layout_model.onnx> <path_to_recognition_model.onnx> [-v]
```

**Arguments:**

*   `<path_to_executable>`: Path to the compiled `SuryaCppApp` executable (e.g., `./bin/SuryaCppApp` or `.\bin\Release\SuryaCppApp.exe`).
*   `<path_to_image>`: Full path to the input image file (e.g., `../static/images/sample.jpg`).
*   `<path_to_layout_model.onnx>`: Full path to the ONNX model file for layout detection.
*   `<path_to_recognition_model.onnx>`: Full path to the ONNX model file for text recognition.
*   `[-v]` (Optional): Enables verbose logging, providing more detailed output about the application's progress and internal states.

**Example:**
```bash
# On Linux/macOS, from the 'build' directory
./bin/SuryaCppApp ../static/images/test_image.png ../models/layout_model.onnx ../models/recognition_model.onnx -v

# On Windows, from the 'build' directory
.\bin\Release\SuryaCppApp.exe ..\static\images\test_image.png ..\models\layout_model.onnx ..\models\recognition_model.onnx -v
```

## 4. ONNX Models

*   **Origin**: The application is designed to work with ONNX models that have been converted from the PyTorch models used in the original Surya project. For detailed guidance on performing this conversion, please refer to `pytorch_to_onnx_guide.md` (located in the project's root or documentation directory).
*   **Input/Output Names**: The C++ application uses hardcoded input and output node names for the ONNX models (e.g., `"input_image"`, `"output_segmentation_map"`, `"pixel_values"`, `"output_text_sequence"`). These **must match** the names in your converted ONNX models. You may need to inspect your ONNX models (e.g., using Netron) and update these names in `main.cpp` or make them configurable.

*   **CRITICAL CAVEAT - Placeholder Parsing Logic**:
    The current implementation of the C++ application, specifically within `layout_parser.cpp` and `ocr_parser.cpp`, contains **placeholder logic for parsing the output tensors from the ONNX models**.
    *   **Layout Model**: The `processLayout` function currently generates dummy polygons (e.g., fixed boxes or simple divisions of the image) based on heuristics about the output tensor's shape. It does **not** perform actual segmentation mask processing (like `findContours`) or complex bounding box decoding.
    *   **Recognition Model**: The `processOCR` function currently generates placeholder text (e.g., `"TxtRgnX"`) and assigns a random confidence score. It does **not** perform actual CTC decoding or sequence-to-text translation from the model's output logits or probabilities.

    **To use this application with your actual converted ONNX models and obtain meaningful results, you MUST modify the output tensor parsing sections in `layout_parser.cpp` and `ocr_parser.cpp` to correctly interpret the specific structure and data format of your models' outputs.** This will involve understanding the tensor shapes, data types, and the encoding scheme used by your layout detection and text recognition models.

## 5. Outputs of the Application

Upon successful execution, the application generates two primary output files in the same directory as the input image:

1.  **JSON Output (`<image_name>_output.json`)**:
    *   A JSON file detailing the detected layout and recognized text.
    *   **`image_path`**: Path to the processed image.
    *   **`image_width`**, **`image_height`**: Dimensions of the input image.
    *   **`layout_polygons`**: An array of polygons. Each polygon is an array of `[x, y]` coordinate pairs representing its vertices. (Currently generated by placeholder logic).
    *   **`ocr_lines`**: An array of text line objects. Each object contains:
        *   `text`: The recognized text string (Currently placeholder text).
        *   `polygon`: The polygon coordinates for the text line.
        *   `confidence`: The recognition confidence score (Currently a placeholder value).
    *   **`version`**: Internal version string of the C++ application.

2.  **Visualization Image (`<image_name>_visualization.png`)**:
    *   A PNG image file where:
        *   Detected layout polygons are drawn in **green**.
        *   Recognized text lines (placeholder text) are drawn in **blue** near their respective polygons.
    *   This provides a visual confirmation of the (currently placeholder) detection and OCR process.

The application will print the full paths to these generated files to the console.

## 6. Troubleshooting Common Issues

*   **CMake: ONNX Runtime / OpenCV Not Found**:
    *   Ensure `ONNXRUNTIME_ROOT_DIR` is set correctly as an environment variable pointing to your ONNX Runtime installation.
    *   For OpenCV, ensure it's installed correctly and either in the system path or `OpenCV_DIR` is set (either as an environment variable or passed to CMake with `-DOpenCV_DIR=...`).
    *   Double-check paths for typos.

*   **Model Loading Errors (ONNX Runtime Exception)**:
    *   Verify that the paths provided to the application for the `.onnx` model files are correct and that the files exist.
    *   Ensure the `.onnx` files are valid ONNX models. If you are using dummy files for testing, the application might fail at this stage or during inference.
    *   The ONNX Runtime version used for building the application should be compatible with the opset version of the ONNX models.

*   **Incorrect Output / Crashes During Inference**:
    *   If you have started implementing custom parsing logic: This is likely due to mismatches between the expected output tensor structure in your C++ code and the actual structure produced by your ONNX models. Carefully debug your parsing logic, and use tools like Netron to inspect model outputs.
    *   Ensure the input preprocessing in `ImageUtils.cpp` aligns with the preprocessing your models expect.

By following this guide, you should be able to compile and run the `SuryaCppApp`. Remember the critical caveat about the placeholder ONNX output parsing logic, which you will need to adapt for your specific models.
