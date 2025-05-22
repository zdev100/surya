cmake_minimum_required(VERSION 3.10)
project(LayoutApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- ONNX Runtime ---
# Users should set ONNXRUNTIME_ROOT_DIR as an environment variable
# or CMake variable to the root of their ONNX Runtime installation.
# Example: C:/onnxruntime-win-x64-1.16.0
find_package(ONNXRuntime REQUIRED)

if(NOT ONNXRuntime_FOUND)
    message(FATAL_ERROR "ONNX Runtime not found. Please set ONNXRUNTIME_ROOT_DIR.")
else()
    message(STATUS "Found ONNX Runtime: ${ONNXRuntime_INCLUDE_DIRS} | ${ONNXRuntime_LIBRARIES}")
endif()

# --- OpenCV ---
# Users should ensure OpenCV is installed and findable by CMake.
# For example, by setting OpenCV_DIR or ensuring it's in the system PATH.
find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV and ensure it's findable by CMake.")
else()
    message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} | ${OpenCV_LIBS}")
endif()

# --- nlohmann/json ---
# Assuming nlohmann/json is findable by CMake (e.g., installed system-wide or added via add_subdirectory)
# If you've added it as a subdirectory (e.g., in an 'external' folder):
# add_subdirectory(external/json) # Assuming json is the nlohmann/json library
# find_package(nlohmann_json REQUIRED) # This might be needed if it provides CMake config

# If it's just a header-only library included directly in the project (e.g., in a 'lib' folder)
# you might just need to add its path to target_include_directories.
# For simplicity, we'll assume it's findable in standard include paths or via CMAKE_PREFIX_PATH.
# If not, the user will need to ensure it's correctly integrated.
# For now, we just signal that we will use it. Users might need to do:
# sudo apt-get install nlohmann-json3-dev (on Debian/Ubuntu)
# or copy the header into the project.

message(STATUS "Attempting to use nlohmann/json. Ensure it is available in the include paths.")


# --- Executable ---
add_executable(LayoutApp main.cpp
    image_utils.cpp
    onnx_inference.cpp
    layout_parser.cpp
    # ocr_parser.cpp # Removed
    json_utils.cpp
    visualization_utils.cpp)

# --- Include Directories ---
target_include_directories(LayoutApp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} # For our own headers
    # If nlohmann/json is in a subdirectory like 'lib/json/single_include/'
    # ${CMAKE_CURRENT_SOURCE_DIR}/lib/json/single_include 
    ${ONNXRuntime_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    # If using find_package(nlohmann_json):
    # ${nlohmann_json_INCLUDE_DIRS} 
)

# --- Link Libraries ---
target_link_libraries(LayoutApp PRIVATE
    ${ONNXRuntime_LIBRARIES}
    ${OpenCV_LIBS}
)

# --- Windows Specific Settings (Optional but good practice) ---
if(WIN32)
    target_compile_definitions(LayoutApp PRIVATE NOMINMAX UNICODE _UNICODE)
    # Add other Windows-specific settings if needed
endif()

# --- Output Directory (Optional) ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "CMake configuration finished. Executable will be LayoutApp.")
message(STATUS "To build, navigate to your build directory and run 'cmake --build .'")
message(STATUS "Ensure ONNX model file is placed where the application expects it at runtime.")
