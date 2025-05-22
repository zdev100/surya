#ifndef LAYOUT_PARSER_H
#define LAYOUT_PARSER_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp> // For cv::Mat and cv::Point
#include <onnxruntime_cxx_api.h> // For Ort::Session

// Forward declare ONNXInferenceManager if its full definition isn't needed here
// class ONNXInferenceManager; // Or include onnx_inference.h if its members are used directly

// Placeholder structure for layout data
// This will be expanded later to hold actual parsed results (e.g., polygons, bounding boxes)
struct LayoutData {
    // Example: A list of polygons, where each polygon is a list of points.
    // This matches the structure in main.cpp, keep it consistent or move definition to a common header.
    std::vector<std::vector<cv::Point>> detected_polygons;
    // Add other relevant data like confidence scores, types of regions, etc.
    // For now, this is just a placeholder.
    std::string raw_output_info; // Placeholder to store some info about the raw output
};

/**
 * @brief Processes an image using the layout detection ONNX model.
 *
 * This function takes an image, preprocesses it, runs it through the layout
 * detection ONNX session, and (currently) returns placeholder layout data.
 * The actual parsing of the model's output tensor into meaningful LayoutData
 * (e.g., bounding boxes or polygons) will be implemented later.
 *
 * @param image The input image (cv::Mat).
 * @param layout_session A reference to the initialized ONNX Runtime session for the layout model.
 * @param inference_manager A reference to the ONNXInferenceManager for tensor creation and running inference.
 *                          (Or pass allocator and runInference function separately if preferred)
 * @param input_node_name The name of the input node of the layout model.
 * @param output_node_names A vector of C-strings for the names of the output nodes of the layout model.
 * @param target_height The target height for preprocessing the image for the layout model.
 * @param target_width The target width for preprocessing the image for the layout model.
 * @return LayoutData structure containing (currently placeholder) results.
 */
LayoutData processLayout(
    const cv::Mat& image,
    Ort::Session& layout_session,
    Ort::AllocatorWithDefaultOptions& allocator, // Pass allocator for tensor creation
    // Assuming a generic runInference function is available or part of a manager
    // For simplicity, let's assume we'll call a static method or a method on a passed object
    // For now, we'll use ONNXInferenceManager::createTensorFromData and session.Run directly
    const char* input_node_name, 
    const std::vector<const char*>& output_node_names,
    bool verbose = false
);

#endif // LAYOUT_PARSER_H
