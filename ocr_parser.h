#ifndef OCR_PARSER_H
#define OCR_PARSER_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp> // For cv::Mat
#include <onnxruntime_cxx_api.h> // For Ort::Session

// Forward declare LayoutData or include layout_parser.h
// For simplicity, let's assume LayoutData is defined in a way that's accessible,
// or we define a simplified version here if it's only for passing bounding boxes.
// #include "layout_parser.h" // If LayoutData is complex and defined there
// Or, ensure the struct definition used in main.cpp is accessible (e.g. in a common header or defined again)
struct LayoutData; // Forward declaration is fine if we only pass by reference/pointer
                   // But if we access its members, we need the full definition.
                   // Let's assume the definition from main.cpp is implicitly available
                   // or would be moved to a common header. For now, this is a placeholder.

// Placeholder structure for OCR data
// This will be expanded later to hold actual recognized text lines and their coordinates.
// This matches the structure in main.cpp, keep it consistent or move definition to a common header.
struct OCRData {
    struct TextLine {
        std::string text;
        std::vector<cv::Point> polygon; // The polygon/bbox of the text line
        float confidence;               // Placeholder for recognition confidence
    };
    std::vector<TextLine> lines;
    std::string raw_output_info; // Placeholder
};

/**
 * @brief Processes cropped image regions (based on layout analysis) using the OCR ONNX model.
 *
 * This function takes the original image and layout data, crops text regions,
 * preprocesses them, runs them through the OCR ONNX session, and (currently)
 * returns placeholder OCR data. The actual parsing of the model's output tensor
 * into recognized text will be implemented later.
 *
 * @param original_image The full original image (cv::Mat).
 * @param layout_results The data obtained from the layout analysis step, containing detected text regions.
 * @param recognition_session A reference to the initialized ONNX Runtime session for the OCR model.
 * @param allocator Ort::AllocatorWithDefaultOptions instance for tensor creation.
 * @param input_node_name The name of the input node of the OCR model.
 * @param output_node_names A vector of C-strings for the names of the output nodes of the OCR model.
 * @param target_ocr_height The target height for preprocessing image patches for the OCR model.
 *                          Width is often variable or padded.
 * @return OCRData structure containing (currently placeholder) recognized text lines.
 */
OCRData processOCR(
    const cv::Mat& original_image,
    const LayoutData& layout_results, // Assuming LayoutData contains polygons/bboxes
    Ort::Session& recognition_session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const char* input_node_name, 
    const std::vector<const char*>& output_node_names,
    bool verbose = false
);

#endif // OCR_PARSER_H
